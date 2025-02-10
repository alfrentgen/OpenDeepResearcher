import asyncio
import aiohttp
import json
import logging
import re
from duckduckgo_search import DDGS

logger = logging.getLogger('server_logger')
logger.setLevel(logging.INFO)
fileHandler = logging.FileHandler('deepsearch.log', mode='w')
formatter = logging.Formatter("%(asctime)s\n%(message)s\n")
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)

# =======================
# Configuration Constants
# =======================

# Local LLM server settings
IP="192.168.0.143"
LLAMA_CPP_URL = f"http://{IP}:8080/v1/chat/completions"
JSON_REGEX =  r"\{.*\}"

# ============================
# Asynchronous Helper Functions
# ============================

def extract_json(response):
    if json_match := re.search(JSON_REGEX, response, re.DOTALL):
        json_str = json_match.group()
        return json.loads(json_str)
    raise Exception(f"No JSON object was found in {response}")

async def call_llamacpp_async(session, messages):
    """
    Asynchronously call the local llama.cpp server with the provided messages.
    Returns the content of the assistant’s reply.
    """
    headers = {"Content-Type": "application/json"}
    payload = {
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": -1
    }

    try:
        async with session.post(LLAMA_CPP_URL, timeout=0, headers=headers, json=payload) as resp:
            if resp.status == 200:
                result = await resp.json()
                try:
                    return result['choices'][0]['message']['content']
                except (KeyError, IndexError):
                    logger.info("Unexpected llama.cpp response structure: %s", result)
                    return None
            else:
                text = await resp.text()
                logger.info("llama.cpp API error: %s - %s",resp.status,text)
                return None
    except Exception as e:
        logger.info("Error calling llama.cpp: %s", e)
        return None

async def generate_search_queries_async(session, user_query):
    """
    Ask the LLM to produce up to four precise search queries (in Python list format)
    based on the user’s query.
    """

    json_key = "queries"
    prompt = (
        "You are an expert research assistant. Given the user's query, generate up to four distinct, "
        "precise search queries that would help gather comprehensive information on the topic. "
        "Return a JSON object that contains a list of queries precisely in the following format: "
        f'{{"{json_key}" : ["query1", "query2", "query3"]}}'
    )
    messages = [
        {"role": "system", "content": "You are a helpful and precise research assistant."},
        {"role": "user", "content": f"User Query: {user_query}\n\n{prompt}"}
    ]
    logger.info("generate_search_queries_async")

    response = await call_llamacpp_async(session, messages)
    logger.info("response: %s", response)
    search_queries = []
    if response:
        try:
            search_queries = extract_json(response).get(json_key, [])
            logger.info(f"Parsed search query list: {search_queries}")
            if not isinstance(search_queries, list):
                raise Exception(f"Could not parse a search query list the response.")
        except Exception as e:
            logger.info(f"Error parsing search queries: {e}")

    return search_queries

def perform_ddg_search(query):
    """
    Asynchronously perform a DuckDuckGo search for the given query.
    Returns a list of result URLs.
    """
    logger.info("perform_ddg_search")

    try:
        with DDGS() as ddgs:
            results = []
            for result in ddgs.text(query, max_results=5):
                logger.info(f"Search result for the query='{query}':\n{result}")
                results.append(result['href'])
            return results
    except Exception as e:
        print("Error performing search:", e)
        return []

async def fetch_webpage_text_async(session, url):
    """
    Asynchronously retrieve the text content of a webpage using direct HTTP GET.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        async with session.get(url, headers=headers, timeout=20) as resp:
            if resp.status == 200:
                return await resp.text()
            print(f"Failed to fetch {url}: {resp.status}")
            return ""
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""

async def is_page_useful_async(session, user_query, page_text):
    """
    Ask the LLM if the provided webpage content is useful for answering the user's query.
    The LLM must reply with exactly "Yes" or "No".
    """

    info = page_text[:20000]

    json_key = "useful"
    prompt = (
        "You are a critical research evaluator. Given the user's query and the content of a webpage, "
        "determine if the webpage contains information relevant and useful for addressing the query. "
        "Respond with a valid JSON precisely in the following format: "
        f'{{"{json_key}" : "<>"}}'
        'substitute <> with word: "Yes" if the page is useful, or "No" if it is not.'
    )
    messages = [
        {"role": "system", "content": "You are a strict and concise evaluator of research relevance."},
        {"role": "user", "content": f"User Query: {user_query}\n\nWebpage Content (first 20000 characters):\n{info}\n\n{prompt}"}
    ]
    response = await call_llamacpp_async(session, messages)
    logger.info("response: %s", response)
    usefulness = None
    if response:
        try:
            usefulness = extract_json(response).get(json_key, None)
            logger.info("The information can be considered: %s", usefulness)
            if not usefulness:
                raise Exception(f"Could not dermine if the {info} in useful.")
            if usefulness in ["Yes", "No"]:
                return usefulness
            else:
                if "Yes" in usefulness:
                    return "Yes"
                elif "No" in usefulness:
                    return "No"
        except Exception as e:
            logger.info("Error deciding on the information usefulness: %s", e)

    return "No"

async def extract_relevant_context_async(session, user_query, search_query, page_text):
    """
    Given the original query, the search query used, and the page content,
    have the LLM extract all information relevant for answering the query.
    """

    json_key = "relevant"
    prompt = (
        "You are an expert information extractor. Given the user's query, the search query that led to this page, "
        "and the webpage content, extract all pieces of information that are relevant to answering the user's query. "
        "Return only the relevant context as a valid JSON precisely in the following format: "
        f'{{"{json_key}" : "<>"}}'
        ", substitute <> with a plain text string without commentaries."
    )
    messages = [
        {"role": "system", "content": "You are an expert in extracting and summarizing relevant information."},
        {"role": "user", "content": f"User Query: {user_query}\nSearch Query: {search_query}\n\nWebpage Content (first 20000 characters):\n{page_text[:20000]}\n\n{prompt}"}
    ]
    response = await call_llamacpp_async(session, messages)
    logger.info("response: %s", response)
    relevant_ctx = None
    if response:
        try:
            relevant_ctx = extract_json(response).get(json_key, None)
            logger.info("Extracted relevant information: %s", relevant_ctx)
            if not relevant_ctx:
                raise Exception(f"Could not extract relevant information from: {page_text}.", )
        except Exception as e:
            logger.info("Error extracting relevant information: %s", e)

    return relevant_ctx if relevant_ctx else ""

async def process_link(session, link, user_query, search_query):
    """
    Process a single link: fetch its content, judge its usefulness, and extract context if useful.
    """
    logger.info("Fetching content from: %s", link)
    page_text = await fetch_webpage_text_async(session, link)
    if not page_text:
        return None
    useful = await is_page_useful_async(session, user_query, page_text)
    logger.info("Page usefulness for %s: %s", link, useful)
    if useful == "Yes":
        context = await extract_relevant_context_async(session, user_query, search_query, page_text)
        if context:
            logger.info("Extracted context from %s (first 200 chars): %s", link, context[:200])
            return context
    return None

async def get_new_search_queries_async(session, user_query, previous_search_queries, all_contexts):
    """
    Determine if additional search queries are needed and return new queries or empty string.
    """
    context_combined = "\n".join(all_contexts)
    prompt = (
        "You are an analytical research assistant. Based on the original query, the search queries performed so far, "
        "and the extracted contexts from webpages, determine if further research is needed. "
        "If further research is needed, provide up to four new search queries as a Python list. "
        "If no further research is needed, respond with exactly ."
    )
    messages = [
        {"role": "system", "content": "You are a systematic research planner."},
        {"role": "user", "content": f"User Query: {user_query}\nPrevious Search Queries: {previous_search_queries}\n\nExtracted Relevant Contexts:\n{context_combined}\n\n{prompt}"}
    ]
    response = await call_llamacpp_async(session, messages)
    if response:
        cleaned = response.strip()
        if cleaned == "":
            return ""
        try:
            new_queries = eval(cleaned)
            return new_queries if isinstance(new_queries, list) else []
        except Exception as e:
            print("Error parsing new search queries:", e, "\nResponse:", response)
            return []
    return []

async def generate_final_report_async(session, user_query, all_contexts):
    """
    Generate the final comprehensive report using all gathered contexts.
    """
    context_combined = "\n".join(all_contexts)
    prompt = (
        "You are an expert researcher and report writer. Based on the gathered contexts below and the original query, "
        "write a comprehensive, well-structured, and detailed report that addresses the query thoroughly. "
        "Include all relevant insights and conclusions without extraneous commentary."
    )
    messages = [
        {"role": "system", "content": "You are a skilled report writer."},
        {"role": "user", "content": f"User Query: {user_query}\n\nGathered Relevant Contexts:\n{context_combined}\n\n{prompt}"}
    ]
    report = await call_llamacpp_async(session, messages)
    return report if report else "No report generated."

# =========================
# Main Asynchronous Routine
# =========================

async def async_main():
    user_query = input("Enter your research query/topic: ").strip()
    iter_limit_input = input("Enter maximum number of iterations (default 5): ").strip()
    iteration_limit = int(iter_limit_input) if iter_limit_input.isdigit() else 5

    aggregated_contexts = []
    all_search_queries = []
    iteration = 0

    async with aiohttp.ClientSession() as session:
        new_search_queries = await generate_search_queries_async(session, user_query)
        if not new_search_queries:
            print("No initial search queries generated. Exiting.")
            return
        all_search_queries.extend(new_search_queries)

        while iteration < iteration_limit:
            print(f"\n=== Iteration {iteration + 1} ===")

            # Perform searches for all current queries
            #search_tasks = [perform_search_async(query) for query in new_search_queries]
            #search_results = await asyncio.gather(*search_tasks)
            search_results = [perform_ddg_search(query) for query in new_search_queries]

            # Map links to their original search queries
            unique_links = {}
            for idx, links in enumerate(search_results):
                query = new_search_queries[idx]
                for link in links:
                    if link not in unique_links:
                        unique_links[link] = query

            print(f"Found {len(unique_links)} unique links for processing")
            #return #HEREWEGO: debug stop

            # Process all links concurrently
            link_tasks = [process_link(session, link, user_query, unique_links[link]) for link in unique_links]
            link_results = await asyncio.gather(*link_tasks)
            return #HEREWEGO: debug stop

            # Aggregate valid contexts
            iteration_contexts = [res for res in link_results if res]
            aggregated_contexts.extend(iteration_contexts)
            print(f"Added {len(iteration_contexts)} new contexts")

            # Check if more research is needed
            new_search_queries = await get_new_search_queries_async(
                session, user_query, all_search_queries, aggregated_contexts
            )

            if new_search_queries == "":
                print("Research complete according to LLM assessment")
                break
            elif new_search_queries:
                print(f"Generated {len(new_search_queries)} new search queries")
                all_search_queries.extend(new_search_queries)
            else:
                print("No new search queries generated")
                break

            iteration += 1

        # Generate final report
        print("\nGenerating final report...")
        final_report = await generate_final_report_async(session, user_query, aggregated_contexts)
        print("\n==== FINAL REPORT ====\n")
        print(final_report)

def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()
