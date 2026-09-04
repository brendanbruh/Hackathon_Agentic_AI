import os
import asyncio
import requests
from dotenv import load_dotenv
from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm
from playwright.async_api import async_playwright

import sys
import os

# =====================================================================
# 1. WINDOWS UTF-8 ENCODING FIX (Must be the absolute first lines of code)
# =====================================================================
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        # Fallback helper for older Python environments
        import codecs

        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

import asyncio
from dotenv import load_dotenv
from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm
from playwright.async_api import async_playwright

# 2. Load active AWS profile and model from the .env file
load_dotenv(".env")

# Ingest the official AWS Bedrock AgentCore SDK
try:
    from bedrock_agentcore.tools.browser_client import browser_session
except ImportError:
    raise ImportError("Please run: pip install bedrock-agentcore playwright")

# Ensure all operations run in us-east-1 to bypass regional SCP blocks
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")


# =====================================================================
# 3. DEFINE THE ASYNC BROWSER TOOL
# =====================================================================
async def aws_browser_search(query: str) -> str:
    """Search the live web for recent information, news, or current facts
    using the official AWS Bedrock AgentCore cloud-managed browser.

    Args:
        query (str): The search query string or direct URL to look up.
    """
    try:
        print(f"\n[AWS] Initiating remote browser session in {AWS_REGION}...")

        # Open a secure cloud browser session using your active AWS SSO credentials
        with browser_session(AWS_REGION) as client:

            # Retrieve the temporary WebSocket URL and SigV4 authentication headers
            ws_url, headers = client.generate_ws_headers()

            # Start Playwright Async engine
            async with async_playwright() as p:
                print("[AWS] Connecting Playwright over CDP to remote cloud browser...")
                browser = await p.chromium.connect_over_cdp(ws_url, headers=headers)

                # =========================================================
                # FOOLPROOF FIX: Use browser.new_page() directly!
                # This completely bypasses browser.contexts and context.pages lists,
                # automatically creating a clean context and page in one single call.
                # =========================================================
                print("[AWS] Opening a clean remote page...")
                page = await browser.new_page()

                # Handle navigation (either direct website URL or DuckDuckGo query)
                if query.startswith("http://") or query.startswith("https://"):
                    target_url = query
                    print(f"[AWS] Navigating directly to: {target_url}")
                else:
                    target_url = f"https://www.duckduckgo.com/search?q={query}"
                    print(f"[AWS] Searching for: '{query}'")

                # Navigate and wait for the page elements to load
                await page.goto(target_url, wait_until="domcontentloaded", timeout=15000)

                # Scrape all human-readable text from the page body
                raw_text = await page.locator("body").inner_text()

                # Close the browser session
                await browser.close()

                # Truncate content to 4000 characters to prevent context-limit errors
                return raw_text[:4000] if raw_text.strip() else "The page loaded but returned no text."

    except Exception as e:
        print(f"\n[AWS ERROR] Async Browser Tool Failed: {str(e)}\n")
        return f"Error executing AWS Browser Tool: {str(e)}"


# =====================================================================
# 4. INSTANTIATE YOUR GOOGLE ADK AGENT
# =====================================================================
root_agent = Agent(
    model=LiteLlm(model=os.getenv("BEDROCK_MODEL")),
    name='root_agent',
    description='A helpful assistant for user questions.',
    instruction=(
    """
        You are a real-time assistant equipped with a live cloud-browsing tool called use_browser_tool. 
        You have full, real-time access to the internet. "
        Whenever the user asks for current events, news, or to access a specific website (such as Google News), 
        you MUST call 'use_browser_tool' with the search query or the direct URL (e.g. https://news.google.com). 

        CRITICAL RULES:
        1. Never apologize and say 'I cannot browse the web' or 'I do not have real-time access'. You DO have this access via your tool.
        2. If the user asks to look up a site or search for something, run the tool first, and then base your answer strictly on the text returned by the tool.
        3. Always summarize the returned web page content directly for the user."
    """
    ),
    tools=[aws_browser_search],  # Tool registered cleanly!
)

if __name__ == "__main__":
    print("=== Google ADK Agent successfully initialized with AWS Cloud Browser ===")
# def web_search(query: str):
#     """Search the live web for recent information, news, or current facts.
#
#     Args:
#         query: The search query string to look up.
#     """
#     api_key = os.getenv("SERP_API_KEY")
#     url = 'https://serpapi.com/search?engine=google'
#     params = {
#         'engine': 'google',
#         'q': query,
#         'no_cache': 'true',
#         'api_key': api_key
#     }
#
#     # Use HTTP GET instead of POST
#     response = requests.get(url, params=params, timeout=10)
#     response.raise_for_status()
#
#     data = response.json()
#
#     # Extract organic results gracefully
#     organic_results = data.get("organic_results", [])
#
#     if not organic_results:
#         return "No relevant search results found."
#
#     # Format the top 5 snippets into a scannable string for the Gemini agent
#     formatted_results = []
#     for item in organic_results[:5]:
#         title = item.get("title", "No Title")
#         link = item.get("link", "")
#         snippet = item.get("snippet", "No snippet available.")
#         formatted_results.append(f"Title: {title}\nLink: {link}\nSnippet: {snippet}\n---")
#
#     return "\n".join(formatted_results)
#
#
#
#
# root_agent = Agent(
#     model=LiteLlm(model= 'bedrock/amazon.nova-lite-v1:0'),
#     name='root_agent',
#     description='A helpful assistant for user questions.',
#     instruction='Answer user questions to the best of your knowledge. If unable to answer, use web_search tool to search for answer',
#     tools = [web_search],
# )
#
