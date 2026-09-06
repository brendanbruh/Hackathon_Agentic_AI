import os

from dotenv import load_dotenv
from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import ToolContext, FunctionTool, google_search


load_dotenv(".env")

root_agent = Agent(
    model=LiteLlm(model= 'bedrock/amazon.nova-lite-v1:0'),
    name='root_agent',
    description='A helpful assistant for user questions.',
    instruction='Answer user questions to the best of your knowledge. If unable to answer, use google_search tool to search for answer',
    tools = [google_search],
)