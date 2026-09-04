import os

from dotenv import load_dotenv
from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm


load_dotenv(".env")

manager_agent = Agent(
    model=LiteLlm(model=os.getenv("BEDROCK_MODEL")),
    name='manager_agent',
    description=
    """
    Coordinator agent for AI Education Consultant that orchestrates specialized sub-agents to 
    provide AI learning or career path for students
    """,
    instruction=
    """
    You are a coordinator agent that is responsible for overseeing the work of the other sub-agents
    to provide AI learning or career .

    Always delegate the task to the appropriate agent. Use your best judgement 
    to determine which agent to delegate to.
    """,
)
