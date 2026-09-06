import os

from enum import Enum
from pydantic import BaseModel,Field
from typing import Optional,Any

from dotenv import load_dotenv
from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import ToolContext,FunctionTool


load_dotenv(".env")

class Intention(Enum):
    UNCERTAIN_GENERAL = 1
    UNCERTAIN_SPECIFIC = 2
    CERTAIN=3

class IntentionContent(BaseModel):
    intention:Intention = Field(description="Enum value recording intention state of students")
    specific_career:Optional[str] = Field(description="String recording specific AI career path given by students that are certain to pursue in AI in the future")


def terminate_and_save_tool(tool_context: ToolContext, data_to_record: IntentionContent):
    # 1. Force write the target data into the session state variables
    tool_context.state["intention_recorder"] = data_to_record

    # 2. Terminate the agent run immediately from within the tool
    tool_context.actions.transfer_to_agent ="root_agent"

    return {}

def check_interested(interested_or_not: bool , tool_context: ToolContext) -> dict[Any,Any]:
    """Indicate if the student is interested in AI career"""
    tool_context.state["interested_in_AI"] = interested_or_not
    return {}

intention_identifier_agent = Agent(
    model=LiteLlm(model=os.getenv("BEDROCK_MODEL")),
    name='intention_identifier_agent',
    description='A specialized sub-agent to identify intention of student using AI education consultant',
    instruction=
    """
    You are a sub-agent responsible for identifying intention of student for the root_agent to determine what action to
    take next. 
    
    GENERAL STEP:
    ### STEP 1: Identify intention:
       - Identify value for intention field (data type: Enum named Intention) to be provided for output schema 
       
       - If students was uncertain if they are going to pursue AI-related career:
         * Example: Student imply that he/she was curious due to the fast growing trend of AI, but not sure they suit for this career or not
         * Let the value be Intention(1). 
         * (MANDATORY) PROCEED TO STEP 2
       
       - ELSE IF students was certain about pursuing AI-career, but uncertain which specific path to choose:
         * Example: Student imply that he/she sure want to do something related to AI in the future, but not sure which specific job related to AI he/she want to pick
         * Let value be Intention(2)
         * (MANDATORY) PROCEED TO STEP 2
         
       - ELSE IF students was certain about pursuing a specific path of AI-career , but didn't know how to start or progress:
         * Example: Student state a specific specialization of AI such as machine learning, NLP, prompt engineering etc, but not sure how to start or progress
         * Let value be Intention(3)
         * Record the name for the specific AI-career path. If the student didn't specify, keep asking the student until he/she says the specific AI-career path
         * (MANDATORY) PROCEED TO STEP 2
    
    ### STEP 2:  
       - If value for intention field is 2 or 3, call check_interested with True boolean value as argument
       - If value for intention field is 1, call check_interested with False boolean value 
       
    ### STEP 3: Generate JSON:
       - Generate valid JSON matching this structure as argument for terminate-and_save_tool to be called later on:
        {
            "intention": value for intention field recorded ,
            "specific_path": "specific AI-career path if students mentioned (Example: machine learning, NLP, data scientist). if not, left it as None",
        }
       - PROCEED TO STEP 4
    
    ### STEP 4: (MANDATORY) CALL terminate_and_save_tool
       - (MANDATORY) IMMEDIATELY AFTER GENERATING RESPONSE, SAVE RESPONSE AND RETURN CONTROL TO root_agent by calling terminate_and_save_tool with the JSON response as arguments . 
       - DON'T ENGAGE IN CONVERSATION 
    """,
    tools=[terminate_and_save_tool,check_interested],
)
