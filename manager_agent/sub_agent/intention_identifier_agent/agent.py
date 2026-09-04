import os

from enum import Enum
from pydantic import BaseModel,Field
from typing import Optional

from dotenv import load_dotenv
from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm


load_dotenv(".env")

class Intention(Enum):
    UNCERTAIN_GENERAL = 1
    UNCERTAIN_SPECIFIC = 2
    CERTAIN=3


class IntentionContent(BaseModel):
    intention:Intention = Field(description="Enum value recording intention state of students")
    specific_career:Optional[str] = Field(description="String recording specific AI career path given by students that are certain to pursue in AI in the future")

root_agent = Agent(
    model=LiteLlm(model=os.getenv("BEDROCK_MODEL")),
    name='intention_identifier_agent',
    description='A specialized sub-agent to identify intention of student using AI education consultant',
    instruction=
    """
    You are a sub-agent responsible for identifying intention of student for the manager_agent to determine what action to
    take next. 
    
    GENERAL STEP:
    1. Request input for understanding student's intention: 
       Example - 
       "To get started, feel free to share your intention. Here are some common intention from students using this application :
       1. I come here since I heard AI jobs are one of the most in-demand career right now. However, I am uncertain if I am 
          suitable for this path
       2. I am certain that I will pursue in AI-related career. However, I either don't know how to start to progress or 
          which specific AI career path I should choose
       3. I am certain that I will pursue in a specific AI career path but I don't know how to start learning and progressing"
       
    2. Identify intention and assign their :
       - Identify value for intention field (data type: Enum named Intention) to be provided for output schema 
       - If students was uncertain if they are going to pursue AI-related career, let the value be Intention(1) 
       - If students was certain about pursuing AI-career, but uncertain which specific path to choose, let value be Intention(2)
       - If students was certain about pursuing a specific path of AI-career, but didn't know how to start or progress, 
         let value be Intention(3). Also, if the student specify the specific AI-career path, record it in the output schema too. If not, ask the student 
         for it (** Only when the value is Intention(3)**).
    
    IMPORTANT: Your response MUST be valid JSON matching this structure:
        {
            "intention": value for intention field recorded ,
            "specific_path": "specific AI-career path if students mentioned. if not, left it as None",
        }
        
    DO NOT include any explanations or additional text outside the JSON response.

    """,
    output_schema=IntentionContent,
    output_key="intention_recorder",
)
