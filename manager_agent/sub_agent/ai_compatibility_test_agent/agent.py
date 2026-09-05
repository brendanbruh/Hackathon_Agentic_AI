import os

from enum import Enum

from pandas.io.pytables import format_doc
from pydantic import BaseModel,Field
from typing import Optional,Any,List,Literal,Dict

from dotenv import load_dotenv
from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import ToolContext,FunctionTool

load_dotenv(".env")


DISLIKE = 1
NEUTRAL = 2
WILLING_TO_LEARN = 3
LIKE = 4
EXPERIENCED = 5




class UserTrait(BaseModel):
    """
    Structured extraction of a student's stated preferences from their free-text input.
    Helps avoid raw keyword string matching.
    """
    traits_category: Literal["math_and_logic", "programming", "debugging_patience"]
    status: Literal["like", "dislike", "neutral","familiar","experienced","beginner"]
    willing_to_learn: bool
    evidence: str = Field(
        description="The exact quote or logical reasoning from the user indicating this preference."
    )

class CompatibilityRatings(BaseModel):
    """
    Enforces a strict 1-to-5 Likert scale interface for scoring the structured questionnaire.
    """
    math_and_logic: int = Field(
        ...,
        description="Ratings (1 to 5) for math, analytical, and logical statements."
    )
    debugging_patience: int = Field(
        ...,
        description="Ratings (1 to 5) for debugging, trial-and-error, and troubleshooting resilience."
    )
    programming: int = Field(
        ...,
        description="Ratings (1 to 5) for software development, hands-on building, and coding."
    )


# ----------------------------------------------------------------------
# Tool A: record_user_traits
# ----------------------------------------------------------------------
def record_user_traits(traits: List[UserTrait], tool_context = ToolContext) -> Dict[str, Any]:
    """
    Saves parsed traits (likes, dislikes, and extracted evidence) directly
    into the active session state. This allows the agent to intelligently
    determine which dimensions are still 'unmentioned'.

    Args:
        traits: List of UserTrait, each storing student's preference for either math and logic, debugging patience, or programming with evidence from student's prompt
        tool_context: Context of tool, used to update state of session

    Returns:
        dict : Dictionary that indicate if the operation is success and the recorded student profile
    """
    if "recorded_profile" not in tool_context.state:
        tool_context.state["recorded_profile"] = {}

    # Retrieve existing profile or initialize an empty dictionary
    recorded_profile = tool_context.state.get("recorded_profile", {})

    for trait in traits:
        recorded_profile[trait.traits_category] = {
            "status": trait.status,
            "evidence": trait.evidence
        }

    tool_context.state["recorded_profile"] = recorded_profile
    return {"status": "success", "recorded_profile": recorded_profile}


# ----------------------------------------------------------------------
# Tool B: compatibility_questionnaire
# ----------------------------------------------------------------------
def compatibility_questionnaire(tool_context:ToolContext) -> Dict[str, Any]:
    """
    Returns specific diagnostic statements for missing categories.
    Students rate these from 1 (Strongly Disagree) to 5 (Strongly Agree).
    Grounded directly in SimplifyNext Hackathon.pdf and Determine_suitable_for_AI.txt.
    """
    all_questions = {
        "math_and_logic": [
            "I enjoy solving problems where the solution is not immediately obvious.",  # [3]
            "I am comfortable with mathematics, algorithms, and analytical thinking."  # [8]
        ],
        "programming": [
            "I enjoy writing and debugging code to solve problems.",  # [2]
            "I enjoy combining different technologies to build a complete system."  # [3]
        ],
        "debugging_patience": [
            "I enjoy troubleshooting systems when something does not work as expected.",  # [3]
            "I enjoy experimenting with different approaches to find the best solution."  # [2]
        ],
        "continuous_learning": [
            "I am comfortable learning new programming languages, frameworks, and technologies continuously.",  # [2]
            "I enjoy keeping up with tech news and tinkering with new tools on weekends."  # [1]
        ],
        "salary_preference": [
            "Is a starting junior salary of SGD 6,000 - SGD 8,500/month aligned with your expectations?"  # [4]
        ],
    }

    if "recorded_profile" not in tool_context.state:
        tool_context.state["recorded_profile"] = {}

    # Filter questions to only return the categories the student hasn't addressed yet
    selected_questions = {
        cat: all_questions[cat] for cat in all_questions if (cat not in tool_context.state["recorded_profile"] and cat != "salary_preference" and cat != "continuous_learning")
    }

    for key in tool_context.state["recorded_profile"]:
        if  tool_context.state["recorded_profile"][key]["status"] in ["familiar","beginner"]:
            selected_questions["continuous_learning_for_"+key] = all_questions["continuous_learning"]

    if "salary_preference" not in tool_context.state:
        selected_questions["salary_preference"] = all_questions["salary_preference"]

    return {
        "instruction": "Please rate these statements from 1 (Strongly Disagree) to 5 (Strongly Agree):",
        "questions": selected_questions
    }


def want_high_salary(tool_context:ToolContext) -> Dict[Any,Any]:
    tool_context.state["salary_preference"] = True
    return {}

def record_score_from_student(question_score_dict:Dict,tool_context:ToolContext) -> Dict:
    """
    Compiling every score recorded and determine if the student is suitable to pursue in AI career
    Args:
      question_score_dict: Dictionary with each question as its key and their corresponding score as its value
      tool_context: Context of tool

    Return:
        Dict: Dictionary indicating success of operation and result.
    """

    score = 0
    for each_section in question_score_dict:
        if question_score_dict[each_section]:
            score += sum(question_score_dict[each_section]) / len(question_score_dict[each_section])

    # known_from_prompt = tool_context.state.get("recorded_profile", {})
    # for each_trait in known_from_prompt:
    #     trait = known_from_prompt[each_trait]
    #     if trait["status"]:
    #         if trait["status"].lower() == "dislike":
    #             score += DISLIKE
    #         elif trait["status"].lower() == "neutral":
    #             score += NEUTRAL
    #         elif trait["status"].lower() == "like":
    #             score += LIKE
    #         elif trait["status"].lower() == "experienced":
    #             score += EXPERIENCED


    if tool_context.state.get("salary_preference", False):
        score += 4

    MAX_SCORE = 20
    suitability_pct = (score / MAX_SCORE) * 100

    # Determine the profile category using Coursera's Green/Orange/Low matching bands [5, 6]
    if suitability_pct >= 80:
        band = "Strong AI Signal (Green Match)"  # [5]
        advice = (
            "You show a stellar natural alignment for an AI career! Your interests in programming, "
            "problem-solving, and continuous experimentation [9] mean you will likely thrive in technical "
            "AI Engineering roles [1]. A starting junior salary of SGD 6,000 - 8,500 is very much aligned [4]!"
        )
    elif suitability_pct >= 60:
        band = "Foundational Match (Orange Match)"  # [6]
        advice = (
            "You have a solid foundation! However, you may need additional technical or mathematical skill development [6]. "
            "If you dislike deep mathematical theory but love building apps, you might want to pivot towards applied "
            "generative AI development (using API integrations and frameworks) or AI Product Management [10, 11]."
        )
    else:
        band = "Low Current Alignment"
        advice = (
            "A highly technical AI engineering path might feel frustrating right now. Consider standard software "
            "engineering, UX design, or other tech domains first [12, 13], or explore low-code AI integrations to test the waters."
        )

    return {
        "status": "success",
        "suitability_percentage": round(suitability_pct, 1),
        "evaluation_band": band,
        # "pillar_breakdowns": {
        #     "Math & Logic": sum(ratings.math_and_logic) / len(
        #         ratings.math_and_logic) if ratings.math_and_logic else 1.0,
        #     "Debugging Resilience": sum(ratings.debugging_patience) / len(
        #         ratings.debugging_patience) if ratings.debugging_patience else 1.0,
        #     "Continuous Learning": sum(ratings.continuous_learning) / len(
        #         ratings.continuous_learning) if ratings.continuous_learning else 1.0,
        #     "Programming Intensity": sum(ratings.programming) / len(
        #         ratings.programming) if ratings.programming else 1.0,
        # },
        "career_advice": advice
    }


root_agent = Agent(
    model=LiteLlm(model=os.getenv("BEDROCK_MODEL")),
    name='root_agent',
    description='A helpful sub-agent responsible to test if a student is interested to pursue in AI career',
    instruction=
    """
You are the "AI Compatibility Test Agent", an empathetic, supportive, yet analytical career counselor. 

Your objective is to evaluate if a student is well-suited to pursue a career in Artificial Intelligence based on their skills, personality, and work style.

## EVALUATION PILLARS
You must evaluate the student across these foundational criteria:
1. Math & Logic: Does the student enjoy linear algebra, calculus, or statistics? 
2. Programming & Application: Do they prefer building working apps over pure theory? 
3. Debugging Patience: Do they have the patience for debugging, data cleaning, and trial-and-error? 
4. Expectation: Are starting salaries of SGD 6,000–8,500/month aligned with their expectations?

## CRITICAL RULE 
> For score evaluation and recording system, PLEASE USE THE TOOLS GIVEN (record_score_from_student) instead of calculating by your own

## CONVERSATIONAL WORKFLOW
1. Welcome the student warmly. Suggest they share their interests, skills, personality, or preferred starting salary.

2. Provide them with two clear choices to begin:
   - OPTION A: Type a natural language prompt explaining their background.
   - OPTION B: Take a structured questionnaire rating statements on a scale of 1 to 5.
   
3. If they provide a Prompt (ONLY LIMITED FOR Option A, NOT OPTION B):
   - Immediately parse their response, generate a list with UserTrait objects for evaluation pillar 1,2,3 ; 
     and run the `record_user_traits` tool to save analyzed traits.
   - (ONLY IF USER PICK OPTION A) Call want_high_salary ONLY WHEN STUDENT MENTIONED A HIGH AMOUNT OF SALARY 
   - DONT CALL want_high_salary when student didn't mentioned about high salary or he/she mentioned money is the biggest concern.
   - Call compatibility_questionnaire tool to ask questions about evaluation pillars that had been missed out. PROCEED TO STEP 5
   
4. If they prefer the Questionnaire (Option B):
   - Directly invoke the `compatibility_questionnaire` tool for all categories without any confimation.
   
5. After invoking 'compatibility_questionnaire' (BOTH OPTION A AND OPTION B MUST FOLLOW):
   CRITICAL RULE:
   1. ANY QUESTIONS COMING OUT FROM compatibility_questionnaire should NOT be added into recorded_profile
   2. IF THE CURRENT EVALUATED PILLAR is found inside recorded_profile in session state, don't need to ask extra question about this pillar 
   3. (MANDATORY) ENSURE EVERYTHING IN THE 'questions' dictionary is accessed and asked 


   ** HOW TO ASK QUESTIONS AND RECORD SCORE: ** 

   1. Inspect the returned 'questions' and ask the student the questions inside the 'questions' dictionary pillar by pillar (key by key) and record the score by key.
   2. (MANDATORY) EVERYTIME THE STUDENT GIVE SCORES, RECORD the score (1-5) given by the students and replace the values for each key with the list of score
   3. (MANDATORY) PROMPT QUESTIONS from the 'questions' dictionary key by key at one time. DON'T OUTPUT ALL QUESTIONS IN DICTIONARY AT ONCE
   4. (MANDATORY) BEFORE EVERY JUMPING TO THE NEXT KEY, if you notice for a particular evaluation pillar in question_score_dict has low score (1-2) 
      IN THE LIST (value by the respective pillar key), we wish to ensure if the student find himself lacking in this particular pillar
        > feel free to ask more questions about each evaluation pillar by yourself (with rating 1-5)
        > update the list of marks for each pillar inside the 'question_score_dict' dictionary
        > IMPORTANT: IF WANT TO ASK EXTRA QUESTION, THE MAXIMUM IS 2 ONLY FOR EACH PILLAR or KEY 
        > IMPORTANT: WHEN ASKING EXTRA QUESTION, PLEASE ONLY ASK POSITIVE QUESTION FOCUSING ON POSSIBILITIES, STRENGTHS AND DESIRED OUTCOMES
        > Example (Prefer to come up by your own POSITIVE QUESTIONS):
           * DON'T ASK : "Do you find yourself getting frustrated easily when debugging or troubleshooting? (Rate 1-5)" 
             (Reason to reject: Asking negative emotion of students encountering one of the evaluating pillar)
   5. IF notice the current question going to be prompted is almost similar to question asked before, dont ask it again, however update the current list with the score from the similar question
   6.(IMPORTANT) Ensure that every key in question_score_dict is not an empty list (at least every questions from 'questions' has been asked and record a mark before)
   
   Example: 
     * If returned dictionary 'questions' = {"math_and_logic" : [list of two questions],'debugging': [list of three questions]}
     * Prompt the student ONLY QUESTIONS from the list accessed by key "math_and_logic":
       > wait student to reply scores for QUESTIONS from the list accessed by key "debugging" and record the score. If student give irrelevant input, ask him 
         to reinput again before proceeding
       > Student offer score 3 for the first questions and 4 for the second question
       > DONT REPLACE THE DICTIONARY. JUST MODIFY THE DICTIONARY (but let us call it question_score_dict onwards)
       > Replace the value for key "math_and_logic" with a list [3,4] and REMAIN the key to be the same
     * After student reply the score for "math_and_logic", then prompt the student ONLY QUESTIONS from the list accessed by key "debugging"
       > wait student to reply scores for QUESTIONS from the list accessed by key "debugging" and record the score. If student give irrelevant input, ask him 
         to reinput again before proceeding
       > Student offer score 1 for the first questions,  5 for the second question and 2 for the third question
       > DONT REPLACE question_score_dict. JUST MODIFY question_score_dict 
       > Replace the value for key "debugging" with a list [1,5,2] and REMAIN the key to be the same
       > Notice that the list has one low score '1', thus ASK AN EXTRA QUESTION (MAX 2) and ask them to rate by 1-5 again.
       > If student give a score of 5, then the question_score_dict will update key "debugging" with a new value which 
          is a updated list [1,5,2,5]
     * Final output : question_score_dict = {"math_and_logic" : [3,4],'debugging': [1,5,2,5]}

   
6. Once profile is completed:
   - Invoke `record_score_from_student` to compile and process the scores.
   - REPORT their matching percentage, their matching band color, and practical advice grounded in Singapore's career standards ONLY USING RESPONSE FROM
     'record_score_from_student' PLEASE DO NOT USE YOUR OWN FORMULA.
   
    """,
    tools=[record_user_traits,compatibility_questionnaire,want_high_salary,record_score_from_student],
)
#
# - If notice the question about to be asked is almost the same to questions asked before,
#   Example:
#   * Previous key inspected : "continuous_learning_for_math_and_logic" and its question include "I am comfortable learning
#     new programming languages, frameworks, and technologies continuously"
#   * Current key inspected: "continuous_learning_for_debugging_patience" and one of its question include the
#     almost-identical question, then don't ask the question again.
#   * For any of this sort of scenario, only ask the question once
#

# 3.(MANDATORY)
# AFTER
# EVERY
# PILLAR
# IS
# ACCESSED
# ONCE
# IN
# question_score_dict
# BEFORE
# INVOKING
# 'record_score_from_student',
# - Example:
# *Assuming
# question_score_dict
# has
# value[1, 5]
# for key "programming", so noticing score 1 is consider a low score,
# so
# we
# need
# to
# investigate
# more in this
# particular
# "programming"
# pillar
# *If
# you
# ask
# an
# extra
# question
# for "programming" pillar such as "Do you try to build any personal project before ?"
# and ask
# them
# to
# rate
# by
# 1 - 5
# again.
# *If
# student
# give
# a
# score
# of
# 5, then
# the
# question_score_dict
# will
# update
# key
# "programming"
# with a new value which
# is a
# updated
# list[3, 4, 5]
# - IMPORTANT: IF
# WANT
# TO
# ASK
# EXTRA
# QUESTION, THE
# MAXIMUM
# IS
# 5
# ONLY.(suggested
# 2 - 3)
# - IMPORTANT: WHEN
# ASKING
# EXTRA
# QUESTION, PLEASE
# ONLY
# ASK
# POSITIVE
# QUESTION
# FOCUSING
# ON
# POSSIBILITIES, STRENGTHS
# AND
# DESIRED
# OUTCOMES
# Example(Prefer
# to
# come
# up
# by
# your
# own):
# *DON
# 'T ASK : "Do you find yourself getting frustrated easily when debugging or troubleshooting? (Rate 1-5)"
# (Reason to reject: Asking negative emotion of students encountering one of the evaluating pillar)
# *DO
# ASK: "Do you find fixing a complex bug sastifying"
# for debugging_patience pillar(FOCUS ON POSSIBILITIES)
# "Do you like to solve complex or mind-boggling puzzzles"
# for programming pillar(FOCUS ON STRENGTHS)
# "Do you wish to become a lead expert in tech field in the future ?"
# for expectation(FOCUS ON DESIRED OUTCOMES)
# -
#
# 4.(MANDATORY)
# ENSURE
# THAT
# ANY
# KEY
# FROM
# recorded_profile
# SHOULD
# NOT
# BE
# ADDED
# INTO
# question_score_dict

