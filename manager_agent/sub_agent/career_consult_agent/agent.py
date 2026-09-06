import os
import random
from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, Any, List, Literal, Dict
from dotenv import load_dotenv

from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import ToolContext, FunctionTool

# Import common tool for control transfer
from sub_agent.common_tool import transfer_control_to_root  # Assuming common_tools.py is in the same directory

# Load environment variables
load_dotenv(".env")

# Define Constants matching your compatibility agent style
DISLIKE = 1
NEUTRAL = 2
WILLING_TO_LEARN = 3
LIKE = 4
EXPERIENCED = 5


# ----------------------------------------------------------------------
# 1. DEFINE SCHEMAS (Pydantic Models)
# ----------------------------------------------------------------------

class UserTrait(BaseModel):
    """
    Structured extraction of a student's stated preferences from their free-text input.
    Maps general baseline computing backgrounds, interests, and aspirations to help identify AI tracks.
    """
    traits_category: Literal[
        "ai_engineer",
        "data_scientist",
        "mlops_engineer",
        "nlp_llm_engineer",
        "computer_vision_engineer",
        "ml_research_scientist"
    ]
    status: Literal["like", "dislike", "neutral", "familiar", "experienced", "beginner"]
    willing_to_learn: bool
    evidence: str = Field(
        description="The exact quote or logical reasoning from the user indicating this preference."
    )


class CareerTrackScores(BaseModel):
    """
    Structured scores given by the student for each specific AI career track.
    Each field contains a list of ratings (integers from 1 to 5) given by the student.
    """
    ai_engineer: List[int] = Field(default_factory=list,
                                   description="Scores (1 to 5) for AI Engineer-related statements.")
    data_scientist: List[int] = Field(default_factory=list,
                                      description="Scores (1 to 5) for Data Scientist-related statements.")
    mlops_engineer: List[int] = Field(default_factory=list,
                                      description="Scores (1 to 5) for MLOps Engineer-related statements.")
    nlp_llm_engineer: List[int] = Field(default_factory=list,
                                        description="Scores (1 to 5) for NLP / LLM Specialist-related statements.")
    computer_vision_engineer: List[int] = Field(default_factory=list,
                                                description="Scores (1 to 5) for Computer Vision Engineer-related statements.")
    ml_research_scientist: List[int] = Field(default_factory=list,
                                             description="Scores (1 to 5) for ML Research Scientist-related statements.")
    # Optional continuous learning scores for beginners/familiars
    continuous_learning_for_ai_engineer: List[int] = Field(default_factory=list,
                                                           description="Scores for continuous learning related to AI Engineering.")
    continuous_learning_for_data_scientist: List[int] = Field(default_factory=list,
                                                              description="Scores for continuous learning related to Data Science.")
    continuous_learning_for_mlops_engineer: List[int] = Field(default_factory=list,
                                                              description="Scores for continuous learning related to MLOps.")
    continuous_learning_for_nlp_llm_engineer: List[int] = Field(default_factory=list,
                                                                description="Scores for continuous learning related to NLP.")
    continuous_learning_for_computer_vision_engineer: List[int] = Field(default_factory=list,
                                                                        description="Scores for continuous learning related to Computer Vision.")
    continuous_learning_for_ml_research_scientist: List[int] = Field(default_factory=list,
                                                                     description="Scores for continuous learning related to ML Research.")


# ----------------------------------------------------------------------
# 2. DEFINE NATIVE TOOL FUNCTIONS
# ----------------------------------------------------------------------

def record_user_traits(traits: List[UserTrait], tool_context=ToolContext) -> Dict[str, Any]:
    """
    Args:
        traits: List of UserTrait to indicate if the student beforehand mentioned his preference for a particular AI career
    Return:
        Dictionary indicating the success of operation and the traits

    Saves parsed traits (likes, dislikes, and extracted evidence) directly into the active session state.
    This allows the agent to intelligently determine which dimensions are still 'unmentioned'.
    """
    if "recorded_profile" not in tool_context.state:
        tool_context.state["recorded_profile"] = {}
    recorded_profile = tool_context.state.get("recorded_profile", {})
    for trait in traits:
        recorded_profile[trait.traits_category] = {
            "status": trait.status,
            "evidence": trait.evidence
        }
    tool_context.state["recorded_profile"] = recorded_profile
    return {"status": "success", "recorded_profile": recorded_profile}


def career_path_questionnaire(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Return:
        Dictionary with specific diagnostic statements/questions for career categories that need further clarification.

    These are calibrated around basic CS/programming, data preferences, personalities, and aspirations.
    The returned list of questions is completely flattened and jumbled (randomly shuffled) under the hood.
    This ensures students rate statements honestly without knowing which career path they map to.
    """
    all_questions = {
        "ai_engineer": [
            "I enjoy basic software development—writing clean functions, building working prototypes, and integrating databases or APIs.",
            "I have a 'builder's mindset' and prefer writing code to make a working app rather than researching theoretical algorithms.",
            "I aspire to build smart, user-facing applications (like automated assistants, tools, or chatbots) that help people solve everyday problems."
        ],
        "data_scientist": [
            "I enjoy working with SQL databases, writing queries, and summarizing large datasets into visual charts or spreadsheets.",
            "I am an analytical thinker who enjoys finding hidden patterns in numbers and using statistics to solve real-world problems.",
            "I aspire to influence strategic decisions by helping leaders understand the story behind the data rather than just writing backend code."
        ],
        "mlops_engineer": [
            "I enjoy system administration tasks—working in the Linux command line (bash), configuring Docker containers, or automating manual files/scripts.",
            "I am highly detail-oriented and prefer organizing clean folder structures, tracking system performance, and ensuring software runs reliably.",
            "I aspire to design robust backend pipelines and automate software deployments so that systems can scale seamlessly to handle thousands of users."
        ],
        "nlp_llm_engineer": [
            "I enjoy text-processing tasks, such as reading text files, parsing strings, searching for keywords, or studying linguistics and grammar.",
            "I prefer working with communication, languages, translation, and structured text documents over heavy image datasets or numeric grids.",
            "I aspire to build conversational AI agents that can read, understand context, write summaries, and chat naturally with humans."
        ],
        "computer_vision_engineer": [
            "I enjoy working with visual assets—manipulating pixel arrays, working with image/video files, or writing graphics programs.",
            "I am a visual thinker who excels at spatial coordination, geometry, and tracking 2D/3D objects.",
            "I aspire to develop smart vision systems that can teach computers to see, recognize objects, or pilot physical hardware like drones and robots."
        ],
        "ml_research_scientist": [
            "I am highly comfortable with college-level advanced mathematics (linear algebra, calculus, and statistics).",
            "I enjoy abstract thinking, solving complex logic puzzles, and reading academic papers to understand how things work at their deepest level.",
            "I aspire to contribute to the scientific community by inventing novel algorithms and mathematical models rather than using existing tools."
        ],
        "continuous_learning": [
            "I am comfortable learning new programming languages, frameworks, and technologies continuously.",
            "I enjoy keeping up with tech news and tinkering with new tools on weekends."
        ],
        "salary_preference": [
            "Are Singapore monthly starting salary benchmarks (SGD 6,000 - SGD 10,000+) aligned with your career goals?"
        ]
    }
    if "recorded_profile" not in tool_context.state:
        tool_context.state["recorded_profile"] = {}

    # Filter categories that the student hasn't addressed yet
    selected_categories = {}
    for cat in all_questions:
        if (cat not in tool_context.state["recorded_profile"] and
                cat != "salary_preference" and
                cat != "continuous_learning"):
            selected_categories[cat] = all_questions[cat]

    # If the user is familiar or beginner with a track, prompt them about continuous learning
    for key in tool_context.state["recorded_profile"]:
        if tool_context.state["recorded_profile"][key]["status"] in ["familiar", "beginner"]:
            selected_categories["continuous_learning_for_" + key] = all_questions["continuous_learning"]

    if "salary_preference" not in tool_context.state:
        selected_categories["salary_preference"] = all_questions["salary_preference"]

    # Flatten and jumble all questions into a single flat list
    flat_questions = []
    for cat, q_list in selected_categories.items():
        for q in q_list:
            flat_questions.append({
                "text": q,
                "category": cat
            })

    # Jumble/randomize the order of the questions
    random.shuffle(flat_questions)

    return {
        "instruction": "Please rate these statements from 1 (Strongly Disagree) to 5 (Strongly Agree). Do NOT show the category names to the user.",
        "questions": flat_questions
    }


def want_high_salary(tool_context: ToolContext) -> Dict[Any, Any]:
    """Marks that the student explicitly desires a high salary path."""
    tool_context.state["salary_preference"] = True
    return {}


def record_score_from_student(scores: CareerTrackScores, tool_context: ToolContext) -> Dict:
    """
    Args:
        scores: CareerTrackScores storing structured scores given by the student for each specific AI career track.
    Return:
        Dictionary for storing the final deduced optimal career path for the student

    Compiling every score recorded and determine which specific AI career path the student should take.
    Uses the exact same scoring style as your text file but calculates career match vectors.
    """
    # Initialize track scores
    tracks = [
        "ai_engineer",
        "data_scientist",
        "mlops_engineer",
        "nlp_llm_engineer",
        "computer_vision_engineer",
        "ml_research_scientist"
    ]

    # Base mapping for prompt traits
    status_score_map = {
        "dislike": DISLIKE,
        "neutral": NEUTRAL,
        "beginner": 2,
        "familiar": WILLING_TO_LEARN,
        "like": LIKE,
        "experienced": EXPERIENCED
    }

    # Compute vector scores for each track
    track_scores = {}
    known_from_prompt = tool_context.state.get("recorded_profile", {})
    for track in tracks:
        # Check if we have explicit ratings from the questionnaire (from CareerTrackScores fields)
        ratings = getattr(scores, track, [])
        if ratings and len(ratings) > 0:
            avg_rating = sum(ratings) / len(ratings)
            track_scores[track] = avg_rating
        # Else check if we have matching prompt traits recorded
        elif track in known_from_prompt:
            trait = known_from_prompt[track]
            status = trait.get("status", "neutral").lower()
            track_scores[track] = status_score_map.get(status, NEUTRAL)
        # Default fallback
        else:
            track_scores[track] = float(NEUTRAL)

    # Add bonus if continuous learning is positive and user expressed willing_to_learn or high rating
    for track in tracks:  # Iterate over tracks to apply CL scores to each
        cl_field_name = f"continuous_learning_for_{track}"
        cl_ratings = getattr(scores, cl_field_name, [])
        if cl_ratings and len(cl_ratings) > 0:
            cl_avg = sum(cl_ratings) / len(cl_ratings)
            # Ensure track_scores[track] exists before modifying
            if track in track_scores:
                track_scores[track] += (cl_avg - 3.0) * 0.25  # Adjust score up/down slightly based on continuous learning interest

    # Add salary preference weighting
    if tool_context.state.get("salary_preference", False):
        # High salary expectation slightly favors Research and MLOps / AI Engineer paths
        track_scores["ml_research_scientist"] += 0.2
        track_scores["mlops_engineer"] += 0.1
        track_scores["ai_engineer"] += 0.1

    # Convert raw 1-5 scores to match percentages (max score is 5.0)
    match_percentages = {}
    for track in tracks:
        raw_score = track_scores[track]
        # Cap raw score to range [1.0, 5.0]
        raw_score = max(1.0, min(5.0, raw_score))
        pct = (raw_score / 5.0) * 100
        match_percentages[track] = round(pct, 1)

    # Sort tracks to find primary and secondary career recommendations
    title_mapping = {
        "ai_engineer": "AI Engineer",
        "data_scientist": "Data Scientist",
        "mlops_engineer": "MLOps Engineer",
        "nlp_llm_engineer": "NLP / LLM Specialist",
        "computer_vision_engineer": "Computer Vision Engineer",
        "ml_research_scientist": "ML Research Scientist"
    }
    sorted_results = sorted(match_percentages.items(), key=lambda x: x[1], reverse=True)
    primary_key, primary_pct = sorted_results
    secondary_key, secondary_pct = sorted_results[1]
    primary_title = title_mapping[primary_key]
    secondary_title = title_mapping[secondary_key]

    # Grounded advice mapping Singapore monthly starting salaries and milestone characteristics
    career_details = {
        "ai_engineer": {
            "salary": "Junior: SGD 6,000 - SGD 8,500/month | Senior: SGD 9,000 - SGD 15,000/month",
            "milestone": "The MLOps ceiling: Plateauing when you hit deployment scaling issues or can't own production infrastructure.",
            "learning_path": "Focus on API integrations, RAG architectures, LangChain, and production app scaling."
        },
        "data_scientist": {
            "salary": "Junior: SGD 5,500 - SGD 7,800/month | Senior: SGD 8,500 - SGD 14,000/month",
            "milestone": "The Business Translation ceiling: Delivering statistically perfect notebooks that business leadership fails to act on.",
            "learning_path": "Focus on SQL mastery, business intelligence storytelling, experimental design, and communication."
        },
        "mlops_engineer": {
            "salary": "Junior: SGD 6,500 - SGD 9,000/month | Senior: SGD 9,500 - SGD 16,000/month",
            "milestone": "The Complexity gap: Mastering container pipelines but failing to interrogate core model weights or debugging latency degradation.",
            "learning_path": "Focus on Docker, Kubernetes, CI/CD pipelines, AWS/GCP, and scalable model-serving frameworks like Triton."
        },
        "nlp_llm_engineer": {
            "salary": "Junior: SGD 6,000 - SGD 8,800/month | Senior: SGD 9,000 - SGD 15,500/month",
            "milestone": "The Text-only limitation: Facing barriers when scaling text models to heavy spatial, video, or multi-modal edge systems.",
            "learning_path": "Focus on Transformers, HuggingFace, fine-tuning large models, prompt engineering patterns, and autonomous agents."
        },
        "computer_vision_engineer": {
            "salary": "Junior: SGD 6,000 - SGD 8,500/month | Senior: SGD 9,000 - SGD 15,000/month",
            "milestone": "The Edge ceiling: Creating mathematically sound visual models that cannot serve in real-time under low-power physical device constraints.",
            "learning_path": "Focus on OpenCV, convolutional neural networks, object detection (YOLO), PyTorch CV, and edge optimization."
        },
        "ml_research_scientist": {
            "salary": "Junior: SGD 7,000 - SGD 10,000/month | Senior: SGD 10,000 - SGD 18,000/month",
            "milestone": "The Application gap: Inventing theoretical models that are computationally beautiful but too expensive to ever ship to production.",
            "learning_path": "Focus on advanced linear algebra, deep mathematical optimization, writing academic papers, and JAX/PyTorch theory."
        }
    }
    primary_info = career_details[primary_key]
    advice = (
        f"Your top match is {primary_title} ({primary_pct}% match)! "
        f"In Singapore, junior starting salaries in this path average {primary_info['salary']}. "
        f"Be aware of your typical mid-career roadblock: '{primary_info['milestone']}' "
        f"Your secondary career fit is {secondary_title} at {secondary_pct}% match. "
        f"We recommend following the: '{primary_info['learning_path']}'"
    )

    results = {
        "status": "success",
        "primary_recommendation": primary_title,
        "primary_match_pct": primary_pct,
        "secondary_recommendation": secondary_title,
        "secondary_match_pct": secondary_pct,
        "all_matches": {title_mapping[k]: f"{v}%" for k, v in match_percentages.items()},
        "salary_benchmark": primary_info["salary"],
        "milestone_risk": primary_info["milestone"],
        "career_advice": advice
    }
    tool_context.state["career_fit_results"] = results

    return results


# ----------------------------------------------------------------------
# 3. CONFIGURE THE LLM AGENT (SYSTEM PROMPT)
# ----------------------------------------------------------------------

SYSTEM_INSTRUCTION = """You are the "AI Career Consultant Agent", a strategic, analytical, and highly structured career advisor.
Your goal is to guide students who are certain they want an AI career to their ideal specific specialization, helping them understand their unique strengths and the real-world milestone plateaus they will face.

## KEY PHILOSOPHY: FOUNDATIONAL CS & PERSONALITY OVER ADVANCED TOOLS
Because many users are students who may not have deep AI skillsets yet, you must evaluate them based on:
1. Basic Computer Science Skills: Basic software building, command line comfort, working with databases (SQL), parsing strings, or math classes.
2. Preferred Data Types: Whether they naturally enjoy working with structured tables, written text, pixel graphics/video, or abstract math coordinates.
3. Personality & Work Style: Prefers applied hands-on building vs. abstract research, methodical reliability vs. creative exploration.
4. Career Aspirations & Salary Expectations: High salary demands vs. interest in creating social impact or highly scaled software pipelines.

## THE 6 SPECIALIZED PATHWAYS & TYPICAL ROADBLOCKS
1. AI Engineer: Focuses on applied application building (writing code, API integrations, prototype assembling).
Roadblock: The MLOps ceiling (plateauing when they hit deployment scaling issues or can't own production infrastructure).
2. Data Scientist: Focuses on data exploration, metrics, SQL databases, and business insights.
Roadblock: The Business Translation ceiling (delivering statistically perfect notebooks that business leadership fails to act on).
3. MLOps Engineer: Focuses on deployment, Linux command line (bash), Docker containers, scaling, and automation pipelines.
Roadblock: The Complexity gap (mastering container pipelines but failing to interrogate core model weights or debugging latency degradation).
4. NLP / LLM Engineer: Focuses on linguistics, text files, parsing strings, text summaries, and conversational flows.
Roadblock: The Text-only limitation (facing barriers when scaling text models to heavy spatial, video, or multi-modal edge systems).
5. Computer Vision Engineer: Focuses on image arrays, video files, pixel coordinates, geometry, and hardware/cameras.
Roadblock: The Edge ceiling (creating mathematically sound visual models that cannot serve in real-time under low-power physical device constraints).
6. ML Research Scientist: Focuses on linear algebra, calculus, advanced logic puzzles, and reading research papers.
Roadblock: The Application gap (inventing theoretical models that are computationally beautiful but too expensive to ever ship to production).

## FOUNDATIONAL INTEREST KEYWORDS TO MATCH FOR
- AI Engineer: Python, building apps, software development, full-stack, backend, APIs, databases, prototyping, hands-on, troubleshooting.
- Data Scientist: SQL, data analysis, spreadsheets, charts, statistics, probability, data visualization, communication, presentation, reporting.
- MLOps Engineer: Linux bash, command-line, Docker, file management, automation, servers, cloud, networking, reliability, backups.
- NLP / LLM Engineer: text strings, file reading, writing essays, linguistics, foreign languages, parsing text, keywords, dictionaries.
- Computer Vision Engineer: image files, video streams, cameras, pixels, spatial coordinate systems, 2D/3D math, graphics.
- ML Research Scientist: linear algebra, calculus, mathematical proofs, research papers, original algorithms, theory, math equations.

## CRITICAL RULES FOR SYSTEM TOOLS
1. DONT CALL record_score_from_student PREMATURELY: This tool MUST ONLY be called ONCE at the very end of the entire questionnaire process when ALL questions in the list have been asked and rated.
2. DO NOT call any tool when the student gives a rating (1-5) for an individual statement. Simply record the score internally in your mind/context. Keep updating your running score maps in your thoughts.
3. STRUCTURED TOOL CALL contract: The tool `record_score_from_student` accepts a structured `scores` parameter conforming to `CareerTrackScores`. When calling this tool, populate each list field (`ai_engineer`, `data_scientist`, etc.) with the exact list of ratings the student provided during the questionnaire. For any fields where continuous learning statements were asked, put those ratings under the matching `continuous_learning_for_` field.
4. GREETING RULE (ANTI-PREMATURE CALLS): Do NOT call `record_user_traits`, `want_high_salary`, or `career_path_questionnaire` on a simple greeting (like 'hi', 'hello', 'hey', 'start'). You must ONLY present the starting choices (Option A and Option B) and wait for the user to make a choice. Do NOT parse a greeting as a prompt.
5. COMPULSORY QUESTIONNAIRE TOOL EXECUTION (DO NOT FAKE QUESTIONS):
- You are STRICTLY PROHIBITED from printing, generating, listing, or guessing questionnaire statements on your own from your memory.
- To get the questions, you MUST CALL the `career_path_questionnaire` tool first in the active turn.
- You CANNOT ask the student even a single statement from the questionnaire (either Option A follow-ups or Option B) without first executing the `career_path_questionnaire` tool. Always execute the tool first to fetch the randomized list of statements!
6. **CONTROL TRANSFER RULE**: Upon completing the career consultation (i.e., after `record_score_from_student` has been called and the final results are reported), you MUST immediately call the `transfer_control_to_tool` tool to explicitly return control to the orchestrator.
7. **EXPLICIT RESULT STORAGE**: Your final career fit evaluation results (including primary/secondary recommendations and advice) are stored in `tool_context.state['career_fit_results']`.
8. (MANDATORY) ENSURE EVERY OUTPUT KEY OR DATA STORED IN STATE SHOULD BE A VALID JSON SO IT IS JSON SERIALIZABLE 

## MASKING & JUMBLED WORKFLOW RULES (ANTI-BIAS)
1. STRICT MASKING OF PATH LABELS: To ensure the student is completely unbiased and answers genuinely, NEVER mention, output, or imply what career path (e.g. AI Engineer, MLOps, Data Scientist) any question maps to.
2. DYNAMIC QUESTIONS: The `career_path_questionnaire` tool returns questions completely flattened and in a jumbled (randomly shuffled) order. Each question contains the text ("text") and its internal category target ("category").
3. DO NOT BATCH ALL QUESTIONS AT ONCE: Present the questions to the student ONE BY ONE in the exact jumbled order returned by the tool. Wait for the user to reply with a rating (1-5) before presenting the next question.
4. MASKING EXAMPLE:
- BAD: "For Data Scientist: 'Do you enjoy SQL?'"
- GOOD: "Rate this statement from 1 (Strongly Disagree) to 5 (Strongly Agree): 'I enjoy working with SQL databases, writing queries, and summarizing large datasets.'"

## CONVERSATIONAL WORKFLOW
1. Welcome the student warmly on greeting. Present them with two clear choices to begin:
* OPTION A: Type a natural language prompt explaining their background.
* OPTION B: Take a structured questionnaire rating statements on a scale of 1 to 5.
* REMEMBER: Do NOT execute any tool calls on initial greeting!

2. If they select OPTION A (Natural Language Prompt):
* Wait for them to provide their background/interests description.
* Once they provide their background description:
- **Turn 1 (Compulsory Step):** Immediately call `record_user_traits` to save their traits (even if they are a beginner, map their Python or math backgrounds to appropriate category vectors with "beginner" or "familiar" status).
- If they specified a high salary expectation, you can call `want_high_salary` in parallel in this same Turn 1.
- **Turn 2 (Transition Step):** Once you receive the `record_user_traits` success response in your tool-context, you **MUST immediately call the `career_path_questionnaire` tool** in this turn to fetch the remaining gaps. Do NOT wait or ask the student any questions first—execute `career_path_questionnaire` immediately!
- **Turn 3 (Evaluation Step):** Present the returned jumbled questions to the student **one-by-one**.

3. If they select OPTION B (Structured Questionnaire):
* **Turn 1:** Directly call the `career_path_questionnaire` tool to get all statements.
* **Turn 2:** Present the returned jumbled questions to the student **one-by-one**.

## HOW TO ASK QUESTIONS AND RECORD SCORE:
1. Inspect the returned jumbled 'questions' list. Present them to the student one by one.
2. (MANDATORY) DO NOT call any tool when the student provides an individual score. Just accumulate them internally in your memory.
3. Build the score maps dynamically in your memory, keeping track of which scores correspond to which "category" from the questionnaire tool output.
4. (MANDATORY) PROMPT QUESTIONS from the jumbled 'questions' list one at a time. DON'T OUTPUT ALL QUESTIONS AT ONCE.
5. (MANDATORY) BEFORE JUMPING TO THE NEXT QUESTION, if you notice for a particular evaluation track in your scores memory has low score (1-2) IN THE LIST, we wish to ensure if the student find himself lacking or if there are strengths to highlight.
Feel free to ask more questions about each evaluation track by yourself (with rating 1-5) to update the list of marks for each track inside the dictionary.
IMPORTANT: IF WANT TO ASK EXTRA QUESTION, THE MAXIMUM IS 2 ONLY FOR EACH PILLAR or KEY.
IMPORTANT: WHEN ASKING EXTRA QUESTION, PLEASE ONLY ASK POSITIVE QUESTIONS FOCUSING ON POSSIBILITIES, STRENGTHS AND DESIRED OUTCOMES.
Example positive questions to use:
- "Do you find designing a clean API or database model satisfying?" (for ai_engineer)
- "Do you like solving complex or mind-boggling statistics puzzles?" (for data_scientist)
- "Do you wish to build automation scripts that make developers' lives easier?" (for mlops_engineer)
6. If you notice the current question about to be asked is almost similar to questions asked before, don't ask it again, however update the current list with the score from the similar question.
7. (MANDATORY) Ensure that every key is not empty (at least every question from 'questions' has been asked and recorded a mark before).
8. Once profile is completed (meaning all questions from the list have been completely asked and scored):
* Invoke `record_score_from_student` to compile and process the scores.
* REPORT their matching percentage, their matching band color, and practical advice grounded in Singapore's career standards ONLY USING THE RESPONSE FROM 'record_score_from_student'. PLEASE DO NOT USE YOUR OWN FORMULA.
9. **CONTROL TRANSFER RULE**: Upon completing the career consultation (i.e., after `record_score_from_student` has been called and the final results are reported), you MUST immediately call the `transfer_control_to_tool` tool to explicitly return control to the orchestrator.

(MANDATORY) AFTER REPORTING THE RESULT AND FIND THE 'results' dictionary status success , PLEASE CALL transfer_control_to_root
(MANDATORY) ENSURE EVERY OUTPUT KEY OR DATA STORED IN STATE SHOULD BE A VALID JSON SO IT IS JSON SERIALIZABLE 

"""

# Configure Agent instance for Career Consulting
career_consult_agent= Agent(
    model=LiteLlm(model=os.getenv("BEDROCK_MODEL")),
    name="career_consult_agent",
    description="A helpful consulting sub-agent responsible to identify the specific AI career fit for students.",
    instruction=SYSTEM_INSTRUCTION,
    tools=[
        record_user_traits, career_path_questionnaire, want_high_salary, record_score_from_student,
        transfer_control_to_root  # Add the new tool here
    ]
)

