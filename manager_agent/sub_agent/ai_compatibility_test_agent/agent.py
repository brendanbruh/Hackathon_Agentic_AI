import os
import random
from typing import Dict, Any, List, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Google ADK imports
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
    Helps identify foundational computer science and mathematical compatibility indicators.
    """
    traits_category: Literal[
        "math_and_logic", "programming", "debugging_patience", "continuous_learning", "salary_preference"]
    status: Literal["like", "dislike", "neutral", "familiar", "experienced", "beginner", "unmentioned"]
    willing_to_learn: bool
    evidence: str = Field(
        description="The exact quote or logical reasoning from the user indicating this preference."
    )


class CompatibilityScores(BaseModel):
    """
    Structured inputs for questionnaire scoring.
    Enforces a strict 1-to-5 Likert scale interface for compiling student scores.
    """
    math_and_logic: List[int] = Field(default_factory=list,
                                      description="Scores (1 to 5) for math/analytical/logical statements.")
    programming: List[int] = Field(default_factory=list,
                                   description="Scores (1 to 5) for software development and hands-on coding statements.")
    debugging_patience: List[int] = Field(default_factory=list,
                                          description="Scores (1 to 5) for troubleshooting, trial-and-error, and debugging resilience.")
    continuous_learning: List[int] = Field(default_factory=list,
                                           description="Scores (1 to 5) for continuous learning, framework curiosity, and upskilling.")


# ----------------------------------------------------------------------
# 2. DEFINE NATIVE TOOL FUNCTIONS & STATE MACHINE
# ----------------------------------------------------------------------

def reset_session(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Resets the session state, clearing previous recorded traits, scores, and evaluations.
    Call this when the student starts a brand-new session, says 'hi', 'hello', 'start over', or 'reset'.
    """
    tool_context.state["recorded_profile"] = {}
    tool_context.state["salary_preference"] = False
    tool_context.state["compatibility_results"] = {}
    tool_context.state["questions_queue"] = []
    tool_context.state["current_question_index"] = 0
    tool_context.state["scores_collected"] = {
        "math_and_logic": [],
        "programming": [],
        "debugging_patience": [],
        "continuous_learning": []
    }
    tool_context.state["is_probing"] = False
    tool_context.state["current_probing_pillar"] = None
    print("\n[STATE MACHINE] Session state reset completed.")
    return {"status": "success", "message": "Session state has been completely reset."}


def record_user_traits(traits: List[UserTrait], tool_context: ToolContext = ToolContext) -> Dict[str, Any]:
    """
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
    print(f"\n[STATE MACHINE] Saved traits to state: {recorded_profile}")
    return {"status": "success", "recorded_profile": recorded_profile}


def want_high_salary(tool_context: ToolContext) -> Dict[Any, Any]:
    """Marks that the student explicitly desires a high salary path."""
    tool_context.state["salary_preference"] = True
    print("\n[STATE MACHINE] Salary preference marked high (True)")
    return {}


def start_compatibility_assessment(tool_context: ToolContext) -> Dict[str, Any]:
    """
    100% DETERMINISTIC GUARANTEE: Initializes the question queue programmatically based on the student's
    extracted profile, randomizes the order, and saves the state to ensure the queue is strictly followed.
    """
    all_questions = {
        "math_and_logic": [
            "I enjoy solving problems where the solution is not immediately obvious.",
            "I am comfortable with mathematics, algorithms, and analytical thinking."
        ],
        "programming": [
            "I enjoy writing and debugging code to solve problems.",
            "I enjoy combining different technologies to build a complete system."
        ],
        "debugging_patience": [
            "I enjoy troubleshooting systems when something does not work as expected.",
            "I enjoy experimenting with different approaches to find the best solution."
        ],
        "continuous_learning": [
            "I am comfortable learning new programming languages, frameworks, and technologies continuously.",
            "I enjoy keeping up with tech news and tinkering with new tools on weekends."
        ],
        "salary_preference": [
            "Is a starting junior salary of SGD 6,000 - SGD 8,500/month aligned with your expectations?"
        ]
    }
    if "recorded_profile" not in tool_context.state:
        tool_context.state["recorded_profile"] = {}
    recorded_profile = tool_context.state.get("recorded_profile", {})

    # Filter categories that the student hasn't addressed yet
    selected_categories = []
    core_pillars = ["math_and_logic", "programming", "debugging_patience", "continuous_learning"]
    for pillar in core_pillars:
        # Core fix: always include unaddressed/unmentioned categories
        if (pillar not in recorded_profile) or (recorded_profile[pillar].get("status") == "unmentioned"):
            selected_categories.append(pillar)

    # Include salary preference statement if not already saved in state
    if "salary_preference" not in tool_context.state:
        selected_categories.append("salary_preference")

    # Flatten and jumble all questions into a single flat list
    flat_questions = []
    for cat in selected_categories:
        for q in all_questions[cat]:
            flat_questions.append({
                "text": q,
                "category": cat
            })

    # Shuffle to prevent status and category bias
    random.shuffle(flat_questions)

    # Initialize queue states in session state
    tool_context.state["questions_queue"] = flat_questions
    tool_context.state["current_question_index"] = 0
    tool_context.state["scores_collected"] = {
        "math_and_logic": [],
        "programming": [],
        "debugging_patience": [],
        "continuous_learning": []
    }
    tool_context.state["is_probing"] = False
    tool_context.state["current_probing_pillar"] = None

    print(f"\n[STATE MACHINE] Dynamic queue initialized with {len(flat_questions)} questions.")

    if not flat_questions:
        results = evaluate_results_from_state(tool_context)
        # **NEW: Transfer control to root after completion**
        return {
            "status": "completed",
            "message": "All categories have been pre-filled from your profile! Evaluation completed.",
            "results": results
        }

    first_question = flat_questions[0]  # Original line was `first_question = flat_questions` which is a list, needs ``
    return {
        "status": "next_question",
        "text": first_question["text"],
        "category": first_question["category"],
        "index": 1,
        "total": len(flat_questions),
        "instruction": "Please rate this statement from 1 (Strongly Disagree) to 5 (Strongly Agree)."
    }


def submit_single_rating(rating: int, tool_context: ToolContext) -> Dict[str, Any]:
    """
    100% DETERMINISTIC GUARANTEE: Submits rating for the current active statement,
    manages the pointer index, triggers probing questions dynamically on scores of 1-2,
    and returns to the standard queue automatically, ensuring the flow is never lost.
    """
    if "questions_queue" not in tool_context.state or "current_question_index" not in tool_context.state:
        return {
            "status": "error",
            "message": "Assessment has not been initialized. Please execute start_compatibility_assessment first."
        }

    # Constrain rating to 1-5
    rating = max(1, min(5, rating))

    queue = tool_context.state["questions_queue"]
    idx = tool_context.state["current_question_index"]
    scores = tool_context.state.get("scores_collected", {
        "math_and_logic": [],
        "programming": [],
        "debugging_patience": [],
        "continuous_learning": []
    })

    # IndexError Prevention Guard: If rating is submitted after queue is fully completed
    if idx >= len(queue):
        results = evaluate_results_from_state(tool_context)
        # **NEW: Transfer control to root after completion**
        tool_context.actions.call_tool("transfer_control_to_root", {})
        return {
            "status": "completed",
            "message": "The assessment is already completed! Here are your compatibility results:",
            "results": results
        }

    # Predefined positive probing questions for low ratings (1-2)
    probing_questions = {
        "math_and_logic": "Do you enjoy breaking down complex logical puzzles or playing strategic board games?",
        "programming": "Have you ever felt a sense of pride or excitement when seeing a program you wrote run successfully?",
        "debugging_patience": "Do you feel a sense of deep satisfaction when you finally locate and fix a tricky bug in code?",
        "continuous_learning": "Do you enjoy watching tech videos or reading about new technology trends online?"
    }

    # If the active state is in "probing" mode
    if tool_context.state.get("is_probing", False):
        pillar = tool_context.state.get("current_probing_pillar")
        if pillar and pillar in scores:
            scores[pillar].append(rating)
        print(f"[STATE MACHINE] Recorded probing score {rating} for pillar '{pillar}'")

        # Turn off probing mode
        tool_context.state["is_probing"] = False
        tool_context.state["current_probing_pillar"] = None

        # Safely increment the index to return to the standard queue
        idx += 1
        tool_context.state["current_question_index"] = idx
    else:
        # Standard questionnaire processing
        current_q = queue[idx]
        category = current_q["category"]
        if category in scores:
            scores[category].append(rating)
            print(f"[STATE MACHINE] Recorded standard score {rating} for pillar '{category}'")
        elif category == "salary_preference":
            if rating >= 4:
                tool_context.state["salary_preference"] = True
            print("[STATE MACHINE] High salary alignment marked True")

        # Trigger dynamic probing if score is low (1-2) and category is probeable
        if rating <= 2 and category in probing_questions:
            tool_context.state["is_probing"] = True
            tool_context.state["current_probing_pillar"] = category
            # Explicitly persist scores state before returning
            tool_context.state["scores_collected"] = scores
            probe_text = probing_questions[category]
            print(f"[STATE MACHINE] Low score detected on '{category}'. Triggering positive probe.")
            return {
                "status": "probing_question",
                "text": probe_text,
                "pillar": category,
                "instruction": "Since you rated the previous statement low, let's explore your strengths. Rate this from 1 (Strongly Disagree) to 5 (Strongly Agree):"
            }
        else:
            # Increment pointer index to advance the queue
            idx += 1
            tool_context.state["current_question_index"] = idx

    # Explicitly write back the scores to session state so nested changes get serialized!
    tool_context.state["scores_collected"] = scores

    # Check if there are more statements in the queue
    if idx < len(queue):
        next_q = queue[idx]
        return {
            "status": "next_question",
            "text": next_q["text"],
            "category": next_q["category"],
            "index": idx + 1,
            "total": len(queue),
            "instruction": "Please rate this statement from 1 (Strongly Disagree) to 5 (Strongly Agree)."
        }
    else:
        # Queue is exhausted! Automatically execute scoring calculations
        results = evaluate_results_from_state(tool_context)
        return {
            "status": "completed",
            "message": "All statements have been rated successfully! Here are your compatibility results:",
            "results": results
        }


def evaluate_results_from_state(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Core scoring function running programmatically on accumulated state variables.
    """
    pillars = [["math_and_logic", "Math & Logic"], ["programming", "Programming Intensity"],
               ["debugging_patience", "Debugging Resilience"], ["continuous_learning", "Continuous Learning"]]

    score = 0.0
    known_from_prompt = tool_context.state.get("recorded_profile", {})
    scores_collected = tool_context.state.get("scores_collected", {})

    status_score_map = {
        "dislike": DISLIKE,
        "neutral": NEUTRAL,
        "beginner": 2.0,
        "familiar": WILLING_TO_LEARN,
        "like": LIKE,
        "experienced": EXPERIENCED
    }

    pillar_breakdown = {}
    for pillar_key, pillar_name in pillars:
        ratings = scores_collected.get(pillar_key, [])
        if ratings and len(ratings) > 0:
            pillar_avg = sum(ratings) / len(ratings)
            score += pillar_avg
            pillar_breakdown[pillar_name] = round(pillar_avg, 2)
        elif pillar_key in known_from_prompt:
            trait = known_from_prompt[pillar_key]
            status = trait.get("status", "neutral").lower()
            pillar_val = float(status_score_map.get(status, NEUTRAL))
            score += pillar_val
            pillar_breakdown[pillar_name] = pillar_val
        else:
            score += float(NEUTRAL)
            pillar_breakdown[pillar_name] = float(NEUTRAL)

    if tool_context.state.get("salary_preference", False):
        score += 4.0

    MAX_SCORE = 20.0
    suitability_pct = (score / MAX_SCORE) * 100.0
    suitability_pct = min(100.0, suitability_pct)

    if suitability_pct >= 80:
        band = "Strong AI Signal (Green Match)"
        advice = (
            "You show a stellar natural alignment for an AI career! Your interests in programming, "
            "problem-solving, and continuous experimentation mean you will likely thrive in technical "
            "AI Engineering roles. A starting junior salary of SGD 6,000 - 8,500 is very much aligned!"
        )
    elif suitability_pct >= 60:
        band = "Foundational Match (Orange Match)"
        advice = (
            "You have a solid foundation! However, you may need additional technical or mathematical skill development. "
            "If you dislike deep mathematical theory but love building apps, you might want to pivot towards applied "
            "generative AI development (using API integrations and frameworks) or AI Product Management."
        )
    else:
        band = "Low Current Alignment"
        advice = (
            "A highly technical AI engineering path might feel frustrating right now. Consider standard software "
            "engineering, UX design, or other tech domains first, or explore low-code AI integrations to test the waters."
        )

    results = {
        "status": "success",
        "suitability_percentage": round(suitability_pct, 1),
        "evaluation_band": band,
        "pillar_breakdowns": {
            "Math & Logic": pillar_breakdown.get("Math & Logic", 2.0),
            "Programming Intensity": pillar_breakdown.get("Programming Intensity", 2.0),
            "Debugging Resilience": pillar_breakdown.get("Debugging Resilience", 2.0),
            "Continuous Learning": pillar_breakdown.get("Continuous Learning", 2.0),
        },
        "career_advice": advice
    }
    tool_context.state["compatibility_results"] = results
    return results


def record_score_from_student(scores: CompatibilityScores, tool_context: ToolContext) -> Dict[str, Any]:
    """
    Backward-compatibility wrapper. If an external process or old logic attempts to call this directly,
    it maps the ratings to state and programmatically triggers evaluation.
    """
    scores_collected = tool_context.state.get("scores_collected", {
        "math_and_logic": [],
        "programming": [],
        "debugging_patience": [],
        "continuous_learning": []
    })
    if scores.math_and_logic:
        scores_collected["math_and_logic"] = scores.math_and_logic
    if scores.programming:
        scores_collected["programming"] = scores.programming
    if scores.debugging_patience:
        scores_collected["debugging_patience"] = scores.debugging_patience
    if scores.continuous_learning:
        scores_collected["continuous_learning"] = scores.continuous_learning
    tool_context.state["scores_collected"] = scores_collected

    return evaluate_results_from_state(tool_context)


# ----------------------------------------------------------------------
# 3. CONFIGURE THE LLM AGENT (SYSTEM PROMPT)
# ----------------------------------------------------------------------

SYSTEM_INSTRUCTION = """You are the "AI Compatibility Test Agent", an empathetic, supportive, yet analytical career counselor.
Your objective is to evaluate if a student is well-suited to pursue a career in Artificial Intelligence based on their skills, personality, and work style.
## EVALUATION PILLARS
You must evaluate the student across these foundational criteria:
1. Math & Logic: Does the student enjoy linear algebra, calculus, or statistics?
2. Programming & Application: Do they prefer building working apps over pure theory?
3. Debugging Patience: Do they have the patience for debugging, data cleaning, and trial-and-error?
4. Expectation: Are starting salaries of SGD 6,000–8,500/month aligned with their expectations?'

## CRITICAL RULES FOR STATE-MACHINE SYSTEM TOOLS
1. DONT COMPUTE OR GUESS RESULTS: You are strictly forbidden from maintaining question indices, deciding when to pause for low scores, or calculating matching percentages on your own. Let the Python state machine tools handle the entire loop deterministically!
2. NO AUTONOMOUS OR AUTOMATIC SCORING (STRICT NO-AUTO RULE):
- You are strictly forbidden from inventing, simulating, or automatically generating ratings (1-5) on behalf of the student.
- For every statement presented, you must write the statement out in the chat and then STOP and WAIT for the user's manual response/rating.
- You MUST ONLY execute `submit_single_rating(rating=...)` after the user has explicitly typed a rating (1-5) in response to that specific statement. Do NOT invoke `submit_single_rating` with prefilled, auto-calculated, or simulated numbers!
3. INITIAL GREETING RULE (NO PREMATURE TOOL CALLS):
- You are strictly forbidden from executing any tool calls (like `record_user_traits` or `start_compatibility_assessment`) on the very first turn when the user says a simple greeting (e.g. "hi", "hello", "hey", "start").
- You must only reset the state by running `reset_session`, welcome the user warmly, present Option A and Option B clearly, and wait for their choice.
4. COMPULSORY QUESTIONNAIRE TOOL EXECUTION:
- You are strictly prohibited from printing, generating, listing, or guessing questionnaire statements on your own from your memory.
- To get the questions, you MUST call the `start_compatibility_assessment` tool to initiate the queue.
- You CANNOT ask the student even a single statement from the questionnaire without first executing the `start_compatibility_assessment` tool!
5. (MANDATORY) **CONTROL TRANSFER RULE**: Upon completing the compatibility assessment (i.e., when the final results are reported after `submit_single_rating` returns `status: 'completed'`), you MUST immediately call the `transfer_control_to_root` tool to explicitly return control to the orchestrator.
6. (MANDATORY) ENSURE EVERY OUTPUT KEY OR DATA STORED IN STATE SHOULD BE A VALID JSON SO IT IS JSON SERIALIZABLE 


## MASKING & JUMBLED WORKFLOW RULES (ANTI-BIAS)
1. STRICT MASKING OF PATH LABELS: To ensure the student is completely unbiased, NEVER mention, output, or imply what category (e.g. math_and_logic, programming, debugging_patience) any question maps to.
2. ONE-BY-ONE presentation: Present the statements strictly one at a time. Wait for the user to reply before presenting the next statement.
## CONVERSATIONAL WORKFLOW (STRICT TURN-BY-TURN PROTOCOL)
1. Welcome the student warmly. Suggest they share their interests, basic programming backgrounds, favorite data types, or preferred starting salary.
2. Provide them with two clear choices to begin:
* OPTION A: Type a natural language prompt explaining their background.
* OPTION B: Take a structured questionnaire rating statements on a scale of 1 to 5.
3. **If the user chooses OPTION A (Natural Language Prompt):**
- **Turn 1 (Prompt for Bio):** Warmly ask the student to describe their background, skills, interests, and what they enjoy/dislike in as much detail as possible. Do NOT call any other tools yet! Stop and wait for the student's background text.
- **Turn 2 (Process Bio):** Once the student provides their background text:
* Immediately call the `record_user_traits` tool to save the student's traits, even if they are a complete beginner.
* Parse their response for any general computing backgrounds, Python skills, or math comfort:
- If they mention writing Python scripts, generate a UserTrait for `programming` with status "beginner" or "familiar", and willing_to_learn true.
- If they mention calculus or algebra comfort, generate a UserTrait for `math_and_logic` with status "familiar" or "like", and willing_to_learn true.
* In parallel in this turn, call `want_high_salary` ONLY WHEN STUDENT MENTIONED A HIGH AMOUNT OF SALARY. DONT call it if they didn't.
* Do NOT call `start_compatibility_assessment` in parallel in this turn to avoid state race conditions.
- **Turn 3 (Fetch Questionnaire Gaps):** Once you receive the `success` response of `record_user_traits` in the tool context, you MUST immediately call the `start_compatibility_assessment` tool in this turn to fetch the remaining gaps. Do NOT wait or ask any questions first!
- **Turn 4+ (Queue Progression):** Inspect the returned statement text from the tool response. Present it to the student and stop and wait for their manual rating response.
4. **If the user chooses OPTION B (Structured Questionnaire):**
- **Turn 1 (Initialize State Machine):** Directly call the `start_compatibility_assessment` tool to initialize the state machine queue.
- **Turn 2+ (Queue Progression):** Present the first returned statement to the student. STOP and WAIT for their manual rating (1-5) input.
5. **State-Machine Progression Rule:**
- Every time the student provides a score for a statement:
* Immediately execute `submit_single_rating(rating=...)` with that score.
* Inspect the returned tool response:
- If status is `"next_question"`: Print the returned statement (`text`) and wait for their rating.
- If status is `"probing_question"`: Print the returned probe statement (`text`) and its prompt instruction, and wait for their rating.
- If status is `"completed"`: The assessment is done! Report their suitability percentage, evaluation band, breakdown metrics, and Singapore salary-grounded advice exactly as returned by the tool.

(MANDATORY) AFTER REPORTING THE RESULT AND FIND THE 'result' dictionary status COMPLETED, PLEASE CALL transfer_control_to_root
(MANDATORY) ENSURE EVERY OUTPUT KEY OR DATA STORED IN STATE SHOULD BE A VALID JSON SO IT IS JSON SERIALIZABLE 

"""

# Configure Agent instance for Compatibility Assessment
ai_compatibility_test_agent = Agent(
    model=LiteLlm(model=os.getenv("BEDROCK_MODEL")),
    name="ai_compatibility_test_agent",
    description="A strategic consulting agent responsible to identify if a student is suited to pursue an AI career.",
    instruction=SYSTEM_INSTRUCTION,
    tools=[
        record_user_traits, start_compatibility_assessment, submit_single_rating, want_high_salary,
        record_score_from_student, reset_session,
        transfer_control_to_root  # Add the new tool here
    ]
)

