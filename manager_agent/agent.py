import os
from typing import Dict, Any, Literal

from litellm.llms.sap.chat import models
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Google ADK imports
from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import ToolContext

# Import the sub-agents directly
# Assuming these imports correctly point to the updated files
from .sub_agent.ai_compatibility_test_agent import ai_compatibility_test_agent
from .sub_agent.career_consult_agent import career_consult_agent
from .sub_agent.roadmap_agent import roadmap_agent
from .sub_agent.intention_identifier_agent import intention_identifier_agent

load_dotenv(".env")

class IntentionContent(BaseModel):
    intention: str = Field(description="The student's identified intention. '1' for Suitability Diagnostic, '2' for Career Consulting, '3' for Roadmap.")
    specific_career: str = Field(default=None, description="The specific AI career path the student is interested in.")

def check_state_and_route(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Checks the current session state variables (such as 'intention_recorder', 'interested_in_AI',
    'compatibility_results', and 'career_fit_results') to determine the next agent to route the student to.
    Automatically sets the tool_context.actions.transfer_to_agent variable to execute the handoff.
    """
    state = tool_context.state

    # Retrieve the intention_recorder stored by the intention identifier agent
    intention_recorder = state.get("intention_recorder")
    intention_val = None
    specific_career = None

    if intention_recorder:
        if isinstance(intention_recorder, dict):
            intention_val = intention_recorder.get("intention")
            specific_career = intention_recorder.get("specific_career") or intention_recorder.get("specific_path")
        else:
            # Handle Pydantic or object attributes
            intention_val = getattr(intention_recorder, "intention", None)
            if intention_val and hasattr(intention_val, "value"):
                intention_val = intention_val.value
            specific_career = getattr(intention_recorder, "specific_career", None)

    compatibility_results = state.get("compatibility_results")
    career_fit_results = state.get("career_fit_results")

    print(f"\n[MANAGER STATE GATES CHECK]")
    print(f" - intention_recorder: {intention_recorder}")
    print(f" - extracted intention: '{intention_val}'")
    print(f" - compatibility results exist: {bool(compatibility_results)}")
    print(f" - career fit results exist: {bool(career_fit_results)}")

    # Bridge any spelling differences between sub-agents to avoid key mismatches
    if career_fit_results:
        primary_match = career_fit_results.get("primary_match") or career_fit_results.get("primary_recommendation")
        if primary_match:
            career_fit_results["primary_recommendation"] = primary_match
            career_fit_results["primary_match"] = primary_match
        state["career_fit"] = primary_match

    # RULE 1: If no intention has been parsed yet, run the intention identifier agent
    # This rule is primarily handled by the prompt, which elicits the intention.
    # If the manager receives control back and intention is still not set, it means the
    # manager's prompt for elicitation was just given, and the next user input needs
    # to be processed by intention_identifier_agent.
    # Or, if an agent returns control without setting intention.
    if not intention_val:
        tool_context.actions.transfer_to_agent = "intention_identifier_agent"
        return {
            "status": "routing",
            "target_agent": "intention_identifier_agent",
            "message": "Student intention is not identified yet. Routing to 'intention_identifier_agent' for parsing."
        }


    intention_str = str(intention_val)

    # RULE 2: If is Intention(1), evaluate suitability diagnostic first
    if "1" in intention_str or "UNCERTAIN_GENERAL" in intention_str:
        if not compatibility_results:
            tool_context.actions.transfer_to_agent = "ai_compatibility_test_agent"
            return {
                "status": "routing",
                "target_agent": "ai_compatibility_test_agent",
                "message": "Intention 1: Checking compatibility. Routing to 'ai_compatibility_test_agent'."
            }
        else:
            # Test if the student is suitable to pursue an AI career
            suitability_pct = compatibility_results.get("suitability_percentage", 0.0)
            is_suitable = suitability_pct >= 60.0
            if not is_suitable:
                state["session_suspended"] = True
                return {
                    "status": "completed",
                    "target_agent": "none",
                    "message": f"Suitability score is {suitability_pct}% (Low current match). Ending guidance session."
                }
            else:
                # If yes, call ai_consult_agent (the rest is like 3)
                if not career_fit_results:
                    tool_context.actions.transfer_to_agent = "career_consult_agent"
                    return {
                        "status": "routing",
                        "target_agent": "career_consult_agent",
                        "message": "Student is suitable! Routing to 'ai_consult_agent' to determine career track."
                    }
                else:
                    tool_context.actions.transfer_to_agent = "roadmap_agent"
                    return {
                        "status": "routing",
                        "target_agent": "roadmap_agent",
                        "message": f"Routing to 'ai_roadmap_agent' to build roadmap for {state.get('career_fit')}."
                    }

    # RULE 3: If is Intention(2), call ai_consult_agent to find specialized career fit
    if "2" in intention_str or "UNCERTAIN_SPECIFIC" in intention_str:
        if not career_fit_results:
            tool_context.actions.transfer_to_agent = "career_consult_agent"
            return {
                "status": "routing",
                "target_agent": "career_consult_agent",
                "message": "Intention 2: Routing to 'career_consult_agent' to discover optimal AI specialization."
            }
        else:
            # Career fit found, transfer to roadmap agent
            if "career_fit" not in state:
                state["career_fit"] = career_fit_results.get("primary_match") or career_fit_results.get("primary_recommendation")
            tool_context.actions.transfer_to_agent = "roadmap_agent"
            return {
                "status": "routing",
                "target_agent": "roadmap_agent",
                "message": f"Career track '{state['career_fit']}' determined. Routing to 'roadmap_agent'."
            }

    # RULE 4: If is Intention(3), call roadmap_agent directly and let them personalize it
    if "3" in intention_str or "CERTAIN" in intention_str:
        career_fit = specific_career or state.get("career_fit")
        if not career_fit:
            # Ask the student to clarify their career choice if not found
            return {
                "status": "need_info",
                "message": "Please specify which Singapore AI career path you want to build a roadmap for (AI Engineer, MLOps Engineer, Data Scientist, NLP / LLM Specialist)."
            }
        state["career_fit"] = career_fit
        if not career_fit_results:
            # Inject simulated result matching their specific choice so the roadmap agent initializes properly
            state["career_fit_results"] = {
                "status": "success",
                "primary_recommendation": career_fit,
                "primary_match_pct": 100.0,
                "career_advice": f"Direct customized roadmap generated for {career_fit}."
            }
        tool_context.actions.transfer_to_agent = "roadmap_agent"
        return {
            "status": "routing",
            "target_agent": "roadmap_agent",
            "message": f"Intention 3: Routing directly to 'roadmap_agent' for career: {career_fit}."
        }

    return {
        "status": "unknown",
        "message": "Orchestrator state is terminal or unknown."
    }

def trigger_graceful_exit(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Suspends the current session and requests state serialization to disk.
    Call this when the student states a keyword like 'exit' or 'quit'.
    """
    tool_context.state["session_suspended"] = True
    print("\n[MANAGER] Graceful progress save requested.")
    return {"status": "success", "message": "Session marked as suspended."}


SYSTEM_INSTRUCTION = """You are the "AI Career Orchestrator Manager", Singapore's stateful AI multi-agent advisory system.
Your task is to coordinate the workflow of 4 specialized sub-agents:
1. `intention_identifier_agent` (used to classify student's initial intent)
2. `ai_compatibility_test_agent` (evaluates general suitability for AI career path)
3. `career_consult_agent` (matches specific technical skills to specialized AI track)
4. `roadmap_agent` (builds customizable and interactive skill checklists)

## ORCHESTRATION RULES:

1.  **INITIAL GREETING & INTENTION ELICITATION**:
    - **If `tool_context.state['intention_recorder']` is NOT yet set (meaning the user's intention is unknown or this is the start of a session):**
        - Your first output MUST be a warm, professional greeting and a brief explanation of the application's capabilities.
            **Example:** "Hello! I'm your personal AI Education Consultant. I'll help you figure out your desired AI learning or career path. I have specialized agents for consulting and generating personalized roadmaps."
        - IMMEDIATELY after the greeting and explanation, you MUST offer clear examples of student intentions to guide them, and explicitly ask them to state their purpose.
            **Example:** "To get started, feel free to share your intention. Here are some common intentions from students using this application:
            1. I come here since I heard AI jobs are one of the most in-demand career right now. However, I am uncertain if I am suitable for this path.
            2. I am certain that I will pursue an AI-related career. However, I either don't know how to start to progress or which specific AI career path I should choose.
            3. I am certain that I will pursue a specific AI career path but I don't know how to start learning and progressing."
        - Do NOT call any tools or make routing decisions yourself in this initial turn. Your goal is to elicit the user's intention conversationally.

2.  **POST-INTENTION CONFIRMATION & ROUTING**:
    - **If `tool_context.state['intention_recorder']` IS set (meaning the `intention_identifier_agent` has successfully identified the user's purpose and control has returned to you):**
        - You MUST express thankfulness for their reply and summarize their identified intention using the `intention` and `specific_career` (if any) fields from `tool_context.state['intention_recorder']`.
            **Example:** "Thank you for your reply. I understand your current situation. I see that you [**summarization of intention: e.g., 'are uncertain about your suitability for an AI career' or 'are certain about an AI career but need help choosing a specific path'**]."
        - IMMEDIATELY after this summarization, you MUST then call the `check_state_and_route` tool to evaluate the session state and proceed with routing to the next appropriate specialized agent. This is a crucial step to continue the guidance process.

3.  **CONTINUOUS STATE EVALUATION & ROUTING (AFTER INITIAL INTENTION)**:
    - When a specialized sub-agent (like `ai_compatibility_test_agent` or `ai_consult_agent`) completes its task and transfers control back to you (the `manager_agent`), you MUST inspect the updated session state and IMMEDIATELY call the `check_state_and_route` tool again. This ensures that the progress made by the sub-agent is evaluated and the student is routed to the next logical step in their guidance journey.

4.  **SUSPENSION GATEWAY**:
    - If the user types "exit", "quit", or "save progress", execute `trigger_graceful_exit` to save state and gracefully suspend the session.
    
(MANDATORY) ENSURE EVERY OUTPUT KEY OR DATA STORED IN STATE SHOULD BE A VALID JSON SO IT IS JSON SERIALIZABLE 
"""

root_agent = Agent(
    model=LiteLlm(model=os.getenv("BEDROCK_MODEL")),
    name="root_agent",
    description="The primary coordinator overseeing intention identifier, compatibility diagnostic, consulting, and roadmap specialists.",
    instruction=SYSTEM_INSTRUCTION,
    sub_agents=[
        intention_identifier_agent,
        ai_compatibility_test_agent,
        career_consult_agent,
        roadmap_agent
    ],
    tools=[check_state_and_route, trigger_graceful_exit]
)