import os
import json
import asyncio
from typing import Dict, Any

# Google ADK imports
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# ----------------------------------------------------------------------
# IMPORT ONLY THE ROOT MANAGER AGENT (AS REQUESTED BY THE USER)
# ----------------------------------------------------------------------
from manager_agent.agent import root_agent

APP_NAME = "ai_career_guidance_ecosystem"
USER_ID = "student_user_123"
SESSION_ID = "active_student_session_001"
SESSION_CACHE_FILE = "session_store.json"

# Initialize our shared, in-memory session service
session_service = InMemorySessionService()

# Create a single Runner targeting ONLY the root supervisor/manager agent
runner = Runner(
    agent= root_agent,
    app_name=APP_NAME,
    session_service=session_service
)


def extract_response_text(events) -> str:
    """
    Utility helper to cleanly extract the final textual response
    from an ADK streaming runner event context.
    """
    for event in events:
        if event.is_final_response() and event.content:
            return event.content.parts[0].text
    return "The system processed your request successfully."


def save_session_to_disk(state: Dict[str, Any]):
    """
    Saves the session state variables to disk for persistent resumption.
    """
    try:
        # Sanitize state of non-JSON serializable elements if any
        sanitized_state = {}
        for k, v in state.items():
            # Avoid storing nested complex ADK system pointers if any
            if k :
                sanitized_state[k] = v

                # in ["questions_queue", "scores_collected", "recorded_profile", "career_fit_results",
                #     "compatibility_results", "learning_nodes", "career_fit", "roadmap_initialized",
                #     "intention_recorder", "intention"]

        with open(SESSION_CACHE_FILE, "w") as f:
            json.dump(sanitized_state,f,indent=2)
        print(f"\n[PERSISTENCE] Session state successfully serialized to '{SESSION_CACHE_FILE}'.")
    except Exception as e:
        print(f"\n[PERSISTENCE WARNING] Could not serialize session: {e}")


def load_session_from_disk() -> Dict[str, Any]:
    """
    Loads persistent session state variables from disk.
    """
    if os.path.exists(SESSION_CACHE_FILE):
        try:
            with open(SESSION_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[PERSISTENCE WARNING] Could not deserialize session file: {e}")
    return {}


async def run_chat_loop():
    print("======================================================================")
    print("      SINGAPORE STATEFUL AI CAREER GUIDANCE ECOSYSTEM CHAT            ")
    print("======================================================================")
    print("Loading control tower orchestrator...")

    # Check if there is an existing session saved on disk
    saved_state = load_session_from_disk()
    use_saved = False

    if saved_state:
        print(f"\n[PERSISTENCE] Found a saved previous guidance session from disk.")
        ans = input("Would you like to resume your previous upskilling progress? (y/n): ").strip().lower()
        if ans in ["y", "yes"]:
            use_saved = True
            print("[PERSISTENCE] Restoring previous profile, compatibility metrics, and roadmap tracker...")
        else:
            print("[PERSISTENCE] Starting a fresh session.")

    # Create session state
    initial_state = saved_state if use_saved else {}
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
        state=initial_state
    )

    print("\nOrchestrator online. Type 'exit' or 'quit' at any time to save progress.")

    # If starting fresh, send a silent initial greeting "hi" to welcome the student
    first_turn = True

    while True:
        if first_turn and not use_saved:
            user_input = "hi"
            first_turn = False
        else:
            user_input = input("\n[Student]: ").strip()

        if not user_input:
            continue

        # Check for manual exit keywords
        if user_input.lower() in ["exit", "quit", "save progress"]:
            print("\nGracefully exiting career ecosystem...")
            # Fetch latest state and save to disk
            current_session = await session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
            if current_session:
                save_session_to_disk(current_session.state)
            print("Goodbye!")
            break

        # Deliver message to the orchestrator runner
        content = types.Content(role="user", parts=[types.Part(text=user_input)])

        try:
            events = runner.run(
                user_id=USER_ID,
                session_id=SESSION_ID,
                new_message=content
            )
            response_text = extract_response_text(events)

            # Print the active agent who processed the turn if logged or noted in state
            current_session = await session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
            active_agent = "AI Orchestrator"

            if current_session:
                # Inspect active routed agent
                # Check if suspension was requested from tool
                if current_session.state.get("session_suspended"):
                    save_session_to_disk(current_session.state)
                    # Reset suspension flag
                    current_session.state["session_suspended"] = False
                    await session_service.update_session_state(APP_NAME, USER_ID, SESSION_ID, current_session.state)

            print(f"\n[{active_agent}]: {response_text}")

        except Exception as e:
            print(f"\n[SYSTEM ERROR] An execution error occurred in the multi-agent graph: {e}")
            break


if __name__ == "__main__":
    asyncio.run(run_chat_loop())
