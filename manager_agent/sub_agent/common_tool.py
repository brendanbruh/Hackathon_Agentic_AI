from google.adk.tools import ToolContext
from typing import Dict, Any

def transfer_control_to_root(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Transfers control back to the manager_agent after the sub-agent has completed its task.
    """
    tool_context.actions.transfer_to_agent = "root_agent"
    print("\n[SUB-AGENT] Transferring control back to root_agent.")
    return {"status": "control_transferred", "message": "Control returned to root_agent."}
