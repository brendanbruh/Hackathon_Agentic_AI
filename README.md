# AI CAREER CONSULANT

**FILES DESCRIPTION**
- runner.py :
  > Python code to create and manage stateful sessions in the Agent Development Kit (ADK), enabling our agents to maintain context and   remember user information across interactions.
- .env :
  >Environment variable storing AWS_PROFILE
- intention_enum.py:
  > An Enum class recording three types of student intention using our application
- manager_agent:
  > Folder containing our agentic AI code for our AI career consultant
  > >Inside folder:
   >* agent.py :
  >  The orchestrator agent that control the flow of our application by deciding the best timing of using given sub-agents
  > * sub_agent file: Agentic AI code for all sub-agents
  >
  >
  >
  > > Inside sub_agent file:
  > * ai_compatibility_test_agent : Being called when need to identify if a student is suitable to pursue in AI-related field or not
  > * career_consult_agent: Being called when a student sure he/she want to pursue in AI career, but not sure which specialization he/she want to choose
  > * roadmap_agent: Being called for students that are confirmed to pursue in a specific AI career path to generate skill roadmap
  > * intention_identifier_agent : Identify intention of student to help root_agent to decide which other three sub-agents to call
  > * common_tool.py: Store transfer_control_to_root to ensure every sub-agent return control to root_agent after finishing their task

** HOW TO RUN**
1. Download the whole GitHub repositories
2. __EITHER__ Run runner.py using any Python platform
   > __OR__ Install google-adk in terminal using pip, then type and excecute adk run manager_agent in terminal 
   >(Refer to ADK documentation if need assistance)
