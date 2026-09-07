# AI CAREER CONSULANT

**FILES DESCRIPTION**
- runner.py (local run) :
  > Python code to create and manage stateful sessions in the Agent Development Kit (ADK), enabling our agents to maintain context and   remember user information across interactions.
- .env :
  >Environment variable storing AWS_PROFILE
- intention_enum.py:
  > An Enum class recording three types of student intention using our application
- unit_test:
  > Record all the unit test for sub-agents using raw json
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

**HOW TO RUN**
1. Download the whole GitHub repositories
2. __EITHER__ Run runner.py using any Python platform
   > __OR__ Install google-adk in terminal using pip, then type and excecute adk run manager_agent in terminal 
   >(Refer to ADK documentation if need assistance)
   >

# OVERALL PROCESS
1. Student reveal his/her intention of using this application : 
  > *Type 1: Curious about AI but didn't know if they are suitable for this role or not
>  *Type 2: Confident to join AI field, but not sure what specialization to choose
> *Type 3: Confident about the specific AI career in the future, but didn't know how to start
2. intention_identifier_agent will determine which type of intention the student possess. For type 3, the sub-agent also record the specific AI career the student wanted (ex: AI engineer or Data Scientist)
3. root_agent will determine which sub_agent should be called next depends on the type of intention recorded
> * If type 1, ai_compatibility_test_agent will be called
>   
>* If type 2, career_consult_agent will be called
>  
>* If type 3, roadmap_agent will be called
>  
4. If ai_compatibility_test_agent is called:
>  Test the student's compatibility with AI field using four scopes : coding ,math, debugging patience and salary preference
>  * Option A: Student can prompt their preference or disinterest and the agent will analyze and update on the founded scopes from the student's prompt. If some scopes are not covered, some extra questions will be asked to understand the student's opinion on the missed scopes.
>  * Option B: Student are given questionnaire covering all four scopes at once.
>    
>   After both options are completed, a suitability band will be release to determine if the student is capable to pursue in AI field
5. If career_consult_agent is called:
>* The procedure is similar to ai_compatibility_test_agent but with the aim to figure out the optimal AI career for a student.
>* The agent is equipped with complex scoring system, more available scopes and more final options (available AI specialization in the market)
6. If roadmap_agent is called:
>*  Depends on the specific career the student decided, a pre-set roadmap will be given to the student
>* Each roadmap contains 4-5 learning scopes with finer topics under it, also consisting of a progression bar
>* Student can:
>>- Express their preference to study any of the learning scopes first -> the learning of this particular scope will be given high priority)
>>- Notify the agent the completion of any of the topics -> The topics will be marked completed and the progression bar given will be updated
>>- Ask description for each scope or topics if the student didn't understand -> Agent will explain in detailed and show some resources material
>>- Ask for relevant websites or physical book to study a specific topic -> web_search_tool will search up-to-date website or pyhsical books for student
7. Persistent memory storage using InMemorySessionService
 
