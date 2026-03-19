# pip install streamlit python-dotenv crewai crewai-tools langchain-openai

# streamlit_app.py

# === Standard Imports ===
import os
import streamlit as st
from dotenv import load_dotenv

# === CrewAI Core ===
from crewai import Agent, Task, Crew, LLM

# === Tools ===
from crewai_tools import CodeInterpreterTool, SerperDevTool
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

# === NVIDIA (OpenAI-compatible client) ===
from openai import OpenAI

# === Load ENV ===
load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# === Validate Keys ===
if not NVIDIA_API_KEY or not SERPER_API_KEY:
    st.error("❌ Missing API keys! Set NVIDIA_API_KEY and SERPER_API_KEY in .env")
    st.stop()

# === NVIDIA Client (for direct API calls / custom tools) ===
# FIX 1: Use the variable, not the literal string "$NVIDIA_API_KEY"
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY  # ✅ was: api_key="$NVIDIA_API_KEY"
)

# FIX 2: Removed the module-level streaming test block — it ran at startup,
#         printed to stdout (not Streamlit), and crashed the app before the UI loaded.

# === LLM for CrewAI Agents ===
# FIX 3: Use crewai.LLM instead of langchain_openai.ChatOpenAI.
#         CrewAI's LLM class routes through LiteLLM; prefix "openai/" tells
#         LiteLLM to use the OpenAI-compatible endpoint, which NVIDIA exposes.
llm = LLM(
    model="openai/meta/llama-3.3-70b-instruct",   # ✅ valid NVIDIA-hosted model
    api_key=NVIDIA_API_KEY,
    base_url="https://integrate.api.nvidia.com/v1",
    temperature=0.3
)

# === Tools Initialization ===
search_tool = SerperDevTool(api_key=SERPER_API_KEY)
code_tool = CodeInterpreterTool()

# === Custom Summarizer Tool ===
class SummarizeToolInput(BaseModel):
    description: str = Field(..., description="Text to summarize")

class SummarizeTool(BaseTool):
    name: str = "Summarizer"
    description: str = "Summarizes text using NVIDIA LLM"
    args_schema: type[BaseModel] = SummarizeToolInput

    def _run(self, description: str) -> str:
        try:
            # FIX 4: Use a valid NVIDIA-hosted model, not a HuggingFace path.
            #         "huggingface/dphn/Dolphin-Mistral-24B-Venice-Edition" caused
            #         a 401 AuthenticationError because it routed to HuggingFace,
            #         not NVIDIA.
            response = client.chat.completions.create(
                model="meta/llama-3.3-70b-instruct",  # ✅ valid NVIDIA model
                messages=[
                    {"role": "system", "content": "You are a helpful summarization assistant."},
                    {"role": "user", "content": f"Summarize this:\n\n{description}"}
                ],
                temperature=0.3,
                max_tokens=200
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Summarization error: {e}"

summarize_tool = SummarizeTool()

# === Tool Mapping ===
TOOL_MAP = {
    "search": search_tool,
    "code": code_tool,
    "summarize": summarize_tool
}

# === Streamlit UI ===
st.set_page_config(page_title="🧠 AI Agent Workflow Builder", layout="centered")

st.title("🧠 AI Agent Workflow Builder")
st.markdown("""
Build and launch a custom crew of AI agents powered by CrewAI.  
Each agent can use tools like web search, Python execution, and summarization.
""")

# === Task Input ===
task_description = st.text_input(
    "📝 What should the agents work on?",
    value="Research the latest advancements in generative AI and summarize them."
)

# === Number of Agents ===
num_agents = st.slider("👥 Number of agents", 2, 5, 3)

# === Agent Config ===
st.markdown("---")
st.subheader("⚙️ Configure Each Agent")

agent_configs = []

for i in range(num_agents):
    with st.expander(f"Agent {i+1}"):
        role = st.text_input(f"Role {i+1}", value=f"Agent {i+1}", key=f"role_{i}")
        goal = st.text_area(f"Goal {i+1}", value=f"Assist with: {task_description}", key=f"goal_{i}")
        tools = st.multiselect(
            f"Tools {i+1}",
            options=list(TOOL_MAP.keys()),
            default=["search"],
            key=f"tools_{i}"
        )

        agent_configs.append({
            "role": role,
            "goal": goal,
            "tools": tools
        })

# === Run Crew ===
if st.button("🚀 Launch Crew Sequentially"):
    st.info("🛠️ Running agents...")

    agents = []

    # Create Agents
    for config in agent_configs:
        selected_tools = [TOOL_MAP[t] for t in config["tools"]]

        agent = Agent(
            role=config["role"],
            goal=config["goal"],
            tools=selected_tools,
            llm=llm,  # ✅ crewai.LLM instance pointing at NVIDIA
            backstory=f"{config['role']} is collaborating on this project.",
            verbose=True
        )
        agents.append(agent)

    # Sequential Execution
    current_input = task_description
    results = []

    for i, agent in enumerate(agents):
        task = Task(
            description=current_input,
            agent=agent,
            expected_output=f"Output from {agent.role}"
        )

        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=True
        )

        with st.spinner(f"🤖 {agent.role} is working..."):
            output = crew.kickoff()

        results.append((agent.role, output))
        current_input = str(output)  # ensure string for next agent's input

    # === Final Output ===
    st.success("✅ All agents completed tasks!")
    st.subheader("📄 Final Output")
    st.write(current_input)

    # === Debug Outputs ===
    with st.expander("🧾 Full Agent Outputs"):
        for role, output in results:
            st.markdown(f"**{role}**")
            st.write(output)