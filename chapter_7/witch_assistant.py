import os
from dotenv import find_dotenv, load_dotenv
from langchain.agents import load_tools
import logging
from datetime import datetime
from functools import partial

import streamlit as st
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers.openai_functions import (
    OpenAIFunctionsAgentOutputParser,
)
from langchain.callbacks.manager import tracing_v2_enabled
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streamlit import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory, StreamlitChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function
from langsmith import Client
from streamlit_feedback import streamlit_feedback

load_dotenv(find_dotenv())
os.environ["LANGCHAIN_API_KEY"] = str(os.getenv("LANGCHAIN_API_KEY"))
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "langsmith-gen-ai-book"
os.environ["SERPAPI_API_KEY"] = str(os.getenv("SERPAPI_API_KEY"))

client = Client()

llm = ChatOpenAI()

st.set_page_config(
    page_title="LangSmith Witch Demo",
    page_icon="🧙‍",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = Client()

st.subheader("🧙‍Ask the bot some questions")

llm = ChatOpenAI(model="gpt-4")


tools = load_tools(["serpapi"], llm=llm)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful modern day witch assistant. You talk and think like a witch. Current date: {time}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
).partial(time=lambda: datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"))

MEMORY = ConversationBufferMemory(
    chat_memory=StreamlitChatMessageHistory(key="langchain_messages"),
    return_messages=True,
    memory_key="chat_history",
)
agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_functions(
                x["intermediate_steps"]
            ),
            "chat_history": lambda x: x.get("chat_history") or [],
        }
        | prompt
        | llm.bind_functions(functions=[format_tool_to_openai_function(t) for t in tools])
        | OpenAIFunctionsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True)


def _submit_feedback(user_response: dict, run_id=None):
    score = {"😀": 1, "🙂": 0.75, "😐": 0.5, "🙁": 0.25, "😞": 0}.get(user_response.get("score"))
    client.create_feedback(
        run_id=run_id,
        key=user_response["type"],
        score=score,
        comment=user_response.get("text"),
        value=user_response.get("score"),
    )
    return user_response



feedback_kwargs = {
    "feedback_type": "faces",
    "optional_text_label": "LangSmith Feedback",
}
if "feedback_key" not in st.session_state:
    st.session_state.feedback_key = 0



messages = st.session_state.get("langchain_messages", [])
for i, msg in enumerate(messages):
    avatar = "🧙" if msg.type == "ai" else None
    with st.chat_message(msg.type, avatar=avatar):
        st.markdown(msg.content)
    if msg.type == "ai":
        feedback_key = f"feedback_{int(i/2)}"

        if feedback_key not in st.session_state:
            st.session_state[feedback_key] = None

        disable_with_score = (
            st.session_state[feedback_key].get("score")
            if st.session_state[feedback_key]
            else None
        )
        streamlit_feedback(
            **feedback_kwargs,
            key=feedback_key,
            disable_with_score=disable_with_score,
            on_submit=partial(
                _submit_feedback, run_id=st.session_state[f"run_{int(i/2)}"]
            ),
        )


if prompt := st.chat_input(placeholder="Alright, shoot me a question! My magic's all geared up and ready to go."):
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant", avatar="🧙"):
        message_placeholder = st.empty()
        full_response = ""
        input_dict = {
            "input": prompt,
        }
        input_dict.update(MEMORY.load_memory_variables({"query": prompt}))
        st_callback = StreamlitCallbackHandler(st.container())
        with tracing_v2_enabled("langsmith-gen-ai-book") as cb:
            for chunk in agent_executor.stream(
                    input_dict,
                    config={"tags": ["Streamlit Agent"], "callbacks": [st_callback]},
            ):
                if chunk.get("output") is not None:
                    full_response += chunk["output"]
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            feedback_kwargs = {
                "feedback_type": "faces",
                "optional_text_label": "Add more details  (optional)",
                "on_submit": _submit_feedback,
            }
            run = cb.latest_run
            MEMORY.save_context(input_dict, {"output": full_response})
            feedback_index = int(
                (len(st.session_state.get("langchain_messages", [])) - 1) / 2
            )
            st.session_state[f"run_{feedback_index}"] = run.id
            streamlit_feedback(**feedback_kwargs, key=f"feedback_{feedback_index}")


