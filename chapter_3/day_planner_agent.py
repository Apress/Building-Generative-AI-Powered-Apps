import streamlit as st
from langchain import LLMChain, OpenAI
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent, ReActTextWorldAgent, initialize_agent,ConversationalAgent
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.vectorstores.weaviate import Weaviate
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentType
import weaviate
import os
from langchain.utilities import GoogleSerperAPIWrapper, OpenWeatherMapAPIWrapper
from langchain.chat_models import ChatOpenAI
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

st.title("Day Planner")

SERPER_API_KEY = ""

AI = OpenAI(temperature=0.7)

# search tool
search = GoogleSerperAPIWrapper(serper_api_key=SERPER_API_KEY)
# weather tool
weather = OpenWeatherMapAPIWrapper()


tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Useful for when you need to get current, up to date answers."
    ),
    Tool(
        name="Weather",
        func=weather.run,
        description="Useful for when you need to get the current weather in a location."
    )
]

msgs = StreamlitChatMessageHistory()

prefix = """You are a friendly, modern day planner. You help users find activities in a given city based on their preferences and the weather.
            You have access to to two tools:"""
suffix = """Begin!"
Chat History:
{chat_history}
Latest Question: {input}
{agent_scratchpad}"""

prompt = ConversationalAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"],
)
memory = ConversationBufferMemory(meessages=msgs, memory_key="chat_history", return_messages=True)
if "memory" not in st.session_state:
    st.session_state.memory = memory

llm_chain = LLMChain(
    llm=ChatOpenAI(
        temperature=0.8, model_name="gpt-4"
    ),
    prompt=prompt,
)
# agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)

agent = ConversationalAgent(llm_chain=llm_chain, tools=tools, verbose=True, memory=memory, max_iterations=3)


agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=memory
)

# Allow the user to enter a query and generate a response
query = st.text_input(
    "What are you in the mood for?",
    placeholder="I can help!",
)

if query:
    with st.spinner(
            "Thinking...."
    ):
        res = agent_chain.run(query)
        st.info(res, icon="🤖")


# Allow the user to view the conversation history and other information stored in the agent's memory
with st.expander("My thinking"):
    st.session_state.memory.return_messages

