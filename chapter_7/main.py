import os
from dotenv import find_dotenv, load_dotenv

from langchain.chat_models import ChatOpenAI
from langsmith import Client
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType


###########################

# set up environment

###########################
load_dotenv(find_dotenv())
os.environ["LANGCHAIN_API_KEY"] = str(os.getenv("LANGCHAIN_API_KEY"))
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "langsmith-presentation"
os.environ["SERPAPI_API_KEY"] = str(os.getenv("SERPAPI_API_KEY"))

llm = ChatOpenAI()

tools = load_tools(["serpapi", "llm-math"], llm=llm)

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

agent.run("What is the square root of the hight in metres of what is commonly considered as the highest mountain on earth?")




