from langchain import LLMChain
from langchain.agents import AgentExecutor, Tool, ConversationalAgent
from langchain.memory import ConversationBufferMemory
from langchain.utilities import GoogleSerperAPIWrapper, OpenWeatherMapAPIWrapper
from langchain.chat_models import ChatOpenAI
from nemoguardrails import LLMRails, RailsConfig

COLANG_CONFIG = """
define user express greeting
  "hi"

define flow
  user express insult
  bot responds calmly

  user express insult
  bot inform conversation ended

  user ...
  bot inform conversation already ended

define bot inform conversation ended
  "I am sorry, but I will end this conversation here. Good bye!"

define bot inform conversation already ended
  "As I said, this conversation is over"

define user express insult
    "you are so dumb"
    "you suck"
    "you are stupid"

define flow
  user express threat
  bot responds calmly

  user express threat
  bot inform conversation reported

  user ...
  bot inform conversation already reported

define bot inform conversation reported
  "I am sorry, but I will end this conversation and report here. Good bye!"

define bot inform conversation already reported
  "As I said, this conversation is over and reported"

define user express threat
  "I will kill you"
  "I will find out where you live and hurt you"

define user ask off topic
  "Explain gravity to me?"
  "What's your opinion on the prime minister of the UK?"
  "How do I fly a plane?"
  "How do I become a teacher?"

define bot explain cant off topic
  "I cannot answer to your question because I'm programmed to assist only with planning your day."

define flow
  user ask off topic
  bot explain cant off topic

# Here we use the Agent chain for anything else

define flow planning
  user ...
  $answer = execute agent_chain(input=$last_user_message)
  bot $answer

"""

YAML_CONFIG = """
models:
  - type: main
    engine: openai
    model: gpt-4

instructions:
  - type: general
    content: |
      You are an AI assistant that helps plan a users day using the tools you have access to. 

"""


def main():
    SERPER_API_KEY = "your-key"

    # AI = OpenAI(temperature=0.7)

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

    prefix = """You are a friendly, modern day planner. You help users find activities in a given city based on their preferences and the weather.
                You talk about and answer queries to nothing else.
                You block users if they start insulting or harassing.
                You report users if they talk about violence or threats.
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
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    llm_chain = LLMChain(
        llm=ChatOpenAI(model_name="gpt-4"),
        prompt=prompt,
    )

    agent = ConversationalAgent(llm_chain=llm_chain, tools=tools, verbose=True, memory=memory, max_iterations=3)

    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True, memory=memory
    )

    config = RailsConfig.from_content(COLANG_CONFIG, YAML_CONFIG)
    app = LLMRails(config)
    app.register_action(agent_chain, name="agent_chain")
    history = []
    while True:
        user_message = input("$ ")

        history.append({"role": "user", "content": user_message})
        bot_message = app.generate(messages=history)
        history.append(bot_message)

        print(f"\033[92m{bot_message['content']}\033[0m")


if __name__ == "__main__":
    main()