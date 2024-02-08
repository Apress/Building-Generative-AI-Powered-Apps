from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chains import ConversationChain

llm = OpenAI(temperature=0)
conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory(human_prefix="Aarushi", ai_prefix="Hermione")
)

conversation.predict(input="What is the largest city in the UK by population?")
conversation.predict(input="And second?")
conversation.predict(input="What about in Germany?")
# just to run one more time
conversation.predict(input="")
