from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryMemory, ChatMessageHistory
from langchain.llms import OpenAI


llm = OpenAI(temperature=0)

conversation_with_summary = ConversationChain(
    llm=llm,
    memory=ConversationSummaryMemory(llm=llm, human_prefix="Aarushi", ai_prefix="Hermione"),
    verbose=True
)
conversation_with_summary.predict(input="How are you Hermione?")
conversation_with_summary.predict(input="What is the third planet from the sun?")
conversation_with_summary.predict(input="second?")
conversation_with_summary.predict(input="fifth?")
conversation_with_summary.predict(input="")