import os
import weaviate
from langchain.document_loaders import SlackDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Weaviate
from langchain.memory import ConversationSummaryMemory
import weaviate
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain

def qa():
    OPENAI_API_KEY = ""
    WEAVIATE_URL = "https://gen-ai-apps-book-2aegqj1u.weaviate.network"

    client = weaviate.Client(
        url=WEAVIATE_URL,
        additional_headers={
            'X-OpenAI-Api-Key': os.environ["OPENAI_API_KEY"]
        }
    )

    vectorstore = Weaviate(client)
    ret = vectorstore.as_retriever()

    llm = OpenAI(temperature=0)
    memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)

    qa = ConversationalRetrievalChain.from_llm(llm, retriever=ret, memory=memory)
    qa("What is the work from anywhere policy?")
    qa("are there any in office days required?")
    qa("any coworking?")

    print(qa(""))

qa()