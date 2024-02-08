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

def start():
    WEAVIATE_URL = "https://gen-ai-apps-book-2aegqj1u.weaviate.network"
    LOCAL_ZIPFILE = "gen_ai_co_slack.zip"
    loader = SlackDirectoryLoader(LOCAL_ZIPFILE)
    docs = loader.load()


    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=40)
    documents = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    db = Weaviate.from_documents(documents, embeddings, weaviate_url=WEAVIATE_URL, by_text=False)

    llm = OpenAI(temperature=0)
    memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)
    ret = db.as_retriever()

    qa = ConversationalRetrievalChain.from_llm(llm, retriever=ret, memory=memory)
    qa("What is the work from anywhere policy?")
    qa("are there any in office days required?")
    qa("any coworking?")

    print(qa(""))


start()