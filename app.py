import asyncio
import logging
from dotenv import load_dotenv
from pathlib import Path
from ingest_data import download_data_and_create_embedding

from langchain_community.vectorstores import FAISS
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from ingest_data import underlying_embeddings, openai_api_key

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser


import chainlit as cl

# load env variables
load_dotenv()

# Specify the path to the file you want to check
file_path = Path('./faiss_index/index.faiss')

# Check if the file exists
if file_path.exists():
    print("Embeddings already done, use the saved index")
    # Combine the retrieved data with the output of the LLM
    vector_store = FAISS.load_local(
        "faiss_index", underlying_embeddings, allow_dangerous_deserialization=True
    )
else:
    vector_store = download_data_and_create_embedding()


# create a prompt template to send to our LLM that will incorporate the documents from our retriever with the
# question we ask the chat model
prompt_template = ChatPromptTemplate.from_template(
    "Answer the {question} based on the following {context}."
)

# create a retriever for our documents
retriever = vector_store.as_retriever()

# create a chat model / LLM
chat_model = ChatOpenAI(
    model="gpt-4o-2024-05-13", temperature=0, api_key=openai_api_key
)

# create a parser to parse the output of our LLM
parser = StrOutputParser()

# 💻 Create the sequence (recipe)
runnable_chain = (
    # TODO: How do we chain the output of our retriever, prompt, model and model output parser so that we can get a good answer to our query?
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt_template
    | chat_model
    | StrOutputParser()
)


# Asynchronous execution (e.g., for a better a chatbot user experience)
async def call_chain_async(question):
    output_chunks = await runnable_chain.ainvoke(question)
    return output_chunks


# output_stream =  asyncio.run(call_chain_async("What are some good sci-fi movies from the 1980s?"))
# print("".join(output_stream))

@cl.on_chat_start
async def on_chat_start():
    model = ChatOpenAI(streaming=True)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You're a very knowledgeable historian who provides accurate and eloquent answers to historical questions.",
            ),
            ("human", "{question}"),
        ]
    )
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)

# @cl.on_message
# async def on_message(message: cl.Message):
#     runnable = cl.user_session.get("runnable")  # type: Runnable

#     msg = cl.Message(content="")

#     async for chunk in runnable.astream(
#         {"question": message.content},
#         config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
#     ):
#         await msg.stream_token(chunk)

#     await msg.send() 

@cl.on_message
async def main(question):
    response = await call_chain_async(question.content)
    await cl.Message(content=response).send()
