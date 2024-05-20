import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from datasets import load_dataset
from langchain.document_loaders.csv_loader import (
    CSVLoader,
)  # import to load our imdb.csv file
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.vectorstores import FAISS

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
underlying_embeddings = OpenAIEmbeddings(api_key=openai_api_key)

def download_data_and_create_embedding():
  # Download an IMDB datset from Hugging Face Hub, load the ShubhamChoksi/IMDB_Movies dataset
  dataset = load_dataset("ShubhamChoksi/IMDB_Movies")
  print(dataset)

  # store imdb.csv from ShubhamChoksi/IMDB_Movies
  dataset_dict = dataset
  dataset_dict["train"].to_csv("imdb.csv")

  # load the csv file exported into a document
  loader = CSVLoader("imdb.csv")  # TODO
  data = loader.load()  # TODO
  print(len(data))  # ensure we have actually loaded data into a format LangChain can recognize

  """# Chunk the loaded data to improve retrieval performance
  In a RAG system, the model needs to be able to quickly and accurately retrieve relevant information 
  from a knowledge base or other data sources to assist in generating high-quality responses. 
  However, working with large, unstructured datasets can be computationally expensive and time-consuming, 
  especially during the retrieval process.

  By splitting the data into these smaller, overlapping chunks, the RAG system can more efficiently search 
  and retrieve the most relevant information to include in the generated response. This can lead to improved performance, 
  as the model doesn't have to process the entire dataset at once, and can focus on the most relevant parts of the data.
  """

  # create a text splitter with 1000 character chunks and 100 character overlap?
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
  chunked_documents = text_splitter.split_documents(
      data
  )  # TODO: How do we chunk the data?
  print(len(chunked_documents))  # ensure we have actually split the data into chunks

  """# Use OpenAI embeddings to create a vector store
  The first step in creating a vector store is to create embeddings from the data that you want the RAG system to be able 
  to retrieve. This is done using an embedding model, which transforms text data into a high-dimensional vector representation. 
  Each piece of text (such as a document, paragraph, or sentence) is converted into a vector that captures its semantic meaning.
  For this exercise, we will use OpenAI's embedding model.
  """

  openai_api_key = os.getenv("OPENAI_API_KEY")
  # create our embedding model
  embedding_model = OpenAIEmbeddings(
      model="text-embedding-3-large", api_key=openai_api_key
  )  

  """# Create embedder
  We will create our embedder using the `CacheBackedEmbeddings` class. This class is designed to optimize the process of generating embeddings by 
  caching the results of expensive embedding computations. This caching mechanism prevents the need to recompute embeddings for the same text 
  multiple times, which can be computationally expensive and time-consuming.
  """

  # create a local file store to for our cached embeddings
  store = LocalFileStore(
      "./cache/"
  )  
  embedder = CacheBackedEmbeddings.from_bytes_store(
      underlying_embeddings, store, namespace=underlying_embeddings.model
  )

  # Create vector store using Facebook AI Similarity Search (FAISS)
  vector_store = FAISS.from_documents(
      documents=chunked_documents, embedding=embedder
  )  # TODO: How do we create our vector store using FAISS?
  print(vector_store.index.ntotal)


  # save our vector store locally
  vector_store.save_local("faiss_index")

  query_embedding(vector_store=vector_store)

  return vector_store

def query_embedding(vector_store) -> None:
    # Ask your RAG system a question!
    query = "What are some good sci-fi movies from the 1980s?"

    # embed our query
    embedded_query = underlying_embeddings.embed_query(query)
    similar_documents = vector_store.similarity_search_by_vector(
        embedded_query
    )  # TODO: How do we do a similarity search to find documents similar to our query?

    for page in similar_documents:
        # Print the similar documents that the similarity search returns?
        print(page.page_content)