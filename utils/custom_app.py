from typing import Any
import gradio as gr
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import PyPDFLoader

import fitz
from PIL import Image
from utils.cleaner import extract_paragraphs

import chromadb
import re
import uuid 
import shutil
import requests
from keys import OPEN_AI_KEY_ACTUAL

enable_box = gr.Textbox.update(value = None, placeholder = 'Upload your OpenAI API key',interactive = True)
disable_box = gr.Textbox.update(value = 'OpenAI API key is Set', interactive = False)

def set_apikey(api_key: str):
        # app.OPENAI_API_KEY = api_key  
        custom_app.OPENAI_API_KEY = OPEN_AI_KEY_ACTUAL
              
        return disable_box
    
def enable_api_box():
        return enable_box





class my_app_custom_v2:
    def __init__(self, OPENAI_API_KEY: str = None ) -> None:
        # self.OPENAI_API_KEY: str = OPENAI_API_KEY
        self.OPENAI_API_KEY:str= OPEN_AI_KEY_ACTUAL
        self.chain = None
        self.chat_history: list = []
        self.N: int = 0 # page 0 of the pdf
        self.count: int = 0

    def __call__(self, filename: str) -> Any:

        if self.count==0:
            self.chain = self.build_chain(filename)
            self.count+=1
        return self.chain
    
    def chroma_client(self):
        #create a chroma client
        client = chromadb.Client()
        #create a collection
        collection = client.get_or_create_collection(name="my-collection")
        return client
    
    def process_file(self,file: str):
        # file.name= "demofile.pdf" # added by me
        loader = PyPDFLoader(file)
        print(f'file name*process_file*: {file}')
        documents = loader.load()  
        # pattern = r"/([^/]+)$"
        # match = re.search(pattern, file)
        # file_name = match.group(1)
        # print(f"captured_file_name*: {file_name}")
        # print(f"documents[:5]: {documents[:5]}")
        return documents, file
    
    def build_chain(self, file: str):
        documents, file_name = self.process_file(file)
        print(f'file name*build_chain*: {file_name}')
        #Load embeddings model
        embeddings = OpenAIEmbeddings(openai_api_key=self.OPENAI_API_KEY) 
        # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # Equivalent to SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        # embeddings= SentenceTransformerEmbeddings(model_name="thenlper/gte-base")
        pdfsearch = Chroma.from_documents(documents, embeddings, collection_name= file_name,)
        chain = ConversationalRetrievalChain.from_llm(
                ChatOpenAI(temperature=0.0, openai_api_key=self.OPENAI_API_KEY), 
                retriever=pdfsearch.as_retriever(search_type="similarity_score_threshold", search_kwargs={'score_threshold': 0.8}), # single score with maximum similarity
                # retriever=pdfsearch.as_retriever(search_kwargs={"k": 1}), # single score with maximum similarity
                return_source_documents=True,)
        return chain
    
    def build_chain_website(self, paragraph: list):
        # documents, file_name = self.process_file(file)
        # print(f'file name*build_chain*: {file_name}')
        #Load embeddings model
        embeddings = OpenAIEmbeddings(openai_api_key=self.OPENAI_API_KEY) 
        # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # Equivalent to SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        # embeddings= SentenceTransformerEmbeddings(model_name="thenlper/gte-base")
        
        content_search = Chroma.from_texts(paragraph, embeddings,  collection_name= "webcontent")
        
        # pdfsearch = Chroma.from_documents(documents, embeddings, collection_name= file_name,)
        chain = ConversationalRetrievalChain.from_llm(
                ChatOpenAI(temperature=0.0, openai_api_key=self.OPENAI_API_KEY), 
                # retriever=content_search.as_retriever(search_kwargs={"k": 1}), # single score with maximum similarity
                retriever=content_search.as_retriever(search_kwargs={"k": 5}), # single score with maximum similarity
                return_source_documents=True,)
        return chain
    
    


class my_app_custom:
    def __init__(self, OPENAI_API_KEY: str = None ) -> None:
        # self.OPENAI_API_KEY: str = OPENAI_API_KEY
        self.OPENAI_API_KEY:str= OPEN_AI_KEY_ACTUAL
        self.chain = None
        self.chat_history: list = []
        self.N: int = 0 # page 0 of the pdf
        self.count: int = 0

    def __call__(self, paragraphs: list) -> Any:

        if self.count==0:
            self.chain = self.build_chain_website(paragraphs)
            self.count+=1
        return self.chain
    
    def chroma_client(self):
        #create a chroma client
        client = chromadb.Client()
        #create a collecyion
        collection = client.get_or_create_collection(name="my-collection")
        return client
    
    def process_file(self,file: str):
        # file.name= "demofile.pdf" # added by me
        loader = PyPDFLoader(file.name)
        print(f'file name*process_file*: {file.name}')
        documents = loader.load()  
        pattern = r"/([^/]+)$"
        match = re.search(pattern, file.name)
        file_name = match.group(1)
        print(f"captured_file_name*: {file_name}")
        print(f"documents[:5]: {documents[:5]}")
        return documents, file_name
    
    def build_chain(self, file: str):
        documents, file_name = self.process_file(file)
        print(f'file name*build_chain*: {file_name}')
        #Load embeddings model
        embeddings = OpenAIEmbeddings(openai_api_key=self.OPENAI_API_KEY) 
        # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # Equivalent to SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        # embeddings= SentenceTransformerEmbeddings(model_name="thenlper/gte-base")
        pdfsearch = Chroma.from_documents(documents, embeddings, collection_name= file_name,)
        chain = ConversationalRetrievalChain.from_llm(
                ChatOpenAI(temperature=0.0, openai_api_key=self.OPENAI_API_KEY), 
                retriever=pdfsearch.as_retriever(search_kwargs={"k": 1}), # single score with maximum similarity
                return_source_documents=True,)
        return chain
    
    def build_chain_website(self, paragraph: list):
        # documents, file_name = self.process_file(file)
        # print(f'file name*build_chain*: {file_name}')
        #Load embeddings model
        embeddings = OpenAIEmbeddings(openai_api_key=self.OPENAI_API_KEY) 
        # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # Equivalent to SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        # embeddings= SentenceTransformerEmbeddings(model_name="thenlper/gte-base")
        
        content_search = Chroma.from_texts(paragraph, embeddings,  collection_name= "webcontent")
        
        # pdfsearch = Chroma.from_documents(documents, embeddings, collection_name= file_name,)
        chain = ConversationalRetrievalChain.from_llm(
                ChatOpenAI(temperature=0.0, openai_api_key=self.OPENAI_API_KEY), 
                retriever=content_search.as_retriever(search_kwargs={"k": 1}), # single score with maximum similarity
                return_source_documents=True,)
        return chain
    
    
    
custom_app= my_app_custom_v2()