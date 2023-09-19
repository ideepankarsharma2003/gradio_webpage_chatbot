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
from utils.custom_app import custom_app

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
        app.OPENAI_API_KEY = OPEN_AI_KEY_ACTUAL
              
        return disable_box
    
def enable_api_box():
        return enable_box

def add_text(history, text: str):
    if not text:
         raise gr.Error('enter text')
    history = history + [(text,'')] 
    return history

class my_app:
    def __init__(self, OPENAI_API_KEY: str = None ) -> None:
        # self.OPENAI_API_KEY: str = OPENAI_API_KEY
        self.OPENAI_API_KEY:str= OPEN_AI_KEY_ACTUAL
        self.chain = None
        self.chat_history: list = []
        self.N: int = 0 # page 0 of the pdf
        self.count: int = 0

    def __call__(self, file: str) -> Any:
        # file.name= "demofile.pdf" # added by me
        new_path= file.name.replace(' ', '')
        if new_path!= file.name:
            shutil.copyfile(file.name, new_path)
        file.name= new_path

        print(f"__call___: {file.name}")
        if self.count==0:
            self.chain = self.build_chain(file)
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
        
        content_search = Chroma.from_texts(paragraph, embeddings)
        
        # pdfsearch = Chroma.from_documents(documents, embeddings, collection_name= file_name,)
        chain = ConversationalRetrievalChain.from_llm(
                ChatOpenAI(temperature=0.0, openai_api_key=self.OPENAI_API_KEY), 
                retriever=content_search.as_retriever(search_kwargs={"k": 1}), # single score with maximum similarity
                return_source_documents=True,)
        return chain
    

def get_response(history, query, file): 
        # history-> chatbot
        # query-> txt
        # file-> btn
        if not file:
            raise gr.Error(message='Upload a PDF')  
        # file.name= "demofile.pdf" # added by me
        print(f'file name*get_response*: {file.name}') 
        chain = app(file)
        result = chain({"question": query, 'chat_history':app.chat_history},return_only_outputs=True)
        app.chat_history += [(query, result["answer"])]
        print(f'result*: {result}')
        app.N = list(result['source_documents'][0])[1][1]['page']
        for char in result['answer']:
           history[-1][-1] += char
           yield history,''

def get_response_website(history, query, url): 
        # history-> chatbot
        # query-> txt
        # file-> btn
        # if not file:
        #     raise gr.Error(message='Upload a PDF')  
        # # file.name= "demofile.pdf" # added by me
        # print(f'file name*get_response*: {file.name}') 
        paragraphs= generate_context(url)
        chain = custom_app(paragraphs)
        
        
        result = chain({"question": query, 'chat_history':app.chat_history},return_only_outputs=True)
        app.chat_history += [(query, result["answer"])]
        print(f'result*: {result}')
        # app.N = list(result['source_documents'][0])[1][1]['page']
        for char in result['answer']:
           history[-1][-1] += char
           yield history,''

def render_file(file):
    
        doc = fitz.open(file.name)
        print(f'file name*render_file*: {file.name}')
        page = doc[app.N]
        #Render the page as a PNG image with a resolution of 300 DPI
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
        image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
        return image

def render_first(file):
        print(f'file name*render_first*: {file.name}')
        new_path= file.name.replace(' ', '')
        if new_path!= file.name:
            shutil.copyfile(file.name, new_path)
        
        shutil.copyfile(file.name, 'demofile.pdf')
        file.name= new_path
        print(f'file name*render_first*: {file.name}')
        
        doc = fitz.open(file.name)
        page = doc[0]
        #Render the page as a PNG image with a resolution of 300 DPI
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
        image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
        return image,[]


def generate_context(url):
    if not url:
        raise gr.Error(message='Enter a URL')
    else:
        print(url)
        paragraphs= extract_paragraphs(url)
        print(f"extract_paragraphs: {paragraphs}")
        return paragraphs
    


app = my_app()
with gr.Blocks() as demo:
    with gr.Row():
        # gr.HighlightedText("URL SECTION")
        gr.Label("PDF Chatbot")
    with gr.Column():
        # with gr.Row():
        #     with gr.Column(scale=0.8):
        #         api_key = gr.Textbox(placeholder='Enter OpenAI API key', show_label=False, interactive=True).style(container=False)
        #     with gr.Column(scale=0.2):
        #         change_api_key = gr.Button('Change Key')
        
        with gr.Row():           
            chatbot = gr.Chatbot(value=[], elem_id='chatbot').style(height=650)
            show_img = gr.Image(label='Upload PDF', tool='select' ).style(height=680)
    with gr.Row():
        with gr.Column(scale=0.60):
            txt = gr.Textbox(
                        show_label=False,
                        placeholder="Enter text and press enter",
                    ).style(container=False)
        with gr.Column(scale=0.20):
            submit_btn = gr.Button('submit')
        with gr.Column(scale=0.20):
            btn = gr.UploadButton("📁 upload a PDF",  file_types=[".pdf"]).style()
            print(f'btn: {btn}')
            # print(f'btn.hash_file(): {btn.hash_file()}')
            # print(f'btn.file_bytes_to_file: {btn.file_bytes_to_file(btn.hash_file())}')
            # shutil.copyfile(btn.hash_file(), 'demofile.pdf')
           
    with gr.Row():
        # gr.HighlightedText("URL SECTION")
        gr.Label("URL WebPage Chatbot")
    with gr.Row():
        with gr.Column(scale=0.8):
            search_url = gr.Textbox(placeholder='Enter the Query', show_label=False, interactive=True).style(container=False)
        with gr.Column(scale=0.1):
            search_url = gr.Textbox(placeholder='# Urls', show_label=False, interactive=True).style(container=False)
        with gr.Column(scale=0.1):
            gen_btn = gr.Button('Generate context')
    with gr.Row():
        with gr.Column(scale=0.80):
            gen_txt = gr.Textbox(
                        show_label=False,
                        placeholder="Enter your text here and press submit",
                    ).style(container=False)
        with gr.Column(scale=0.20):
            gen_submit_btn = gr.Button('submit')
    with gr.Row():           
        chatbot_gen = gr.Chatbot(value=[], elem_id='chatbot_gen').style(height=850)
    
        
    # api_key.submit(
    #         fn=set_apikey, 
    #         inputs=[api_key], 
    #         outputs=[api_key,])
    # change_api_key.click(
    #         fn= enable_api_box,
    #         outputs=[api_key])
    
    
    gen_btn.click(
        fn=generate_context, 
        inputs=[search_url]
        
    )
    
    gen_submit_btn.click(
                            fn=add_text, 
                            inputs=[chatbot_gen, gen_txt], 
                            outputs=[chatbot_gen, ], 
                            queue=False
                        ).success(
                                    fn=get_response_website,
                                    inputs = [chatbot, txt, search_url],
                                    outputs = [chatbot,txt]
                                    )
    
    
    btn.upload(
            fn=render_first, 
            inputs=[btn], 
            outputs=[show_img,chatbot],)
    
    submit_btn.click(
                        fn=add_text, 
                        inputs=[chatbot,txt], 
                        outputs=[chatbot, ], 
                        queue=False
                        
                        ).success(
                                    fn=get_response,
                                    inputs = [chatbot, txt, btn],
                                    outputs = [chatbot,txt]
                                    ).success(
                                                fn=render_file,
                                                inputs = [btn], 
                                                outputs=[show_img]
                                                )

    
demo.queue()
demo.launch()  