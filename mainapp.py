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
from utils.create_pdf import create_pdf
import os

import chromadb
import re
import json
import uuid 
import shutil
import requests
from keys import OPEN_AI_KEY_ACTUAL

enable_box = gr.Textbox.update(value = None, placeholder = 'Upload your OpenAI API key',interactive = True)
disable_box = gr.Textbox.update(value = 'OpenAI API key is Set', interactive = False)

    
def enable_api_box():
        return enable_box

def add_text(history, text: str):
    if not text:
         raise gr.Error('enter text')
    history = history + [(text,'')] 
    return history



def get_response_website(history, query, url): 
        # history-> chatbot
        # query-> txt
        # file-> btn
        # if not file:
        #     raise gr.Error(message='Upload a PDF')  
        # # file.name= "demofile.pdf" # added by me
        # print(f'file name*get_response*: {file.name}') 
        # paragraphs= generate_context(url)
        chain = custom_app("output.pdf")
        
        
        result = chain({"question": query, 'chat_history':custom_app.chat_history},return_only_outputs=True)
        custom_app.chat_history += [(query, result["answer"])]
        print(f'result*: {result}')
        custom_app.N = list(result['source_documents'][0])[1][1]['page']
        for char in result['answer']:
           history[-1][-1] += char
           yield history,''
           

def render_file():
    
        doc = fitz.open('output.pdf')
        print(f'file name*render_file*: output.pdf')
        page = doc[custom_app.N]
        #Render the page as a PNG image with a resolution of 300 DPI
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
        image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
        return image
           
           

def generate_context(query:str, num_urls:int):
    
    print(f'generate_paragraphs_out_of_query: {query}')
    print(f'# urls: {num_urls}')
    if not query or not num_urls:
        raise gr.Error(message='Either query or number of urls are empty')
    response= requests.post('https://qagen.paperbot.ai/extract_all_passages', json={
                                                                                    "query": query,
                                                                                    "num_urls": int(num_urls),
                                                                                    } )
    
    if response.ok:
        # d= eval(response.content)
        paragrahs= json.loads(response.content.decode(
                                                        'utf-8'
                                                    ))['paragraphs']
        if os.path.exists('output.pdf'):
            print(f'Found output.pdf and hence deleting it')
            os.remove('output.pdf')
        create_pdf(paragrahs)
        return render_first()
    else:
        print("Couldn't get the response from the 'extract-all-passages'   🥲")
        return 
    
    
def render_first():
        
        doc = fitz.open('output.pdf')
        page = doc[0]
        #Render the page as a PNG image with a resolution of 300 DPI
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
        image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
        return image,[]

    
    
def generate_paragraphs_out_of_query(query:str, num_urls:int):
    print(f'generate_paragraphs_out_of_query: {query}')
    print(f'generate_paragraphs_out_of_query: {num_urls}')
    if not query or num_urls:
        raise gr.Error(message='Either query or number of urls are empty')
    response= requests.post('https://qagen.paperbot.ai/extract_all_passages', json={
                                                                                    "query": "best cat ear headphones",
                                                                                    "num_keywords": 50,
                                                                                    "num_paragraphs": 20,
                                                                                    "num_urls": 5,
                                                                                    "keyphrase_count": 4
                                                                                    } )
    
    if response.ok:
        # d= eval(response.content)
        paragrahs= json.loads(response.content.decode(
                                                        'utf-8'
                                                    ))['paragraphs']
        return paragrahs
    else:
        print("Couldn't get the response from the 'extract-all-passages'   🥲")
        return 
    
    
    
    
    
    
with gr.Blocks() as demo:       
    with gr.Row():
        # gr.HighlightedText("URL SECTION")
        gr.Label("URL WebPage Chatbot")
    with gr.Row():
        with gr.Column(scale=0.8):
            search_url = gr.Textbox(placeholder='Enter the Query', show_label=False, interactive=True).style(container=False)
        with gr.Column(scale=0.1):
            # num_urls = gr.Textbox(placeholder='# Urls', show_label=False, interactive=True).style(container=False)
            num_urls = gr.Number(placeholder='# Urls', show_label=False, interactive=True).style(container=False)
            
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
        # gen_show_img = gr.Image(label='Generated PDF', tool='select' ).style(height=680)
    
        
    # api_key.submit(
    #         fn=set_apikey, 
    #         inputs=[api_key], 
    #         outputs=[api_key,])
    # change_api_key.click(
    #         fn= enable_api_box,
    #         outputs=[api_key])
    
    
    gen_btn.click(
        fn=generate_context, 
        inputs=[search_url, num_urls],
        # outputs=[gen_show_img, chatbot_gen]
        )
    
    gen_submit_btn.click(
                            fn=add_text, 
                            inputs=[chatbot_gen, gen_txt], 
                            outputs=[chatbot_gen, ], 
                            queue=False
                        ).success(
                                    fn=get_response_website,
                                    inputs = [chatbot_gen, gen_txt, search_url],
                                    outputs = [chatbot_gen,gen_txt]
                                    )
                                    # ).success(
                                    #             fn=render_file,
                                    #             # fn=render_first,
                                    #             outputs=[gen_show_img])
    
demo.queue()
demo.launch()               




