import uvicorn
import sys
import os
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.responses import Response, JSONResponse
from starlette.responses import RedirectResponse
from typing import Any
# import gradio as gr
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import PyPDFLoader

import fitz
from PIL import Image
from utils.cleaner import extract_paragraphs
from utils.custom_app import my_app_custom_v2
from utils.create_pdf import create_pdf
import os

import chromadb
import re
import json
import uuid 
import shutil
import requests
from keys import *


from pydantic import BaseModel



HISTORY= [] # history variable
custom_app= my_app_custom_v2()

class Keyword(BaseModel):
    query:str
    # num_keywords=50
    # num_paragraphs=20
    num_urls= 5
    # keyphrase_count= 4

# helper functions

def add_text(history, text: str):
    print({
        'text': text,
        'history': history
    }, end="\n\n")
    history = history + [(text,'')] 
    return history



def get_response_website(history, query): 
        # history-> chatbot
        # query-> txt
        # file-> btn
        # if not file:
        #     raise gr.Error(message='Upload a PDF')  
        # # file.name= "demofile.pdf" # added by me
        # print(f'file name*get_response*: {file.name}') 
        # paragraphs= generate_context(url)
        global custom_app
        
        chain = custom_app("output.pdf")
        
        
        result = chain({"question": query, 'chat_history':custom_app.chat_history},return_only_outputs=True)
        custom_app.chat_history += [(query, result["answer"])]
        print(f'result*: {result}')
        # custom_app.N = list(result['source_documents'][0])[1][1]['page']
        # for char in result['answer']:
        #    history[-1][-1] += char
        #    yield history,''
        return result




app= FastAPI()
print("initializing app")

@app.get('/', tags=['authentication'])
async def index():
    return RedirectResponse(url='/docs')
    # return "Hello world!"




@app.post('/generate_pdf_content')
async def generate_pdf_content(keyword:Keyword):
    if os.path.exists('output.pdf'):
        print(f'Found output.pdf and hence deleting it')
        os.remove('output.pdf')
    global HISTORY
    HISTORY= []
    global custom_app
    custom_app= my_app_custom_v2()
    
    query= keyword.query
    num_urls= keyword.num_urls
    print(f'generate_paragraphs_out_of_query: {query}')
    print(f'# urls: {num_urls}')
    
    try: 
        response= requests.post(qa_gen_api_extract_all_passages, json={
                                                                                    "query": query,
                                                                                    "num_urls": int(num_urls),
                                                                                    } )
    
        if response.ok:
            # d= eval(response.content)
            paragrahs= json.loads(response.content.decode(
                                                            'utf-8'
                                                        ))['paragraphs']
            create_pdf(paragrahs)
            is_content= f"YES 😁, buffer content: {query}" 
            
            return is_content
            
            
    except Exception as e:
        print("Couldn't get the response from the 'extract-all-passages'   🥲")
        return Response(f'Error occured: {e}')
    
 


@app.get('/generate_question')
async def generate_question(text):
    
    try: 
              
        global HISTORY
        HISTORY= add_text(history=HISTORY, text=text)
        result= get_response_website(history=HISTORY, query=text)
        
        return result
    except Exception as e:
        return Response(f'Error occured: {e}')


   
    
    

if __name__=='__main__':
    uvicorn.run(app, host='127.0.0.1', port=8056)
    # uvicorn.run(app, host='127.0.0.1', port=8020)

