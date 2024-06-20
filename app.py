import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import pandas as pd
import re
import os
import shutil
import time
import openai
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

chave_openai = ''

def iniciarProjeto(chave_openai_par):
    chave_openai = chave_openai_par
    openai.api_key = chave_openai_par
    if chave_openai != '':
        embedding = OpenAIEmbeddings(openai_api_key=openai.api_key)

        path_projeto = os.getcwd()
        persist_directory = path_projeto + '/Trabalho_NLP/'

        vectordb = Chroma(
            embedding_function=embedding,
            persist_directory=persist_directory
        )
        llm_name = "gpt-3.5-turbo"
        llm = ChatOpenAI(model_name=llm_name, temperature=1, openai_api_key=openai.api_key)

        template = """Utilize os seguintes elementos de contexto para responder à pergunta no final.
        Se você não souber a resposta, simplesmente diga que não sabe, não tente inventar uma resposta.
        Mantenha a resposta o mais concisa possível, saiba que o seu contexto corresponde a posts e comentários feitos no Reddit e
        tente manter as respostas sempre dentro do contexto que será passado a seguir respondendo somente na língua portuguesa. Não esqueça de formatar as respostas, retirando informações desnecessárias.
        {context}
        Pergunta: {question}
        Helpful Answer:"""
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

        return RetrievalQA.from_chain_type(
            llm,
            retriever=vectordb.as_retriever(), 
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )

st.title('Conversando com o Reddit sobre Jogos')

with st.sidebar.form("chave-api", clear_on_submit=True):
    texto_a_ser_lido = st.text_input(label = 'Chave API:')
    submitted = st.form_submit_button("Enviar")

    if submitted is not None:
        chave_openai = texto_a_ser_lido

if chave_openai == '':
    pass
else:
    qa_chain = iniciarProjeto(chave_openai)
    texto_a_ser_analisado = ''

    with st.form("pergunta", clear_on_submit=True):
        texto_a_ser_lido = st.text_area(label = 'Texto a ser analisado:', 
                                        height = 100)
        submitted = st.form_submit_button("Enviar")

        if submitted and texto_a_ser_analisado is not None:
            texto_a_ser_analisado = texto_a_ser_lido
            result = qa_chain({"query": texto_a_ser_analisado})
            st.write(result)
