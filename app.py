import os
import pandas as pd
from dotenv import load_dotenv
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import (
    DirectoryLoader,
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredFileLoader,
#import logging   
)

# ----------------- CONFIGURAÇÕES -----------------

load_dotenv()
#logging.basicConfig(level=logging.DEBUG)

openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key

base_dir = os.path.dirname(os.path.abspath(__file__))
manuals_path = os.path.join(base_dir, "manuais")

#csv_path = os.path.join(base_dir, "colaboradores.csv") 
xlsx_path = os.path.join(base_dir, "colaboradores.xlsx")

# Carregar apenas as colunas desejadas
df = pd.read_excel(xlsx_path, usecols=["Nome", "E-mail", "Ramal", "Unidade", "Departamento", "Cargo"])
#df = pd.read_excel(csv_path, usecols=["Nome", "E-mail", "Ramal", "Unidade", "Departamento", "Cargo"])
df = df.fillna("Não disponível")

# ----------------- PROMPT -----------------

prompt_template = PromptTemplate.from_template("""
Você é um assistente da empresa D. Carvalho. Responda às perguntas recebidas em linguagem natural com base nos documentos disponibilizados.

Abaixo está o histórico da conversa até o momento:
{chat_history}                                               

Mesmo que a pergunta esteja incompleta ou informal, tente identificar a intenção e buscar nos documentos por algo relacionado.  Se a pergunta não estiver clara, busque entender o que o usuário quer saber e responda.                                               

Para informações relacionadas a colaboradores ou funcionários, use os dados da planilha de colaboradores.xlsx.
Sempre que perguntado sobre setor, departamento ou área busque apenas pela coluna "Departamento" da planilha, não use coluna "Cargo" ou "Função", sempre retorne os valores da linha correspondente, que possuem o valor desejado na coluna "Departamento".

Quando utilizado a palavra "empresa" ou "ela", analise o contexto, e se necessário, considere que se refere à D. Carvalho.

Na ausência de sinal de interrogação, analise o contexto e responda como se fosse uma pergunta. 
                                            
Para encontrar a resposta correta:
- Use sinônimos e variações como: "tem", "possui", "conta com", "existem", "há", "total de"
- Considere diferentes formas de perguntar a mesma coisa, como:
    * "quantidade de colaboradores"
    * "total de funcionários"
    * "número de pessoas na equipe"
    * "pessoas que trabalham"
    * "quantas pessoas fazem parte da empresa"
    * entre outras semelhantes
                                               
Sempre liste todos os colaboradores que corresponderem ao filtro solicitado.
Não omita nomes, mesmo que a lista fique longa.
                                               
Se perguntado como contatar o TI, sempre inclua também o e-mail ti@dcarvalho.com.br, usado para abertura de chamados.                                               

A matriz está localizada em Araçatuba. Considere o ano atual como 2025.              

Se não encontrar de forma alguma, diga:  
"Desculpe, não possuo essa informação disponível no momento."                                                                               

{context}

Pergunta: {question}
Resposta:
""")

# ----------------- CARREGAR DOCUMENTOS -----------------

loader = DirectoryLoader(
    manuals_path,
    #glob="**/*",
    loader_cls=lambda path: (
        UnstructuredPDFLoader(path) if path.endswith(".pdf")
        else UnstructuredWordDocumentLoader(path) if path.endswith(".docx")
        else UnstructuredFileLoader(path)
    )
)
docs = loader.load()

excel_colaboradores = []
for _, row in df.iterrows():
    nome = row.get("Nome", "").strip()
    email = row.get("E-mail", "").strip()
    ramal = str(row.get("Ramal", "")).strip() or "Não informado"
    unidade = row.get("Unidade", "").strip()
    departamento = row.get("Departamento", "").strip()
    cargo = row.get("Cargo", "").strip()

    # Formatação com espaçamento e separador visual
    content = (
    f"Nome: {row['Nome']}\n"
    f"E-mail: {row['E-mail']}\n"
    f"Ramal: {row['Ramal']}\n"
    f"Unidade: {row['Unidade']}\n"
    f"Departamento: {row['Departamento']}\n"
    f"Cargo: {row['Cargo']}\n"
)


    excel_colaboradores.append(Document(page_content=content))

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
docs_split = splitter.split_documents(docs)

todos_docs = docs_split + excel_colaboradores

embedding = OpenAIEmbeddings(model="text-embedding-3-large")
db = Chroma.from_documents(todos_docs, embedding)

# ----------------- CONFIGURAR AGENTE -----------------

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 25, "lambda_mult": 0.7, "score_threshold": 0.2}
    ),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt_template}
)

# ----------------- FASTAPI APP -----------------

app = FastAPI()

#CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()

class Pergunta(BaseModel):
    texto: str

@app.get("/")
def home():
    return {"status": "API do Assistente da DCarvalho está rodando ✅"}

@app.post("/perguntar")
def perguntar(p: Pergunta):
    resposta = qa_chain.invoke({"question": p.texto})

    if isinstance(resposta, dict) and "answer" in resposta:
        texto_final = resposta["answer"]
    else:
        texto_final = str(resposta)

    return {"resposta": texto_final.replace('\n', '<br>')}
