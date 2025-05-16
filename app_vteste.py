import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import google.generativeai as genai

# Configuração inicial
load_dotenv()
st.set_page_config(page_title="Assistente IA", page_icon="🤖")

def get_valid_model():
    """Retorna o modelo Gemini disponível"""
    try:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        available_models = [m.name for m in genai.list_models()]
        
        # Ordem de prioridade dos modelos
        model_preference = [
            "models/gemini-1.5-pro-latest",  # Modelo mais recente
            "models/gemini-1.0-pro-latest",  # Versão estável
            "models/gemini-pro"              # Nome alternativo
        ]
        
        for model in model_preference:
            if model in available_models:
                return model
        
        st.error("Nenhum modelo Gemini compatível encontrado. Modelos disponíveis: " + ", ".join(available_models))
        return None
    except Exception as e:
        st.error(f"Erro ao acessar API: {str(e)}")
        return None

@st.cache_resource
def load_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

def process_files(uploaded_files):
    """Processa arquivos PDF e TXT"""
    texts = []
    for uploaded_file in uploaded_files:
        try:
            if uploaded_file.type == "application/pdf":
                reader = PdfReader(uploaded_file)
                texts.append("\n".join([page.extract_text() or "" for page in reader.pages]))
            elif uploaded_file.type == "text/plain":
                texts.append(uploaded_file.read().decode("utf-8"))
        except Exception as e:
            st.warning(f"Erro ao processar {uploaded_file.name}: {str(e)}")
    return texts

def is_about_documents(question):
    """Define quando consultar os documentos"""
    greetings = ['bom dia', 'boa tarde', 'boa noite', 'olá', 'oi', 'saudações']
    return not any(greeting in question.lower() for greeting in greetings)

def main():
    st.title("📚 DocGenius - Seu Assistente Inteligente para Documentos")
    
    # Verifica modelo disponível
    model_name = get_valid_model()
    if not model_name:
        return
    
    # Configura o modelo selecionado
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    llm = genai.GenerativeModel(model_name)
    st.sidebar.success(f"Modelo: {model_name.split('/')[-1]}")

    # Upload de arquivos
    uploaded_files = st.file_uploader(
        "Carregue seus documentos (PDF/TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    # Processamento dos documentos
    vectorstore = None
    if uploaded_files:
        with st.spinner("Processando documentos..."):
            texts = process_files(uploaded_files)
            if texts:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                chunks = []
                for text in texts:
                    chunks.extend(text_splitter.split_text(text))
                
                documents = [Document(page_content=chunk) for chunk in chunks]
                embeddings = load_embeddings()
                vectorstore = FAISS.from_documents(documents, embeddings)
                st.success(f"{len(documents)} trechos processados!")

    # Chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Como posso ajudar?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                try:
                    if vectorstore and is_about_documents(prompt):
                        # Consulta aos documentos
                        docs = vectorstore.similarity_search(prompt, k=2)
                        context = "\n\n".join([f"📄 Trecho {i+1}:\n{doc.page_content}" 
                                             for i, doc in enumerate(docs)])
                        
                        response = llm.generate_content(
                            f"Baseado nestes documentos:\n{context}\n\nPergunta: {prompt}\n\nResposta:"
                        )
                        answer = response.text
                        
                        with st.expander("📌 Fontes utilizadas"):
                            for doc in docs:
                                st.text(doc.page_content[:500] + "...")
                    else:
                        # Resposta geral
                        response = llm.generate_content(prompt)
                        answer = response.text
                    
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                
                except Exception as e:
                    st.error(f"Erro na geração: {str(e)}")

if __name__ == "__main__":
    main()