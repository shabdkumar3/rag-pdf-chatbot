import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
import tempfile
import os

st.title("Chat with your PDF 📄")

api_key = st.text_input("Enter Groq API Key", type="password")
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file and api_key:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded_file.read())
        temp_path = f.name

    with st.spinner("Processing PDF..."):
        loader = PyPDFLoader(temp_path)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(pages)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(chunks, embedding=embeddings)

    st.success("PDF is ready! Ask anything.")

    question = st.text_input("Enter your question")

    if question:
        with st.spinner("Thinking..."):
            llm = ChatGroq(api_key=api_key, model_name="llama-3.1-8b-instant")
            docs = vectorstore.similarity_search(question, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])
            prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
            response = llm.invoke(prompt)
            st.write(response.content)