import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
import tempfile

st.title("Chat with your PDFs 📄")

api_key = st.secrets["GROQ_API_KEY"]
uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if uploaded_files and api_key:
    if st.button("Process PDFs"):
        with st.spinner("Processing PDFs..."):
            all_chunks = []
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                    f.write(uploaded_file.read())
                    temp_path = f.name
                loader = PyPDFLoader(temp_path)
                pages = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = splitter.split_documents(pages)
                all_chunks.extend(chunks)

            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            st.session_state.vectorstore = Chroma.from_documents(all_chunks, embedding=embeddings)
            st.success(f"{len(uploaded_files)} PDF(s) ready! Ask anything.")

if st.session_state.vectorstore and api_key:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input("Ask a question...")

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                llm = ChatGroq(api_key=api_key, model_name="llama-3.1-8b-instant")
                docs = st.session_state.vectorstore.similarity_search(question, k=3)
                context = "\n\n".join([doc.page_content for doc in docs])

                history_text = ""
                for msg in st.session_state.chat_history[-4:]:
                    history_text += f"{msg['role']}: {msg['content']}\n"

                prompt = f"Chat History:\n{history_text}\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
                response = llm.invoke(prompt)
                st.write(response.content)
                st.session_state.chat_history.append({"role": "assistant", "content": response.content})