import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import tempfile

# Streamlit page config
st.set_page_config(page_title="PDF Chatbot", layout="centered")
st.markdown(
    "<h1 style='text-align: center;'>📘 Ask Questions About Your PDF</h1>",
    unsafe_allow_html=True
)

# File uploader
st.markdown("#### 📤 Upload your PDF file:")
pdf_file = st.file_uploader("", type=["pdf"], label_visibility="collapsed")

if pdf_file:
    st.success(f"Uploaded: {pdf_file.name}", icon="✅")

    # Save uploaded PDF to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_path = tmp_file.name

    with st.spinner("🔍 Processing PDF..."):
        # Load and split
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        texts = splitter.split_documents(documents)

        # Embeddings + FAISS DB
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.from_documents(texts, embeddings)

        # Load QA model
        model_name = "deepset/roberta-large-squad2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

    # Question input
    st.markdown("---")
    st.markdown("#### 💬 Ask a question about the uploaded PDF:")
    question = st.text_input("Type your question here...")

    if question:
        with st.spinner("📚 Searching for the answer..."):
            docs = db.similarity_search(question, k=3)
            context = " ".join([doc.page_content for doc in docs])

            result = qa_pipeline({
                "question": question,
                "context": context
            })

        # Display answer
        st.markdown("### ✅ Answer:")
        st.write(result["answer"])

