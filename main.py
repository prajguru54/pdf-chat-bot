import warnings
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM  # Ensure langchain-ollama is installed
from langchain.embeddings import HuggingFaceEmbeddings  # Correct import path

# Suppress specific FutureWarning from transformers
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="`clean_up_tokenization_spaces` was not set.*",
)

# Optional: Load environment variables if needed
# from dotenv import load_dotenv
# load_dotenv()

# Function to initialize the Ollama LLM
@st.cache_resource
def load_llama_model():
    """
    Initialize the Ollama LLM using langchain-ollama's OllamaLLM.
    """
    try:
        # Initialize the Ollama LLM with the specific model
        llm = OllamaLLM(model="llama3.1")
        return llm
    except Exception as e:
        st.error(f"Error initializing Ollama LLM: {e}")
        return None

# Function to extract text from uploaded PDF
def extract_text_from_pdf(pdf_file):
    """
    Extract text from an uploaded PDF file.
    """
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return ""

# Function to process the document and create a retriever using Langchain
def create_retriever_from_pdf(pdf_text):
    """
    Process the PDF text, split it into chunks, embed the chunks, and create a FAISS retriever.
    """
    try:
        # Split the text into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_text(pdf_text)

        # Initialize HuggingFace Embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Create FAISS vector store from texts and embeddings
        docsearch = FAISS.from_texts(texts, embeddings)

        # Create a retriever from the vector store
        retriever = docsearch.as_retriever()
        return retriever
    except Exception as e:
        st.error(f"Error creating retriever: {e}")
        return None

# Function to initialize the RetrievalQA chain
def initialize_qa(llm, retriever):
    """
    Initialize the RetrievalQA chain with the provided LLM and retriever.
    """
    try:
        qa = RetrievalQA.from_llm(
            llm=llm,
            retriever=retriever,
            # Removed chain_type to fix validation error
            # chain_type="stuff"  # This line is removed
        )
        return qa
    except Exception as e:
        st.error(f"Error initializing QA system: {e}")
        return None

# Streamlit App Layout
def main():
    st.set_page_config(page_title="📄🤖 PDF Q&A App with LLaMA", layout="wide")
    st.title("📄🤖 Chat with Your PDF Documents using LLaMA 3.1")
    st.markdown("Upload a PDF and ask questions about its content.")

    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar for PDF upload
    with st.sidebar:
        st.header("🛠️ Upload PDF")
        uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file:
        # Extract text from the uploaded PDF
        pdf_text = extract_text_from_pdf(uploaded_file)
        if pdf_text:
            st.success("📄 PDF uploaded and text extracted successfully!")

            # Load the local LLaMA model
            llm = load_llama_model()

            if llm:
                # Create retriever from the PDF text
                retriever = create_retriever_from_pdf(pdf_text)

                if retriever:
                    # Initialize the RetrievalQA chain
                    qa = initialize_qa(llm, retriever)

                    if qa:
                        st.success("✅ QA system initialized successfully!")

                        # Chat input for user questions
                        user_question = st.text_input("💬 Ask a question based on the uploaded PDF:")

                        if user_question:
                            with st.spinner("🔍 Retrieving answer..."):
                                try:
                                    # Get response from the QA system
                                    response = qa.run(user_question)

                                    # Display the response
                                    st.markdown(f"**Answer:** {response}")

                                    # Append to chat history
                                    st.session_state.chat_history.append(("User", user_question))
                                    st.session_state.chat_history.append(("LLaMA", response))
                                except Exception as e:
                                    st.error(f"Error generating response: {e}")

                        # Display chat history
                        if st.session_state.chat_history:
                            st.markdown("### 🗨️ Chat History")
                            for speaker, message in st.session_state.chat_history:
                                if speaker == "User":
                                    st.markdown(f"**You:** {message}")
                                else:
                                    st.markdown(f"**LLaMA:** {message}")

if __name__ == "__main__":
    main()
