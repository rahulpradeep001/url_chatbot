import os
import streamlit as st
import pickle
import time
import logging
from urllib.parse import urlparse
from pathlib import Path
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()  # take environment variables from .env

st.title("Equity Research Tool 📈")
st.sidebar.title("News Article URLs")

def is_valid_url(url):
    """Validate URL format and scheme"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
    except Exception:
        return False

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url.strip():
        url = url.strip()
        if is_valid_url(url):
            urls.append(url)
        else:
            st.sidebar.warning(f"Invalid URL format: {url}")

process_url_clicked = st.sidebar.button("Process URLs")

# Use a more secure file path
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
file_path = DATA_DIR / "faiss_store_openai.pkl"

main_placeholder = st.empty()
# Initialize LLM with better configuration
try:
    llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.7, max_tokens=500)
except Exception as e:
    st.error(f"Failed to initialize OpenAI model: {str(e)}")
    st.stop()
if process_url_clicked:
    if not any(urls):
        st.error("Please enter at least one URL")
    else:
        try:
            # Load data
            valid_urls = [url for url in urls if url]
            logger.info(f"Processing {len(valid_urls)} URLs")
            
            loader = UnstructuredURLLoader(urls=valid_urls)
            main_placeholder.text("Data Loading...Started...✅✅✅")
            data = loader.load()
            
            if not data:
                st.error("No data could be loaded from the provided URLs")
                logger.error("No data loaded from URLs")
                st.stop()
            
            if data:
                # Split data
                text_splitter = RecursiveCharacterTextSplitter(
                    separators=['\n\n', '\n', '.', ','],
                    chunk_size=1000
                )
                main_placeholder.text("Text Splitter...Started...✅✅✅")
                docs = text_splitter.split_documents(data)

                if docs:
                    # Create embeddings and save it to FAISS index
                    try:
                        embeddings = OpenAIEmbeddings()
                        vectorstore_openai = FAISS.from_documents(docs, embeddings)
                        main_placeholder.text("Embedding Vector Started Building...✅✅✅")
                        time.sleep(2)

                        # Save the FAISS index to a pickle file securely
                        with open(file_path, "wb") as f:
                            pickle.dump(vectorstore_openai, f)
                        
                        logger.info(f"FAISS index saved to {file_path}")
                        main_placeholder.text("Processing completed! You can now ask questions.")
                    except Exception as embedding_error:
                        st.error(f"Failed to create embeddings: {str(embedding_error)}")
                        logger.error(f"Embedding error: {embedding_error}")
                else:
                    st.error("Text splitter produced empty documents. Check data.")
            else:
                st.error("Data loading failed. Check URLs or network connection.")
                
        except Exception as e:
            error_msg = f"An error occurred during processing: {str(e)}"
            st.error(error_msg)
            logger.error(f"Processing error: {e}", exc_info=True)

query = main_placeholder.text_input("Question: ")
if query:
    if file_path.exists():
        try:
            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)
            
            # Validate the loaded vectorstore
            if not hasattr(vectorstore, 'as_retriever'):
                st.error("Invalid vectorstore format. Please reprocess URLs.")
                logger.error("Invalid vectorstore format")
            else:
                chain = RetrievalQAWithSourcesChain.from_llm(
                    llm=llm, 
                    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
                )
                result = chain({"question": query}, return_only_outputs=True)
                
                st.header("Answer")
                answer = result.get("answer", "No answer available")
                st.write(answer)

                # Display sources, if available
                sources = result.get("sources", "")
                if sources and sources.strip():
                    st.subheader("Sources:")
                    sources_list = [s.strip() for s in sources.split("\n") if s.strip()]
                    for source in sources_list:
                        st.write(f"• {source}")
                        
        except pickle.PickleError as pe:
            st.error("Corrupted knowledge base file. Please reprocess URLs.")
            logger.error(f"Pickle error: {pe}")
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            st.error(error_msg)
            logger.error(f"Query processing error: {e}", exc_info=True)
    else:
        st.error("Please process URLs first to create the knowledge base.")

