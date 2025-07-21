import os
import streamlit as st
import pickle
import time
from langchain_openai import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env

st.title("Equity Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
   url = st.sidebar.text_input(f"URL {i+1}")
   urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()
llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.9, max_tokens=500)

if process_url_clicked:
    # Validate URLs
    valid_urls = [url for url in urls if url.strip()]
    if not valid_urls:
        main_placeholder.error("Please enter at least one valid URL.")
        st.stop()
    
    # Load data
    try:
        loader = UnstructuredURLLoader(urls=valid_urls)
        main_placeholder.text("Data Loading...Started...✅✅✅")
        data = loader.load()
    except Exception as e:
        main_placeholder.error(f"Error loading data: {str(e)}")
        st.stop()
    # Split data
    if data:
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000
            )
            main_placeholder.text("Text Splitter...Started...✅✅✅")
            docs = text_splitter.split_documents(data)

            if docs:
                # Create embeddings and save it to FAISS index
                embeddings = OpenAIEmbeddings()
                vectorstore_openai = FAISS.from_documents(docs, embeddings)
                main_placeholder.text("Embedding Vector Started Building...✅✅✅")
                time.sleep(2)
                
                # Save the FAISS index to a pickle file
                with open(file_path, "wb") as f:
                    pickle.dump(vectorstore_openai, f)
                main_placeholder.success("Processing completed! You can now ask questions.")
            else:
                main_placeholder.error("Text Splitter produced empty documents. Check data.")
        except Exception as e:
            main_placeholder.error(f"Error processing documents: {str(e)}")
    else:
        main_placeholder.error("Data loading failed. Check URLs or network connection.")

query = st.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        try:
            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)
            
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            
            # Display results
            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")
                for source in sources_list:
                    if source.strip():
                        st.write(source)
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
    else:
        st.warning("Please process URLs first to create the knowledge base.")

