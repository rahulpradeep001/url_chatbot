import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
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
   # load data
   loader = UnstructuredURLLoader(urls=urls)
   main_placeholder.text("Data Loading...Started...✅✅✅")
   data = loader.load()
   # split data
   if data:
       # split data
       text_splitter = RecursiveCharacterTextSplitter(
           separators=['\n\n', '\n', '.', ','],
           chunk_size=1000
       )
       main_placeholder.text("Text Splitter...Started...✅✅✅")
       docs = text_splitter.split_documents(data)

       if docs:
           # create embeddings and save it to FAISS index
           embeddings = OpenAIEmbeddings()
           vectorstore_openai = FAISS.from_documents(docs, embeddings)
           main_placeholder.text("Embedding Vector Started Building...✅✅✅")
           time.sleep(2)

           # # Save the FAISS index to a pickle file
           # vectorstore_openai.save_local(file_path)
       else:
           main_placeholder.text("Text Splitter produced empty documents. Check data.")
   else:
       main_placeholder.text("Data loading failed. Check URLs or network connection.")


   # Save the FAISS index to a pickle file
   with open(file_path, "wb") as f:
       pickle.dump(vectorstore_openai, f)

query = main_placeholder.text_input("Question: ")
if query:
   if os.path.exists(file_path):
       with open(file_path, "rb") as f:
           vectorstore = pickle.load(f)
           chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
           result = chain({"question": query}, return_only_outputs=True)
           # result - dictionary
           st.header("Answer")
           st.write(result["answer"])

           # Display sources, if available
           sources = result.get("sources", "")
           if sources:
               st.subheader("Sources:")
               sources_list = sources.split("\n")  # Split the sources by newline
               for source in sources_list:
                   st.write(source)

