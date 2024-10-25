
# Equity Research Tool 📈

A Streamlit-based application for analyzing financial and news articles. This tool processes article URLs to enable question answering with contextual source retrieval.

## Features

- **URL Processing**: Input up to three URLs of news articles.
- **Data Embedding**: Converts article text into a searchable FAISS index using Langchain’s OpenAI embeddings.
- **Interactive Q&A**: Ask questions related to the content, with answers sourced from the articles.
- **Persistent Index**: Saves the FAISS index to a `.pkl` file, allowing retrieval between sessions.

## Requirements

- **Python 3.7+**
- **Dependencies**:
  - `Streamlit`
  - `Langchain`
  - `OpenAI API`
  - `FAISS`
  - `dotenv`
  - `pickle`
- **OpenAI API Key**: Required in a `.env` file for embedding and language model services.

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd equity-research-tool
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up the OpenAI API key**:
   - Create a `.env` file in the root directory.
   - Add your OpenAI API key:
     ```env
     OPENAI_API_KEY=your_openai_api_key
     ```

## Usage

1. **Run the application**:
   ```bash
   streamlit run app.py
   ```

2. **Input URLs**:
   - Enter up to three article URLs in the sidebar.
   - Click “Process URLs” to load and process the article data.

3. **Ask Questions**:
   - Enter your question in the main input field.
   - The model retrieves relevant information and displays an answer with sources.

## Code Structure

- `app.py`: Main application file containing functionalities for data loading, embedding, and Q&A.
- **Modules**:
  - `langchain`: Handles OpenAI model and embeddings.
  - `FAISS`: Manages vector storage and retrieval.

## Notes

- **Persistence**: The FAISS index is saved to `faiss_store_openai.pkl`. If the file is missing, you must process URLs to rebuild the index.
- **Environment Variables**: The OpenAI API key is loaded from a `.env` file.

## Troubleshooting

- **Missing `.pkl` file**: If `faiss_store_openai.pkl` is not present, re-run the URL processing to create a new index.
- **Network Issues**: Ensure a stable connection during URL processing, as data loading may fail otherwise.
