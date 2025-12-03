# Tamil Bible RAG Application

This is a RAG (Retrieval-Augmented Generation) application that allows users to ask questions about the Bible in Tamil or English. It uses Google's Gemini Pro model for generation and ChromaDB for vector storage.

## Features

- **Bible Q&A**: Answers questions based on the Tamil Common Bible.
- **Web Search Fallback**: If the answer isn't found in the Bible, it searches the web using DuckDuckGo.
- **Source Citations**: Provides Bible verse references for answers.
- **Streamlit UI**: Simple and interactive web interface.

## Deployment on Streamlit Cloud

This repository is configured for deployment on Streamlit Cloud.

### Prerequisites

- A Google Gemini API Key.

### Setup

1. Fork/Clone this repository.
2. Deploy to Streamlit Cloud.
3. Add your `GOOGLE_API_KEY` in the Streamlit Cloud secrets.

## Local Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the app:
   ```bash
   streamlit run app.py
   ```
