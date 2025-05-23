

# **RAG-Based PDF Question-Answering System**

This project demonstrates the use of **Retrieval-Augmented Generation (RAG)** for extracting detailed and contextually accurate answers from PDF documents. It integrates multiple AI technologies, such as **Sentence Transformers** for text embeddings, **FAISS** for similarity search, and **Groq API** for leveraging the **Mistral-SABA-24B** model to generate responses based on the retrieved context.

Overview

The goal of this project is to create an interactive web application where users can upload PDF files and ask questions regarding the content of these documents. The app processes the PDF, retrieves relevant context from the text, and uses a state-of-the-art language model to generate a detailed and accurate response.

## **Key Features**

- **PDF Processing**: Text extraction from uploaded PDFs using PyMuPDF (fitz).
- **Text Chunking & Embedding**: The extracted text is split into manageable chunks, which are then converted into embeddings using the `SentenceTransformer`.
- **Contextual Search**: Using FAISS, the system performs a similarity search to retrieve the most relevant text chunks.
- **Answer Generation**: The app utilizes the **Mistral-SABA-24B** model via the **Groq API** to provide a detailed answer based on the retrieved context.
- **Caching Mechanism**: To optimize performance, embeddings and text chunks are cached locally, ensuring that repeated queries for the same PDF file can be answered quickly.

## **Technologies Used**

- **Streamlit**: Web framework for creating the interactive user interface.
- **PyMuPDF (fitz)**: Library for extracting text from PDF documents.
- **SentenceTransformers**: A library for generating high-quality sentence embeddings.
- **FAISS**: A library for performing efficient similarity search on embeddings.
- **Groq API**: An AI inference service for using the Mistral-SABA-24B model.
- **Langchain**: A tool for text splitting, making it easier to handle large documents.


## **System Requirements**
- Python 3.7+
- Streamlit
- Sentence-Transformers
- PyMuPDF
- FAISS-CPU
- Requests
- Langchain
- Access to the **Groq API** (for the Mistral-SABA-24B model)


**Dependencies**:
- `streamlit`
- `sentence-transformers`
- `PyMuPDF`
- `faiss-cpu`
- `requests`
- `langchain`


## **How to Run**

1. **Start the Streamlit Application**:

```bash
streamlit run app.py
```

2. **Upload a PDF**: On the web interface, upload a PDF file to process.

3. **Ask a Question**: Once the PDF is processed, enter a question in the text box, and the system will generate a response based on the content of the PDF.

## **How It Works**

### **1. PDF Upload & Processing**
- The user uploads a PDF through the Streamlit interface.
- The system extracts the text from the PDF using **PyMuPDF**.

### **2. Text Chunking & Embedding**
- The text is split into smaller chunks to facilitate efficient processing.
- Each chunk is converted into an embedding using the **Sentence-Transformer** model.

### **3. Similarity Search**
- **FAISS** is used to perform a similarity search over the text embeddings, finding the most relevant chunks based on the query.

### **4. Answer Generation**
- The system sends the context retrieved from the text chunks along with the query to the **Groq API**, which uses the **Mistral-SABA-24B** model to generate a detailed and accurate answer.

### **5. Caching**
- The system caches embeddings and chunks to avoid recalculating them for subsequent queries, improving performance.

## **Usage Example**

1. Upload a PDF document.
2. Ask questions like:
    - "What is the summary of chapter 2?"
    - "Can you explain the methodology mentioned in section 4?"
    - "Provide details about the results on page 12."

The system will retrieve the most relevant context from the PDF and use the **Mistral-SABA-24B** model to generate a detailed answer.

## **Development**

For further development or modifications to the system:
1. Ensure your **Groq API key** is set up correctly.
2. If caching is disabled, the system will process embeddings each time a new document is uploaded, which could slow down response times.
3. The project is designed to be modular, so adding new models or optimizing the search process is easy.

