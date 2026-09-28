# CPG & Retail Domain-Specific RAG Chatbot

A domain-specific **Retrieval-Augmented Generation (RAG)** chatbot for the **CPG (Consumer Packaged Goods) and Retail domain**.

The application loads CPG/Retail PDF documents, splits them into smaller chunks, creates embeddings using **HuggingFace**, stores them in a **FAISS vector database**, retrieves the most relevant documents for a user query, and generates an answer using **Groq LLM**.

---

## 🚀 Project Overview

This project demonstrates an end-to-end RAG pipeline:

```text
                CPG / Retail PDFs
                       │
                       ▼
                PDF Document Loader
                       │
                       ▼
                Text Chunking
                       │
                       ▼
          HuggingFace Embeddings
                       │
                       ▼
                  FAISS Index
                       │
                       │
              User Question
                       │
                       ▼
                Similarity Search
                       │
                       ▼
              Top-K Relevant Chunks
                       │
                       ▼
             Context + User Query
                       │
                       ▼
                  Groq LLM
                       │
                       ▼
                Final Answer
                       │
                       ▼
             Source Documents
```

---

# 🎯 Objective

The objective of this project is to build a domain-specific chatbot that can answer questions using information contained in CPG and Retail business documents.

The chatbot is designed to answer questions related to:

* Products
* Product pricing
* Sales
* Inventory
* Invoices
* Promotions
* Returns
* Suppliers
* Store operations
* Retail policies

The chatbot should not invent information that is not available in the provided documents.

---

# 🏢 Domain

**Domain:** CPG – Consumer Packaged Goods / Retail

Example business areas:

* Product Catalog
* Retail Sales
* Inventory Management
* Supplier Management
* Promotions
* Store Operations
* Invoice Management
* Return Policies

---

# 📂 Dataset

The project uses a collection of CPG/Retail PDF documents.

Example documents:

```text
data/
│
├── retail_inventory_policy.pdf
├── retail_invoice_001.pdf
├── retail_invoice_002.pdf
├── retail_product_catalog.pdf
├── retail_promotion_policy.pdf
├── retail_return_policy.pdf
├── retail_sales_report.pdf
├── retail_store_operations.pdf
└── retail_supplier_policy.pdf
```

These documents contain synthetic CPG/Retail business information for educational and demonstration purposes.

---

# 🛠️ Technology Stack

| Technology                     | Purpose                 |
| ------------------------------ | ----------------------- |
| Python                         | Application development |
| LangChain                      | RAG orchestration       |
| PyPDF                          | PDF document loading    |
| RecursiveCharacterTextSplitter | Text chunking           |
| HuggingFace                    | Text embeddings         |
| Sentence Transformers          | Embedding model         |
| FAISS                          | Vector database         |
| Groq                           | LLM inference           |
| python-dotenv                  | Environment variables   |
| CLI                            | User interface          |

---

# 🤖 Models Used

## Embedding Model

```text
sentence-transformers/all-MiniLM-L6-v2
```

This model converts text chunks into numerical vectors.

These vectors are stored in FAISS and used for semantic similarity search.

## LLM

The project uses Groq for fast LLM inference.

Current configuration:

```python
LLM_MODEL = "openai/gpt-oss-120b"
```

You can change the model in:

```text
app/config.py
```

---

# 📁 Project Structure

```text
rag-domain-assignment/
│
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── ingest.py
│   ├── retriever.py
│   ├── rag_chain.py
│   └── main.py
│
├── data/
│   └── *.pdf
│
├── store/
│   ├── vector_index/
│   │   ├── index.faiss
│   │   └── index.pkl
│   │
│   └── chunks/
│       └── chunks.txt
│
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

# 🔄 RAG Pipeline

## 1. Load Documents

PDF documents are loaded from the `data/` directory.

```python
PyPDFDirectoryLoader()
```

The loader extracts text from all PDF files.

---

## 2. Text Chunking

Large documents are split into smaller chunks using:

```text
RecursiveCharacterTextSplitter
```

Current configuration:

```python
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
```

Chunking allows the retriever to find relevant sections instead of passing entire documents to the LLM.

---

## 3. Generate Embeddings

Each text chunk is converted into a vector using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Example:

```text
" Coca-Cola 500 ml price is ₹40 "

              ↓

        Embedding Vector
```

---

## 4. Store Embeddings

The generated vectors are stored locally using:

```text
FAISS
```

The vector index is saved under:

```text
store/vector_index/
```

---

## 5. Retrieve Relevant Documents

When the user asks a question:

```text
Which products are below their reorder point?
```

FAISS performs a similarity search and returns the most relevant chunks.

Current configuration:

```python
TOP_K = 4
```

Therefore, the system retrieves the top 4 relevant chunks.

---

## 6. Generate Answer

The retrieved chunks are combined with the user's question.

The prompt instructs the LLM to:

* Use only retrieved information
* Avoid hallucinating
* Preserve numerical values
* Avoid inventing products or prices
* Mention when information is unavailable
* Answer specifically for the CPG/Retail domain

The final request is sent to Groq.

---

# 🔐 Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

Do not commit `.env` to GitHub.

Use `.env.example` instead:

```env
GROQ_API_KEY=your_groq_api_key
```

---

# 📦 Installation

## Option 1: Using pip

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Option 2: Using uv

Initialize the environment:

```bash
uv venv
```

Activate it:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
uv pip install -r requirements.txt
```

---

# 📋 Requirements

Example `requirements.txt`:

```text
langchain
langchain-community
langchain-groq
langchain-huggingface
langchain-text-splitters
faiss-cpu
pypdf
sentence-transformers
python-dotenv
```

---

# ▶️ Running the Project

## Step 1: Add PDFs

Place all CPG/Retail PDFs inside:

```text
data/
```

Example:

```text
data/
├── retail_product_catalog.pdf
├── retail_sales_report.pdf
├── retail_inventory_policy.pdf
├── retail_promotion_policy.pdf
└── ...
```

---

# Step 2: Create FAISS Index

Run:

```bash
python -m app.ingest
```

Using `uv`:

```bash
uv run python -m app.ingest
```

The ingestion process:

```text
PDF
 ↓
Extract text
 ↓
Chunk text
 ↓
Generate embeddings
 ↓
Create FAISS index
 ↓
Save index
```

You should see a message similar to:

```text
Ingestion completed successfully.
```

---

# Step 3: Start the Chatbot

Run:

```bash
python -m app.main
```

Or:

```bash
uv run python -m app.main
```

You should see:

```text
==================================================
        CPG RETAIL RAG ASSISTANT
==================================================

Ask questions about:
- Products
- Sales
- Inventory
- Invoices
- Promotions
- Returns
- Suppliers
- Store Operations

Type 'exit' or 'quit' to stop.
```

---

# 💬 Example Questions

### Product

```text
What is the price of Coca-Cola 500 ml?
```

### Inventory

```text
Which products are below their reorder point?
```

### Invoice

```text
What is the total amount of invoice INV-1001?
```

### Promotion

```text
What promotion is available for snacks?
```

### Sales

```text
Which category had the highest September sales?
```

### Supplier

```text
What is the MOQ for SUP001?
```

### Store Operations

```text
What are the daily store closing procedures?
```

### Return Policy

```text
What is the return period for eligible products?
```

---

# 📌 Example Output

```text
Question:
What is the price of Coca-Cola 500 ml?

Answer:
The price of Coca-Cola 500 ml is ₹40.

Sources:
- retail_product_catalog.pdf - Page 1
```

---

# 🧠 Hallucination Control

The RAG prompt contains domain-specific restrictions.

The LLM is instructed to answer only using the retrieved documents.

For example, if the user asks:

```text
What is the price of Pepsi 2L?
```

and the information does not exist in the documents, the chatbot responds:

```text
I could not find this information in the provided CPG documents.
```

This prevents the model from generating an unsupported price.

---

# 📚 Source Citations

The chatbot returns the source documents used to generate the answer.

Example:

```text
Answer:
The reorder point for Product A is 50 units.

Sources:
- retail_inventory_policy.pdf - Page 2
- retail_product_catalog.pdf - Page 3
```

This makes the response easier to verify.

---

# 🧩 Main Components

## `app/config.py`

Contains project configuration:

```text
Embedding model
LLM model
Chunk size
Chunk overlap
Top-K
Directory paths
Groq API key
```

---

## `app/ingest.py`

Responsible for:

```text
PDF loading
↓
Text splitting
↓
Embedding generation
↓
FAISS creation
↓
Saving vector index
```

Run:

```bash
python -m app.ingest
```

---

## `app/retriever.py`

Responsible for:

```text
Loading FAISS
↓
Converting query to embedding
↓
Similarity search
↓
Returning relevant chunks
```

---

## `app/rag_chain.py`

Responsible for:

```text
User question
↓
Retriever
↓
Relevant documents
↓
Prompt
↓
Groq LLM
↓
Answer + sources
```

---

## `app/main.py`

Provides the command-line chatbot interface.

It handles:

* User input
* Empty questions
* Exit command
* Exceptions
* Answer display
* Source display

---

# ⚙️ Configuration

You can modify the following values in:

```text
app/config.py
```

### Chunk Size

```python
CHUNK_SIZE = 800
```

### Chunk Overlap

```python
CHUNK_OVERLAP = 150
```

### Number of Retrieved Documents

```python
TOP_K = 4
```

### Embedding Model

```python
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```

### Groq Model

```python
LLM_MODEL = "openai/gpt-oss-120b"
```

---

# 🐛 Troubleshooting

## 1. Groq API Key Error

Error:

```text
GROQ_API_KEY is not set
```

Solution:

Create `.env`:

```env
GROQ_API_KEY=your_actual_api_key
```

Restart the application.

---

## 2. FAISS Index Not Found

Error:

```text
FAISS index not found
```

Run ingestion first:

```bash
python -m app.ingest
```

Then start the chatbot:

```bash
python -m app.main
```

---

## 3. No PDF Files Found

Make sure your files are inside:

```text
data/
```

For example:

```text
data/retail_product_catalog.pdf
```

---

## 4. Model Not Found

If Groq returns:

```text
model_not_found
```

check the configured model in:

```text
app/config.py
```

For the current project configuration, use:

```python
LLM_MODEL = "openai/gpt-oss-120b"
```

---

# 🔒 Security

Never commit your API key.

Add the following to `.gitignore`:

```text
.env
.venv/
__pycache__/
*.pyc
```

The GitHub repository should contain:

```text
.env.example
```

but not:

```text
.env
```

---

# 🚫 Out-of-Domain Questions

This chatbot is designed for the CPG/Retail domain.

For example:

```text
Who is the Prime Minister of India?
```

or:

```text
Explain quantum physics.
```

The application should not use external knowledge to answer such questions.

The system prompt restricts responses to the information available in the CPG/Retail documents.

---

# 📊 Evaluation Criteria

This project addresses the following RAG evaluation areas:

### 1. Correctness

Answers are generated using retrieved document context.

### 2. Retrieval Relevance

FAISS similarity search retrieves the most relevant chunks.

### 3. Domain Understanding

The prompt is specifically designed for:

```text
CPG + Retail
```

### 4. Code Quality

The project separates:

```text
Ingestion
Retriever
RAG Chain
Application
Configuration
```

### 5. Edge Cases

The application handles:

* Empty questions
* Missing PDFs
* Missing FAISS index
* Missing API key
* Invalid queries
* Runtime exceptions
* Exit commands

---

# 🚀 Future Enhancements

The following features can be added later:

* Streamlit web interface
* PDF page-level citations
* Document upload through UI
* Chat history
* Conversation memory
* Hybrid search
* BM25 + vector search
* Reranking
* Metadata filtering
* DOCX support
* TXT support
* Answer confidence
* Query rewriting
* Multi-query retrieval
* Evaluation using RAGAS
* Docker deployment
* FastAPI backend
* Azure deployment

---

# 🌐 Optional Streamlit UI

A Streamlit interface can be added on top of the existing RAG pipeline.

Example architecture:

```text
              Streamlit UI
                   │
                   ▼
             User Question
                   │
                   ▼
             RAG Pipeline
              /         \
             ▼           ▼
         FAISS          Groq
             \           /
              ▼         ▼
                Answer
                   │
                   ▼
             Source Files
```

---

# 🧪 Testing Strategy

Test the application using questions whose answers are present in the documents.

Example:

```text
What is the price of Coca-Cola 500 ml?
```

Then test questions where information is not available:

```text
What is the price of iPhone 17?
```

The chatbot should not invent an answer.

Also test:

```text
""
```

```text
exit
```

```text
quit
```

---

# 📈 RAG Architecture Summary

```text
                     ┌─────────────────────┐
                     │   CPG/Retail PDFs   │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │   PDF Loader        │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │   Text Chunking     │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │ HuggingFace Model   │
                     │    Embeddings       │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │       FAISS         │
                     │    Vector Store     │
                     └──────────┬──────────┘
                                │
                                │
                       User Question
                                │
                                ▼
                     ┌─────────────────────┐
                     │ Similarity Search   │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │ Relevant Context    │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │     Groq LLM        │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │ Answer + Sources    │
                     └─────────────────────┘
```

---

# 🎓 Skills Demonstrated

This project demonstrates practical knowledge of:

* Generative AI
* Large Language Models
* Retrieval-Augmented Generation
* LangChain
* Prompt Engineering
* Vector Databases
* FAISS
* HuggingFace Embeddings
* Sentence Transformers
* Groq
* Semantic Search
* Document Processing
* PDF Processing
* Python
* Environment Variables
* Modular Application Design
* Domain-specific AI
* Hallucination Control

---

# 👨‍💻 Author

**Satish Shinde**

CPG/Retail Domain-Specific RAG Chatbot built as a Generative AI project.

---

# ⭐ Project Summary

```text
PDF Documents
      ↓
Document Loading
      ↓
Chunking
      ↓
HuggingFace Embeddings
      ↓
FAISS Vector Database
      ↓
Semantic Retrieval
      ↓
Context Injection
      ↓
Groq LLM
      ↓
CPG/Retail Answer
      ↓
Source Citations
```

This project provides an end-to-end implementation of a **production-style domain-specific RAG pipeline** using open-source embeddings, FAISS vector search, and Groq LLM inference.
