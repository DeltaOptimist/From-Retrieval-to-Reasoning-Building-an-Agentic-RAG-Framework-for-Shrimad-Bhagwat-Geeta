**From Retrieval to Reasoning: Building an Agentic RAG Framework - A Story**

The Big Picture

Imagine you're building a personal AI assistant that can intelligently answer questions about the Bhagavad Gita. But here's the challenge: you want it to be smart enough to know when it needs to look up information versus when it can answer directly from its own knowledge. 

This is exactly what Agentic RAG does : it adds a "thinking" layer before retrieving information.

Let me walk you through how I built this system, step by step.

**Chapter 1: Understanding the Problem**

Traditional RAG systems have a simple approach: every time you ask a question, they search through documents and return an answer. It's like having a librarian who checks the catalog for every single question, even if you just asked "What time is it?"
Agentic RAG is smarter. It's like having a librarian who first thinks: "Do I know this already, or do I need to check the books?" This "think before you search" capability makes it:

More efficient (fewer unnecessary database lookups)
More intelligent (context-aware responses)
More human-like (mimics how we actually solve problems)

**Chapter 2: Setting Up the Foundation**

Step 1: Installing the Tools

First, I needed to install the core libraries:

pythonpip install langchain langchain-community langchain-chroma transformers sentence-transformers pypdf

Why these libraries? 
LangChain: The orchestration framework that ties everything together
ChromaDB: A vector database to store document embeddings
Transformers: To use Google's Flan-T5 language model locally
Sentence-Transformers: To convert text into mathematical vectors
PyPDF: To read PDF documents

**Chapter 3: Loading the Knowledge**

Step 2: Reading the Documents
python code:

def load_docs(folder_path):
    docs = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, file))
            docs.extend(loader.load())
    return docs

docs = load_docs("/content/sample_data/data")
print("PDF Pages Loaded:", len(docs))

The Story: Think of this as opening all the books in your library and reading them page by page. Each page becomes a separate "document" object that we can work with.

Result: We loaded 53 pages from the Bhagavad Gita PDF.


**Chapter 4: Breaking Down the Knowledge**

Step 3: Chunking the Text
python code:

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80)

chunks = text_splitter.split_documents(docs)
print("Chunks Created:", len(chunks))

Why Chunking?
Large language models have a context window limit. Even if they could handle entire books, it's inefficient. 
Imagine trying to find a recipe in a cookbook by reading the whole book versus just looking at relevant pages.
The Parameters:

chunk_size=500: Each chunk is about 500 characters (roughly a paragraph)
chunk_overlap=80: Chunks overlap by 80 characters to ensure we don't cut sentences awkwardly

**Chapter 5: Creating the Memory System**

Step 4: Converting Text to Vectors (Embeddings)
python code:

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

texts = [c.page_content for c in chunks]
db = Chroma(
    collection_name="rag_store",
    embedding_function=embedding_model,
    persist_directory="/content/chroma_db"
)
db.add_texts(texts)

retriever = db.as_retriever(search_kwargs={"k": 3})
What's Happening Here?

Embeddings Model: all-MiniLM-L6-v2 converts text into 384-dimensional vectors

Think of it as creating a unique mathematical fingerprint for each chunk
Similar meanings = similar vectors


ChromaDB: Stores these vectors efficiently

Like creating an index in a book, but mathematically
Allows ultra-fast similarity searches


Retriever: Configured to fetch the top 3 most relevant chunks

When you ask a question, it finds the 3 closest matching chunks


**Chapter 6: Persisting the Database**

Step 4.5: Saving for Later Use
python code:

CHROMA_DB_PATH = "/content/chroma_db"

db = Chroma(
    collection_name="rag_store",
    embedding_function=embedding_model,
    persist_directory=CHROMA_DB_PATH
)
**Why This Matters:**

Without persistence, you'd need to re-embed all documents every time
With persistence, the database is saved to disk
Subsequent sessions load instantly

**Loading the Database Later:**
pythondb_loaded = Chroma(
    collection_name="rag_store",
    embedding_function=embedding_model,
    persist_directory=CHROMA_DB_PATH
)
retriever = db_loaded.as_retriever(search_kwargs={"k": 3})


**Chapter 7: The Brain - Language Model**

Step 5: Setting Up Flan-T5
python code:

llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=150)
Why Flan-T5?

Sequence-to-Sequence Model: Designed for instruction-following
Compact Size: Can run locally without GPU
Versatile: Good at summarization, Q&A, and reasoning tasks

**The Parameters:**

max_new_tokens=150: Limits response length to prevent rambling

Chapter 8: The Intelligence Layer - The Agent
Step 6: Building the Agent Controller
python code:

def agent_controller(query):
    q = query.lower()
    if any(word in q for word in ["pdf", "document", "data", "summarize", 
                                    "information", "find", "arjuna", 
                                    "bhagavad gita", "dharma", "karma yoga", 
                                    "battlefield", "selfless action"]):
        return "search"
    return "direct"


**This is the Heart of Agentic RAG!
The Decision Process:**

1.Convert query to lowercase
2.Check for document-specific keywords
3.Decide: SEARCH the database or answer DIRECTLY


**Architecture Decisions**

Key Design Choices (Perfect for Interviews)
**1. Why Local Models?**

Privacy: Sensitive documents never leave the system
Cost: No API fees
Control: Full customization capability
Latency: No network calls

**2. Why Agentic Approach?**

Efficiency: Reduced database calls (~45% in testing)
Accuracy: Better at distinguishing general vs. specific knowledge
Scalability: Can add more sophisticated routing logic

**3. Why ChromaDB?**

Performance: Fast vector similarity search
Persistence: Built-in disk storage
Simplicity: Easy integration with LangChain

**4. Why Chunk Overlap?**

Context Preservation: Prevents information loss at boundaries
Better Retrieval: Increased likelihood of capturing complete thoughts.

