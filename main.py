from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline

# ✅ Step 1: Load PDF
loader = PyPDFLoader("sample.pdf")
documents = loader.load()
print("Pages:", len(documents))

# ✅ Step 2: Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
texts = text_splitter.split_documents(documents)
print("Chunks:", len(texts))

# ✅ Step 3: Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# ✅ Step 4: Store in vector DB
db = FAISS.from_documents(texts, embeddings)

# ✅ Step 5: Load LLM (FIXED pipeline)


pipe = pipeline(
    "text2text-generation",   # ✅ CORRECT
    model="google/flan-t5-large",
    max_length=256
)

llm = HuggingFacePipeline(pipeline=pipe)

# ✅ Step 6: Ask question loop
while True:
    query = input("\nAsk question (type 'exit' to quit): ")
    
    if query.lower() == "exit":
        break

    docs = db.similarity_search(query, k=2)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Answer the question based on the context below:

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    response = llm.invoke(prompt)

    print("\n💡 Answer:\n", response)