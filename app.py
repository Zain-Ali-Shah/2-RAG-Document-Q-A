import streamlit as st
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# -------------------- LLM --------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

# -------------------- Prompt --------------------
prompt = ChatPromptTemplate.from_template(
    """
You are a helpful research assistant.
Answer the question strictly using the provided context.

<context>
{context}
</context>

Question: {question}
"""
)

# -------------------- Vector Creation --------------------
def create_vector_embedding():
    if "vectorstore" not in st.session_state:
        embeddings = OpenAIEmbeddings()

        loader = PyPDFDirectoryLoader("research_papers")
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(documents[:50])

        vectorstore = FAISS.from_documents(chunks, embeddings)

        st.session_state.vectorstore = vectorstore


# -------------------- RAG Chain (LCEL) --------------------
def build_rag_chain(retriever):
    return (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )


# -------------------- Streamlit UI --------------------
st.title("📄 RAG Document Q&A (Groq + LLaMA 3.1)")

query = st.text_input("Ask a question from the research papers")

if st.button("Create Document Embeddings"):
    create_vector_embedding()
    st.success("Vector database created successfully")

if query:
    if "vectorstore" not in st.session_state:
        st.warning("Please create embeddings first")
    else:
        retriever = st.session_state.vectorstore.as_retriever(
            search_kwargs={"k": 4}
        )

        rag_chain = build_rag_chain(retriever)

        start = time.process_time()
        answer = rag_chain.invoke(query)
        end = time.process_time()

        st.write("### ✅ Answer")
        st.write(answer)

        st.caption(f"⏱️ Response time: {end - start:.2f} seconds")

        with st.expander("🔍 Retrieved Context"):
            docs = retriever.invoke(query)
            for i, doc in enumerate(docs, 1):
                st.markdown(f"**Chunk {i}**")
                st.write(doc.page_content)
                st.divider()
