import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
import numpy as np

# Load cleaned dataset
df = pd.read_csv("cleaned_test.csv")

# Load Sentence Transformer model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Generate embeddings for the dataset
@st.cache_resource
def create_faiss_index():
    corpus = df["TITLE"] + " " + df["ABSTRACT"]
    corpus_embeddings = model.encode(corpus.tolist(), convert_to_numpy=True)
    
    # Create FAISS index
    embedding_dim = corpus_embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(corpus_embeddings)
    
    return index, corpus_embeddings

index, corpus_embeddings = create_faiss_index()

# Function to retrieve similar documents
def retrieve_similar_documents(query, top_k=5):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx, score in zip(indices[0], distances[0]):
        title = df.iloc[idx]["TITLE"]
        abstract = df.iloc[idx]["ABSTRACT"]
        results.append((title, abstract, score))
    
    return results

# Streamlit UI
st.title("🔍 Semantic Search for Research Papers")
st.write("Enter a search query below to retrieve the most relevant research articles.")

# User input field
query_text = st.text_input("Enter your query:", "")

# Optional settings
top_k = st.slider("Number of results to retrieve:", 1, 10, 5)
filter_category = st.selectbox("Filter by category (optional):", ["All", "Neural Networks", "Computer Vision", "NLP"])

# Search button
if st.button("Search"):
    if query_text:
        results = retrieve_similar_documents(query_text, top_k=top_k)
        if results:
            st.subheader("Search Results:")
            for i, (title, abstract, score) in enumerate(results):
                st.markdown(f"### {i+1}. {title}")
                st.write(abstract)
                st.write(f"**Relevance Score:** {score:.4f}")
                st.markdown("---")
        else:
            st.warning("No results found. Try a different query.")
    else:
        st.warning("Please enter a search query.")

# Sidebar with additional options
st.sidebar.header("🔧 Advanced Settings")
if st.sidebar.checkbox("Show raw dataset preview"):
    st.sidebar.write(df.head())

if st.sidebar.button("🔄 Refresh Data"):
    st.experimental_rerun()
