import streamlit as st
import json
from google.cloud import aiplatform
from langchain_google_vertexai import VectorSearchVectorStore, VertexAIEmbeddings

# Page config
st.set_page_config(
    page_title="USMC Doctrine Vector Search",
    page_icon="🪖",
    layout="wide"
)

@st.cache_resource
def load_vector_store():
    """Load and cache the vector store connection"""
    # Load config
    with open("config.json", "r") as f:
        config = json.load(f)

    PROJECT_ID = config["project_id"]
    LOCATION = config["location"]
    INDEX_ID = config["index_id"]
    ENDPOINT_ID = config["endpoint_id"]
    GCS_BUCKET_NAME = "usmc_bucket_test"

    # Initialize Vertex AI
    aiplatform.init(project=PROJECT_ID, location=LOCATION)

    # Initialize embeddings
    embedder = VertexAIEmbeddings(
        model_name="text-embedding-005",
        project=PROJECT_ID,
        location=LOCATION
    )

    # Initialize Vector Store
    vector_store = VectorSearchVectorStore.from_components(
        project_id=PROJECT_ID,
        region=LOCATION,
        gcs_bucket_name=GCS_BUCKET_NAME,
        index_id=INDEX_ID,
        endpoint_id=ENDPOINT_ID,
        embedding=embedder,
        stream_update=True
    )
    
    return vector_store

def search_documents(query, num_results=5):
    """Search the vector database for relevant documents"""
    vector_store = load_vector_store()
    results = vector_store.similarity_search(query, k=num_results)
    return results

# App UI
st.title("🪖 USMC Doctrine Vector Search")
st.markdown("Search through Marine Corps Doctrinal Publication (MCDP-1) using AI-powered semantic search")

# Sidebar
with st.sidebar:
    st.header("⚙️ Search Settings")
    num_results = st.slider("Number of results", min_value=1, max_value=20, value=5)
    
    st.header("📊 Database Info")
    st.info("""
    **Vector Database Stats:**
    - 619 document chunks
    - 768-dimensional embeddings
    - Source: MCDP-1.pdf
    - Model: text-embedding-005
    """)

# Main search interface
st.header("🔍 Search Military Doctrine")

# Search input
query = st.text_input(
    "Enter your question or topic:",
    placeholder="e.g., What is leadership in the military? How does command structure work?",
    help="Ask questions about Marine Corps doctrine, leadership, tactics, strategy, etc."
)

# Search button
if st.button("🔍 Search", type="primary") or query:
    if query.strip():
        with st.spinner("Searching through military doctrine..."):
            try:
                results = search_documents(query, num_results)
                
                if results:
                    st.success(f"Found {len(results)} relevant documents")
                    
                    # Display results
                    for i, result in enumerate(results, 1):
                        with st.expander(f"📄 Result {i}: {result.metadata.get('element_type', 'Document')}", expanded=(i<=2)):
                            # Main content
                            st.markdown("**Summary:**")
                            st.write(result.page_content)
                            
                            # Metadata
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.markdown(f"**Source:** {result.metadata.get('source', 'Unknown')}")
                            with col2:
                                page_num = result.metadata.get('page_number')
                                st.markdown(f"**Page:** {page_num if page_num else 'Unknown'}")
                            with col3:
                                seq_id = result.metadata.get('sequence_id')
                                st.markdown(f"**Sequence:** {seq_id if seq_id is not None else 'Unknown'}")
                            
                            # Original text preview (no nested expander)
                            if result.metadata.get('original_text'):
                                st.markdown("**📖 Original Text Preview:**")
                                st.text(result.metadata['original_text'])
                else:
                    st.warning("No results found. Try a different query.")
                    
            except Exception as e:
                st.error(f"Search error: {str(e)}")
                st.error("Make sure your vector database is properly configured.")
    else:
        st.warning("Please enter a search query.")




# Footer
st.markdown("---")
st.markdown("*Built with Streamlit, Google Cloud Vector Search, and LangChain*") 