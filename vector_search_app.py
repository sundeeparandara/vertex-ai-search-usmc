import streamlit as st
import json
import os
from google.cloud import aiplatform

# Try different import paths for VertexAI components
try:
    from langchain_google_vertexai import VectorSearchVectorStore, VertexAIEmbeddings
except ImportError:
    try:
        from langchain_google_vertexai.vectorstores import VectorSearchVectorStore
        from langchain_google_vertexai.embeddings import VertexAIEmbeddings
    except ImportError:
        st.error("Failed to import VertexAI components. Please check langchain-google-vertexai version compatibility.")
        st.stop()

def load_config():
    """Load configuration from local file or environment variable"""
    # Try to load from local config.json first
    if os.path.exists("config.json"):
        try:
            with open("config.json", "r") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Failed to load local config.json: {e}")
            st.stop()
    else:
        # Assume running in cloud, get config from environment variable
        config_json = os.getenv('CONFIG_JSON_USMC_AI_1')
        if config_json:
            try:
                return json.loads(config_json)
            except Exception as e:
                st.error(f"Failed to parse CONFIG_JSON_USMC_AI_1 environment variable: {e}")
                st.stop()
        else:
            st.error("No configuration found. Expected either config.json file locally or CONFIG_JSON_USMC_AI_1 environment variable in cloud.")
            st.stop()

# Page config
st.set_page_config(
    page_title="USMC Doctrine Vector Search",
    page_icon="🪖",
    layout="wide"
)

# Environment indicator
if os.path.exists("config.json"):
    st.sidebar.info("💻 Running locally")
else:
    st.sidebar.success("🌐 Running in cloud")

# What gets cached:
# Connection Objects:
# ✅ VectorSearchVectorStore object (connection interface)
# ✅ VertexAIEmbeddings object (API client)
# ✅ Authentication tokens and session info
# ✅ Configuration settings (project ID, location, etc.)
# Size: ~A few MB of connection objects
#
# ☁️ What stays remote (in Google Cloud):
# The Actual Data:
# ❌ 619 × 768 vector matrix (stays in Google Vector Search)
# ❌ Embedding model weights (text-embedding-005 model on Google's servers)
# ❌ Vector database index (your tree-AH algorithm structure)
# ❌ Document content (your 619 document summaries)
# Size: ~Hundreds of MB to GB of actual data
#
# 🔍 What happens when you search:
# Cached: Connection objects (fast lookup)
# Remote API call: Your query → Google's embedding service → 768-dimensional vector
# Remote database query: Vector similarity search in Google Cloud
# Remote API response: Matching documents sent back

@st.cache_resource
def load_vector_store():
    """Load and cache the vector store connection"""
    # Load config from appropriate source
    config = load_config()

    PROJECT_ID = config["project_id"]
    LOCATION = config["location"]
    INDEX_ID = config["index_id"]
    ENDPOINT_ID = config["endpoint_id"]
    GCS_BUCKET_NAME = config["bucket_name"]

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
                    
                    # Debug: Show what type of results we got
                    # if os.path.exists("config.json"):
                    #     # Local - don't show debug
                    #     pass
                    # else:
                    #     # Cloud - add debug info
                    #     st.info(f"Debug: Result type: {type(results[0])}")
                    #     if hasattr(results[0], '__dict__'):
                    #         st.info(f"Debug: Result attributes: {list(results[0].__dict__.keys())}")
                    
                    # Display results
                    for i, result in enumerate(results, 1):
                        # Handle different result formats
                        try:
                            # Try standard LangChain Document format first
                            content = getattr(result, 'page_content', None)
                            metadata = getattr(result, 'metadata', {})
                            
                            # Check if content is actually a JSON string (cloud issue)
                            if content and isinstance(content, str) and content.startswith('{"'):
                                try:
                                    # Parse the JSON content
                                    parsed_content = json.loads(content)
                                    # Extract the actual page content
                                    content = parsed_content.get('page_content', content)
                                    # Merge metadata from JSON if available
                                    if 'metadata' in parsed_content:
                                        metadata.update(parsed_content['metadata'])
                                except json.JSONDecodeError:
                                    # If JSON parsing fails, use original content
                                    pass
                            
                            # If that fails, try dict format
                            if content is None and isinstance(result, dict):
                                content = result.get('page_content', str(result))
                                metadata = result.get('metadata', {})
                            
                            # If still no content, convert to string
                            if content is None:
                                content = str(result)
                                metadata = {}
                                
                        except Exception as e:
                            st.error(f"Error parsing result {i}: {e}")
                            content = str(result)
                            metadata = {}
                        
                        element_type = metadata.get('element_type', 'Document')
                        with st.expander(f"📄 Result {i}: {element_type}", expanded=(i<=2)):
                            # Main content
                            st.markdown("**Summary:**")
                            st.write(content)
                            
                            # Metadata
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.markdown(f"**Source:** {metadata.get('source', 'Unknown')}")
                            with col2:
                                page_num = metadata.get('page_number')
                                st.markdown(f"**Page:** {page_num if page_num else 'Unknown'}")
                            with col3:
                                seq_id = metadata.get('sequence_id')
                                st.markdown(f"**Sequence:** {seq_id if seq_id is not None else 'Unknown'}")
                            
                            # Original text preview (no nested expander)
                            if metadata.get('original_text'):
                                st.markdown("**📖 Original Text Preview:**")
                                st.text(metadata['original_text'])
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