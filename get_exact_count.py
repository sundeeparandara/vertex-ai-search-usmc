import json
from google.cloud import aiplatform_v1
from langchain_google_vertexai import VertexAIEmbeddings

def get_exact_vector_count():
    """Get exact vector count and dimensions using Google Cloud APIs"""
    
    # Load config
    with open("config.json", "r") as f:
        config = json.load(f)

    PROJECT_ID = config["project_id"]
    LOCATION = config["location"]
    INDEX_ID = config["index_id"]

    print("🔍 Getting Exact Vector Matrix Size...")
    print(f"Project: {PROJECT_ID}")
    print(f"Location: {LOCATION}")
    print("-" * 40)

    # Get exact vector count from Index API
    client_options = {"api_endpoint": f"{LOCATION}-aiplatform.googleapis.com"}
    client = aiplatform_v1.IndexServiceClient(client_options=client_options)
    
    index_name = f"projects/{PROJECT_ID}/locations/{LOCATION}/indexes/{INDEX_ID.split('/')[-1]}"
    print(f"Querying index: {index_name}")
    
    # Get index details
    index = client.get_index(name=index_name)
    print(f"Index display name: {index.display_name}")
    
    # Get exact vector count
    if hasattr(index, 'index_stats') and index.index_stats:
        vector_count = index.index_stats.vectors_count
        print(f"✅ EXACT Vector Count: {vector_count}")
    else:
        print("❌ No index_stats available")
        return

    # Get actual vector dimensions by testing embeddings
    print("\n🔍 Testing actual vector dimensions...")
    embedder = VertexAIEmbeddings(
        model_name="text-embedding-005",
        project=PROJECT_ID,
        location=LOCATION
    )
    
    # Test embedding to get actual dimensions
    sample_text = "test vector dimensions"
    embedding = embedder.embed_query(sample_text)
    actual_dimensions = len(embedding)
    print(f"✅ ACTUAL Vector Dimensions: {actual_dimensions}")
    
    # Output exact matrix size
    print("\n" + "=" * 40)
    print("EXACT VECTOR MATRIX SIZE:")
    print("=" * 40)
    print(f"Vectors: {vector_count}")
    print(f"Dimensions: {actual_dimensions}")
    print(f"Matrix Size: {vector_count} × {actual_dimensions}")
    print("=" * 40)

if __name__ == "__main__":
    get_exact_vector_count() 