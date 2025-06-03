import json
from google.cloud import aiplatform
from langchain_google_vertexai import VectorSearchVectorStore, VertexAIEmbeddings

def verify_vector_index():
    """Determine the actual matrix size (vectors × dimensions) in the Vector Search index"""
    
    # Load config
    with open("config.json", "r") as f:
        config = json.load(f)

    PROJECT_ID = config["project_id"]
    LOCATION = config["location"]
    INDEX_ID = config["index_id"]
    ENDPOINT_ID = config["endpoint_id"]
    GCS_BUCKET_NAME = "usmc_bucket_test"

    print("🔍 Determining Vector Matrix Size...")
    print("-" * 40)

    # Initialize Vertex AI
    aiplatform.init(project=PROJECT_ID, location=LOCATION)

    # Initialize embeddings and vector store
    embedder = VertexAIEmbeddings(
        model_name="text-embedding-005",
        project=PROJECT_ID,
        location=LOCATION
    )

    vector_store = VectorSearchVectorStore.from_components(
        project_id=PROJECT_ID,
        region=LOCATION,
        gcs_bucket_name=GCS_BUCKET_NAME,
        index_id=INDEX_ID,
        endpoint_id=ENDPOINT_ID,
        embedding=embedder,
        stream_update=True
    )

    # Determine vector dimensions
    sample_text = "test embedding dimensions"
    embedding = embedder.embed_query(sample_text)
    dimensions = len(embedding)

    # Estimate vector count through sequence ID sampling
    results = vector_store.similarity_search("military doctrine leadership", k=100)
    sequence_ids = [result.metadata.get('sequence_id') for result in results if result.metadata.get('sequence_id') is not None]
    
    if sequence_ids:
        max_sequence_id = max(sequence_ids)
        estimated_vector_count = max_sequence_id + 1  # sequence_id starts from 0
    else:
        # Fallback: sample multiple queries to estimate count
        test_queries = ["leadership", "military", "doctrine", "operations", "strategy"]
        all_unique_ids = set()
        
        for query in test_queries:
            results = vector_store.similarity_search(query, k=50)
            for result in results:
                unique_id = result.metadata.get('sequence_id', f"unknown_{hash(result.page_content[:50])}")
                all_unique_ids.add(unique_id)
        
        estimated_vector_count = len(all_unique_ids)

    # Output matrix size
    print("=" * 40)
    print("VECTOR MATRIX SIZE:")
    print("=" * 40)
    print(f"Estimated Vectors: {estimated_vector_count}")
    print(f"Dimensions per Vector: {dimensions}")
    print(f"Matrix Size: {estimated_vector_count} × {dimensions}")
    print("=" * 40)

if __name__ == "__main__":
    verify_vector_index() 