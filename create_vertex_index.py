"""
📦 PURPOSE OF THIS SCRIPT

This script sets up the foundation for our document-based AI system.
You only need to run it ONCE.

It does 3 main things:
1. Creates a 'vector index' – this is like an empty digital filing cabinet 
   where we will store the AI-readable summaries (embeddings) of our documents.
2. Creates an 'index endpoint' – this gives us a way to talk to that cabinet 
   from other apps like Cloud Functions and Streamlit.
3. Connects the index to the endpoint – so it’s live and ready to use.

After this script runs:
✅ The index and endpoint will stay active in Google Cloud.
✅ You can keep adding more documents to it over time.
✅ Other apps (like the chat app) can search it to answer questions.

You do NOT need to run this again unless you're starting from scratch.

Next step: Set up a Cloud Function to automatically add new PDFs to this index.
"""



from google.cloud import aiplatform
import json

# Load configuration from keys.json
with open('config.json', 'r') as f:
    config = json.load(f)

PROJECT_ID = config['project_id']
LOCATION = config['location']

INDEX_DISPLAY_NAME = config['index_display_name']
ENDPOINT_DISPLAY_NAME = config['endpoint_display_name']
DEPLOYED_INDEX_ID = config['deployed_index_id']

DIMENSIONS = 768  # For text-embedding-005
#TREE-AH
anc=150 #approximate_neighbors_count
lnec=500 #leaf_node_embedding_count    
lntsp=7 #leaf_nodes_to_search_percent
ium="STREAM_UPDATE" #index_update_method
desc="Multimodal RAG index for USMC project"

def main():
    print("🔧 Initializing Vertex AI...")
    aiplatform.init(project=PROJECT_ID, location=LOCATION)

    # Step 1: Create Index
    print("🧠 Creating Matching Engine Index...")
    index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
        display_name=INDEX_DISPLAY_NAME,
        dimensions=DIMENSIONS,
        approximate_neighbors_count=anc,
        leaf_node_embedding_count=lnec,
        leaf_nodes_to_search_percent=lntsp,
        index_update_method=ium,
        description=desc,
    )
    print(f"✅ Index created: {index.resource_name}")

    # Step 2: Create Index Endpoint
    print("🌐 Creating Index Endpoint...")
    endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
        display_name=ENDPOINT_DISPLAY_NAME,
        public_endpoint_enabled=True,
    )
    print(f"✅ Endpoint created: {endpoint.resource_name}")

    # Step 3: Deploy Index to Endpoint
    print("🚀 Deploying index to endpoint...")
    endpoint.deploy_index(
        index=index,
        deployed_index_id=DEPLOYED_INDEX_ID,
    )
    print("✅ Deployment complete!")
    print("\n🔗 FINAL RESOURCE IDS:")
    print(f"INDEX_ID = '{index.resource_name}'")
    print(f"ENDPOINT_ID = '{endpoint.resource_name}'")
    print(f"DEPLOYED_INDEX_ID = '{DEPLOYED_INDEX_ID}'")

if __name__ == "__main__":
    main()