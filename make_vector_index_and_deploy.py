"""
🧠 PURPOSE OF THIS SCRIPT

This script creates a Vertex AI Matching Engine index and deploys it to an endpoint.
This index will store vector embeddings of your document summaries for use in RAG-based search.

You only need to run this ONCE to set up the "filing cabinet" (index + endpoint).

---

🔐 AUTHENTICATION REQUIREMENT: Application Default Credentials (ADC)

To run this script, your local environment must be authenticated with Google Cloud using ADC.

✅ STEPS TO SET UP (ONE TIME):

1. Install the Google Cloud SDK if you haven't already:
   https://cloud.google.com/sdk/docs/install

2. Authenticate your local machine using the following command:
   gcloud auth application-default login

   - This will open a browser
   - Log in with your GCP account
   - It will store credentials that your Python code can access

3. Ensure your active project is set correctly (optional but recommended):
   gcloud config set project your-gcp-project-id

Once this is done, this script will be able to authenticate with Google Cloud services.

📌 For more info on ADC:
https://cloud.google.com/docs/authentication/application-default-credentials

---

Some useful commands to run in Google Cloud SDK Shell, after you've installed the SDK and authenticated:
gcloud config list
gcloud config get-value project
gcloud ai indexes list

---

🧠 WHY THIS MATTERS

Before you can store any document embeddings or perform semantic search,
you need to set up the infrastructure — just like creating a blank database schema.

Once this is deployed, your RAG pipeline can:
- ✅ Add new documents (as summaries + vectors)
- ✅ Search the index for relevant content
- ✅ Use metadata to trace matches back to source pages

---

📦 KEY OUTPUTS

- `INDEX_ID`: The full path to your vector index in Vertex AI
- `ENDPOINT_ID`: The API endpoint used to store/query vectors
- `DEPLOYED_INDEX_ID`: Your custom label to refer to the index when querying

You can reuse this infrastructure indefinitely — you **do not** need to recreate it for each document.

---

📦 KEY OUTPUTS

- `INDEX_ID`: The full path to your vector index in Vertex AI
- `ENDPOINT_ID`: The API endpoint used to store/query vectors
- `DEPLOYED_INDEX_ID`: Your custom label to refer to the index when querying

These values are printed when the script runs — save them to `config.json` or environment variables for reuse.

---

📍 WHERE TO FIND THESE IN GCP CONSOLE

1. Go to: https://console.cloud.google.com/vertex-ai
2. In the left sidebar:
   - Expand **Agent Builder → Vector Search**
   - Click on **Indexes** to see your vector indexes
   - Click on **Index Endpoints** to see your deployed endpoints
3. Inside each index, you'll see:
   - Metadata (dimensions, shard size, etc.)
   - Linked deployed indexes and endpoints
   - Dense vector count (how many vectors have been added so far)

"""


from google.cloud import aiplatform
import json

# Load configuration from config.json (recommended)
with open('config.json', 'r') as f:
    config = json.load(f)

PROJECT_ID = config['project_id']
LOCATION = config['location']
INDEX_DISPLAY_NAME = config['index_display_name']
ENDPOINT_DISPLAY_NAME = config['endpoint_display_name']
DEPLOYED_INDEX_ID = config['deployed_index_id']

DIMENSIONS = 768  # Required for text-embedding-005 output

# Tree-AH indexing parameters
anc = 150  # Approximate neighbors count
lnec = 500  # Leaf node embedding count
lntsp = 7   # Leaf nodes to search percent
ium = "STREAM_UPDATE"  # Live updates possible
desc = "Multimodal RAG index for USMC project"

def main():
    print("🔧 Initializing Vertex AI...")
    aiplatform.init(project=PROJECT_ID, location=LOCATION)

     # Step 1: Create the index
    print("🧠 Creating Matching Engine index...")
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

     # Step 2: Create the endpoint
    print("🌐 Creating index endpoint...")
    endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
        display_name=ENDPOINT_DISPLAY_NAME,
        public_endpoint_enabled=True,
    )
    print(f"✅ Endpoint created: {endpoint.resource_name}")

        # Step 3: Deploy the index to the endpoint
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