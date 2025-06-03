import pickle
import json
from langchain.schema import Document
from langchain_google_vertexai import VectorSearchVectorStore
from langchain_google_vertexai import VertexAIEmbeddings
import vertexai
from vertexai.generative_models import GenerativeModel

from google.cloud import aiplatform
from unstructured.documents.elements import CompositeElement, Title, NarrativeText, Text

# Load config
with open("config.json", "r") as f:
    config = json.load(f)

PROJECT_ID = config["project_id"]
LOCATION = config["location"]
INDEX_ID = config["index_id"]
ENDPOINT_ID = config["endpoint_id"]
DEPLOYED_INDEX_ID = config["deployed_index_id"]

# You'll need to create a GCS bucket for storing vector data
# Replace this with your actual bucket name
GCS_BUCKET_NAME = "usmc_bucket_test"  # Using your existing bucket - we won't read from it, just need it for API requirement

# Load partitioned elements from previous step
PKL_PATH = "./partitioned_output.pkl"
with open(PKL_PATH, "rb") as f:
    elements = pickle.load(f)

print(f"📦 Loaded {len(elements)} elements from {PKL_PATH}")

# Init Vertex AI
aiplatform.init(project=PROJECT_ID, location=LOCATION)

# Init Gemini (Text Generation Model)
# Using the newer Gemini API instead of the deprecated TextGenerationModel
# The GenerativeModel class is the correct way to access Gemini models
# For available models, see: https://cloud.google.com/vertex-ai/generative-ai/docs/models
# Pricing: https://ai.google.dev/gemini-api/docs/pricing

# Using Gemini 2.0 Flash Lite - optimized for cost efficiency and low latency
gemini = GenerativeModel("gemini-2.0-flash-lite")
#This instance is used to generate summaries of the document chunks. 
#It's doing the creative work of understanding the text and creating new content (summaries) while preserving important military concepts.

# Init Embeddings
embedder = VertexAIEmbeddings(
    model_name="text-embedding-005",  # Latest stable embedding model supported by LangChain, dimensions=768
    project=PROJECT_ID,
    location=LOCATION
)
#This instance is used to convert text into vector representations. 
# It's not generating new text - instead, it's converting text into numerical vectors that capture the semantic meaning of the text for search purposes.
#The VertexAIEmbeddings class from LangChain is a wrapper that handles the interaction with Vertex AI's embedding service.

# Init Vector Store
# Using your existing bucket name - this is just required by the API
# All document content comes from your pkl file, not from the bucket
vector_store = VectorSearchVectorStore.from_components(
    project_id=PROJECT_ID,
    region=LOCATION,
    gcs_bucket_name=GCS_BUCKET_NAME,  # Your existing bucket - we won't read from it
    index_id=INDEX_ID,
    endpoint_id=ENDPOINT_ID,
    embedding=embedder,
    stream_update=True  # Enable streaming updates to match your index configuration
)

# Note: All document content comes from your pkl file and gets stored in the 
# Document.page_content field. The bucket is just an API requirement.

# Process and store
documents_to_upload = []
SOURCE_NAME = "MCDP_1.pdf"

for i, el in enumerate(elements):
    # Debug: print first few elements to see what we're working with
    if i < 10:
        print(f"Element {i}: {type(el).__name__} - Has text: {hasattr(el, 'text')}")
        print(f"   isinstance(CompositeElement): {isinstance(el, CompositeElement)}")
        print(f"   MRO: {[cls.__name__ for cls in type(el).__mro__]}")
        if hasattr(el, 'text'):
            print(f"   Text preview: {el.text[:100] if el.text else 'No text'}...")
    
    # Instead of checking CompositeElement, check for text-based elements directly
    # or check if it has text and is not an Image (which shouldn't be processed)
    if not hasattr(el, 'text') or not el.text.strip():
        continue  # Skip elements without text content
        
    # Skip Image elements even if they have text
    if type(el).__name__ == 'Image':
        continue
        
    print(f"✅ Processing element {i}: {type(el).__name__}")

    # Get the text from the previous element if it exists and has a text attribute
    if i > 0 and hasattr(elements[i - 1], 'text'):
        prev_text = elements[i - 1].text
    else:
        prev_text = ""

    # Get the text from the next element if it exists and has a text attribute
    if i < len(elements) - 1 and hasattr(elements[i + 1], 'text'):
        next_text = elements[i + 1].text
    else:
        next_text = ""

    # Combine the previous text, current text, and next text with double newlines between them
    combined_context = f"{prev_text}\n\n{el.text}\n\n{next_text}"

    # Summarize with Gemini
    prompt = (
        "Summarize the central idea of the following text for search purposes. "
        "Preserve important military concepts and definitions.\n\n"
        f"{combined_context}"
    )
    # Using the new Gemini API syntax
    response = gemini.generate_content(prompt)
    summary = response.text.strip()

    # Create LangChain Document
    doc = Document(
        page_content=summary,
        metadata={
            "source": SOURCE_NAME,
            "original_text": el.text[:300],  # preview
            "sequence_id": i,
            "page_number": el.metadata.page_number if hasattr(el.metadata, "page_number") else None,
            "element_type": type(el).__name__,
        }
    )
    #This creates a LangChain Document object that contains the summary and metadata.
    #The Document class is a core class in LangChain that represents a single document with its content and metadata.
    documents_to_upload.append(doc)

    # Optional: print preview  
    print(f"\n🧾 Chunk {i} — Summary:")
    print(summary[:250])
    
    # # Limit to first few for testing
    # if len(documents_to_upload) >= 3:
    #     print("🔍 Stopping after 3 documents for testing...")
    #     break

# Embed and store
print(f"\n🚀 Storing {len(documents_to_upload)} documents to vector index...")
vector_store.add_documents(documents_to_upload)
print("✅ Upload complete.")