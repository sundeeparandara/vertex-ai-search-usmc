You are building a production-ready multimodal Retrieval-Augmented Generation (RAG) system on Google Cloud.

## System Overview

Build a system that lets users chat with PDFs stored in Google Cloud Storage (GCS), including documents with images, tables, and graphs. The system should automatically index new PDFs as they are uploaded, and allow users to ask questions via a Streamlit frontend deployed on Cloud Run.

## Key Requirements

### 1. Streamlit Frontend (Cloud Run)
- Build a Streamlit app that runs on port 8501 and:
  - Accepts a user’s natural language query.
  - Retrieves relevant document chunks from a Vertex AI Vector Search index.
  - Sends the chunks + query to the Gemini model (via Vertex AI API).
  - Displays the AI-generated answer.
- Use LangChain for retrieval and Gemini chat integration.
- Use Vertex AI’s `text-embedding-005` for embeddings.
- Store the vector data in Vertex AI Matching Engine.
- No local file uploads — the app reads only from pre-indexed documents.

### 2. Auto-Indexer (Cloud Function, GCS Trigger)
- Build a Python Cloud Function that:
  - Triggers when a PDF is uploaded to a specific GCS bucket.
  - Downloads and parses the PDF using `unstructured`.
  - Extracts text, tables, and image descriptions.
  - Summarizes each part using Gemini.
  - Embeds the summaries with `text-embedding-005`.
  - Adds them to the Vertex AI vector index using `VectorSearchVectorStore`.

### 3. Folder Structure
Use this structure:
root/
├── streamlit_app/
│ ├── app.py
│ ├── rag_chain.py
│ ├── requirements.txt
│ └── Dockerfile
├── cloud_function/
│ ├── main.py
│ ├── requirements.txt
│ └── README.md
└── README.md


### 4. Deployment Targets
- The Streamlit app should be deployable directly from GitHub to Cloud Run (Docker-based).
- The Cloud Function should be deployable via `gcloud functions deploy` and triggered by `google.storage.object.finalize` events.

### 5. Environment Configuration
- Use environment variables for:
  - `GCP_PROJECT_ID`
  - `GCS_BUCKET`
  - `LOCATION`
  - `MODEL_NAME`
  - `INDEX_ID`
  - `ENDPOINT_ID`
- These can be set in Cloud Run and Cloud Functions via `--set-env-vars`.

### 6. Technologies
- Python 3.10+
- Streamlit
- LangChain
- Vertex AI APIs (Gemini + Embeddings + Matching Engine)
- Google Cloud Storage
- `unstructured` for document parsing

## Deliverables

- Full working code in the folder structure above.
- Working retrieval + QA pipeline using Gemini.
- Auto-indexing pipeline triggered by GCS uploads.
- README with setup and deployment instructions for both components.

This system should allow for scalable, event-driven document QA using Google Cloud’s hosted AI stack.
