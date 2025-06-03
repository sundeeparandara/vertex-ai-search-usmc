# Tutorial Sources:
* [Grounding for Gemini with Vertex AI Search and DIY RAG](https://youtu.be/v4s5eU2tfd4?si=tAOG29xhoMX-63Wv)
* [Github Example from GCP](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/use-cases/retrieval-augmented-generation/multimodal_rag_langchain.ipynb!)

# USMC Documentation Sources:
* [MCDP](https://www.marines.mil/News/Publications/MCPEL/Custompubtype/2001/?Page=17!)

# Steps

1. Parition the pdf ("pdf_partitioner.py"). The library "unstructured" is used and this chunks the information using its internal algo to semanitcally meaningful chunks. Results are stored on the pkl file "partitioned_output.pkl", so that this algo does not have been run again (it takes a long time).
2. (optional) Read the pkl file. See "pkl_reader.ipynb".
3. Create the index / vector database in GCP (called "Matching Engine"), make it accessible / make the endpoint and identify it with a label. You making the "cabinet" here and "grabbing the keys to it", you still have not loaded any information into it.
    
    3.1 Matching Engine reading material:
        
    * https://cloud.google.com/blog/products/ai-machine-learning/vertex-matching-engine-blazing-fast-and-massively-scalable-nearest-neighbor-search
        
    * https://js.langchain.com/docs/integrations/vectorstores/googlevertexai/

