# Tutorial Sources:
* [Grounding for Gemini with Vertex AI Search and DIY RAG](https://youtu.be/v4s5eU2tfd4?si=tAOG29xhoMX-63Wv)
* [Github Example from GCP](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/use-cases/retrieval-augmented-generation/multimodal_rag_langchain.ipynb![image](https://github.com/user-attachments/assets/b6d163d6-b41f-4058-b4eb-bb4dd4a43384)

# USMC Documentation Sources:
* [MCDP](https://www.marines.mil/News/Publications/MCPEL/Custompubtype/2001/?Page=17![image](https://github.com/user-attachments/assets/f99148dd-588d-485b-b92b-d2ffff83a91c)


# Steps

1. Parition the pdf ("pdf_partitioner.py"). The library "unstructured" is used and this chunks the information using its internal algo to semanitcally meaningful chunks. Results are stored on the pkl file "partitioned_output.pkl", so that this algo does not have been run again (it takes a long time).
2. (optional) Read the pkl file. See "pkl_reader.ipynb".