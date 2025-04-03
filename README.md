Please look for conclusion reports as well.(pptx presentation)
main.py = YouTube_OpenAIChatGPT4o_Project, A chatbot to answer the question on medical/surgical animated youtube videos. 
Medical_Insight_AI_Chatbot = This is Phase 1, input to the AI is Lab Report of a patient.(tabular and text data)
Phase 2: Added Multimodal Support (X-ray/Scan Analysis) with LangChain
integrate image analysis into the system to process X-rays, MRI scans, and surgical images. This will allow the AI to analyze both textual reports and medical images for better insights.

 1. Approach for Multimodal RAG
 Extract text from reports (Phase 1 - already built)
 Analyze images (X-ray, MRI, surgical scans) using CNN & Vision Transformer (ViT)
 Store image embeddings in FAISS for retrieval
 Use LangChain to combine text + image insights
 Query both text and image data to generate medical insights
