# Langchain_RAG_app
 
Retrival augumented generation :

Retrieval-Augmented Generation (RAG) is the process of optimizing the output of a large language model, so it references an authoritative knowledge base outside of its training data sources before generating a response.  

Which means we can use the power LLMs to make it do something outside it's training knowledege.

In this exmaple we use GEMINI pro as think tank , FAISS as vector store to generate and store embeddings.

User query (based on the pdf) is taken as input and we retive the embeddings from the vector store and use it as context for the LLM to generate the answer.

This approach helps in improving the accuracy and relevance of the generated response.
