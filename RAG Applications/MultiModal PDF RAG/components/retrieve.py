def retrieve(query, similarity_retriever, bm25_retriever): 
    similarity_response = similarity_retriever.invoke(query) 
    bm25_response = bm25_retriever.invoke(query)
    
    response = similarity_response + bm25_response 
    
    return response