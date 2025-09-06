from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def generate_response(query, model, retrieved_documents):
    text, tables, images = '', '', ''  
    text = ''.join([document.page_content for document in retrieved_documents if document.metadata['type'] == 'text'])
    tables = '\n\n'.join([document for document in retrieved_documents if document.metadata['type'] == 'table'])
    images = '\n\n'.join([document.page_content + '\n' + document.metadata['image_url'] for document in retrieved_documents if document.metadata['type'] == 'image'])
    
    prompt = ChatPromptTemplate([
        (   
            'system', '''Answer user Question based on retrieved documents. 
                      Retrieved documents can be images, text or tables.
                      Answer the question in 100 words.
                      If you used an image to answer the question then say which image you used
                      '''
        ),
        
        (   
            'human', '''user_question: {query}\n\n 
                        text: \n\n{text}\n\n
                        tables: {tables}\n\n 
                        images: {images}
                     '''
        )
    ])
    
    chain = prompt | model | StrOutputParser() 
    
    response = chain.invoke({'query': query, 
                  'text': text, 
                  'tables': tables, 
                  'images': images})
    
    return response