from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

def generate_image_summaries(all_images): 
    model = ChatOllama(temperature=0, model="gemma3:latest")
    for image in all_images: 
        if not image.metadata['image_tag']:
            image_tag = model.invoke([ SystemMessage(content = 'Generate a image summary in 30 words')
                                                        ,HumanMessage(content = [{'type': 'image_url', 
                                                                                  'image_url' : image.metadata['image_url']
                                                                                  }
                                                                                ]
                                                                      )
                                                        ]
                                                       ).content
            
            image.page_content += image_tag
    return all_images