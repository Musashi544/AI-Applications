import fitz
import pandas as pd
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PIL import Image
from io import BytesIO

recursive_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, 
                                                    chunk_overlap = 200)
def extract_pdf_elements(path): 
    all_text = [] 
    all_tables = [] 
    all_images = [] 
    image_data_store = {}
    image_tag = ''
    doc = fitz.open(path)
    
    for page_no, page in enumerate(doc): 
        # Extracting Text
        # Convert text in each page to langchain Document, then perform Recursive Character Text Splitting
        text = page.get_text()
        text =  Document(page_content = text, 
                                 metadata = {'type': 'text',
                                             'page': page_no + 1})
        
        text = recursive_splitter.split_documents([text])
        all_text.extend(text)
        
        # Extracting Tables
        tabs = page.find_tables()
        if tabs.tables:
            # tabs.tables is a list of Table objects
            for tab in tabs.tables: 
                table = tab.extract()
                df = pd.DataFrame(table[1:], columns = table[0])
                all_tables.append(Document(page_content = df.to_markdown(),
                                           metadata = {'page': page_no, 
                                                       'type': 'table'}))
                
        # Extracting Images 
        for image_no, image in enumerate(page.get_images(full = True)): 
            blocks = page.get_text('blocks')
            bbox = page.get_image_bbox(image)
            for b in blocks: 
                # Gets the x0, y0, x1, y1 from the block 
                rect = fitz.Rect(b[:4])  
                text = b[4]
                # the text block just below the image
                if rect.y0 > bbox.y1 and "Figure" in text:
                    image_tag = text
                    break
                    
            xref = image[0] 
            base_image = doc.extract_image(xref) 
            image_bytes = base_image['image'] 
            image_ext = base_image['ext'] 
            pil_image = Image.open(BytesIO(image_bytes)).convert('RGB')
            image_url = f'image_page_{page_no + 1}.{image_ext}'
            pil_image.save(image_url, format = image_ext)

            image_id = f"page_{page_no}_img_{image_no}"
            
            image_data_store[image_id] = pil_image 

            all_images.append(Document(page_content = f'Image in page {page_no + 1}\n' + image_tag, 
                                       metadata = {'page': page_no + 1, 
                                                   'type': 'image',  
                                                   'image_tag': image_tag,
                                                   'image_url': image_url}))
            
    return all_text, all_tables, all_images, image_data_store