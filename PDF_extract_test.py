from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.document_loaders import PDFMinerPDFasHTMLLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
import os
import re

def _clean_text(text):
        """ Clean the document text by removing unnecessary headers, footers, and formatting characters. """
        # Remove headers and footers using regex
        text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text)  # Remove "Page X of Y"
        # text = re.sub(r'_{10,}', '', text)  # Remove long sequences of underscores (formatting characters)
        
        # Remove multiple spaces, tabs, and newline characters
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize white spaces
        text = re.sub(r' +', ' ', text).strip()
        return text

dict_list = os.listdir("pdf/")
pdf_text = []
for pdf in dict_list:
    pdf_loader = UnstructuredPDFLoader(file_path="pdf/"+ pdf)
    pdf_text.append(pdf_loader.load())
print(_clean_text(pdf_text[0][0].page_content))
with open("test.html","w",encoding="utf_8_sig") as file:
      file.write(pdf_text[0][0].page_content)
