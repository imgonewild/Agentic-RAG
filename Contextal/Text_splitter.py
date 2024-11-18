from langchain.text_splitter import SpacyTextSplitter , RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import os

doc = PyPDFLoader("data2/" + os.listdir('data2')[0]).load()
splitter = RecursiveCharacterTextSplitter()
chunks = splitter.split_documents(documents=doc)

for i in chunks:
    print(i.page_content)
    print()
    print("next chunk")
    print()