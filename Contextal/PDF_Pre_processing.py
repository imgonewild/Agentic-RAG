import os
import camelot
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import SpacyTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
import winsound

pre_process_model = 'llama3.2:1b-instruct-q8_0'
embedding_model = "nomic-embed-text"
document_dir = 'data2'
text_vectorDB_dir = 'text_DB'
table_vectorDB_dir = 'table_DB'

SYS_PROMPT = """
You are a world class text pre-processor, here is the raw data from a PDF, please parse and return it in a way that is crispy and usable to send to a podcast writer.

The raw data is messed up with new lines, Latex math and you will see fluff that we can remove completely. Basically take away any details that you think might be useless in a podcast author's transcript.

Remember, the podcast could be on any topic whatsoever so the issues listed above are not exhaustive

Please be smart with what you remove and be creative ok?

Remember DO NOT START SUMMARIZING THIS, YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED

Be very smart and aggressive with removing details, you will get a running portion of the text and keep returning the processed text.

PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPATILISATION ETC LIKES

ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions ok?
Here is the text:

"""

def extract_document(File_dir):
    doc = PyPDFLoader(File_dir + '/' + os.listdir(File_dir)[0]).load()
    return doc

def extract_table(File_dir):
    tables = camelot.read_pdf(File_dir)
    return tables

def process_document(document,LLM,File_dir):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 200
        )
    
    chunks = splitter.split_documents(documents=document)
    
    processed_doc = []
    for chunk in chunks:
        tmp = Ollama(
            model=LLM,
            temperature=0.7,
            top_p=0.9
            ).invoke(SYS_PROMPT + chunk.page_content)
        print(tmp)
        tmp_document = Document(page_content=tmp)
        tmp_document.metadata = chunk.metadata
        tmp_document.metadata["document_name"] = os.listdir(File_dir)[0]
        processed_doc.append(tmp_document)
    print(f"processed the document. Get {len(processed_doc)} documents")
    return processed_doc

def split_text(doc):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 200
    )

    chunks = splitter.split_documents(doc)
    print(f"Split the text. Get {len(chunks)} documents")
    return chunks

def insert_document(doc,vector_dir,embedding_model):
    print("insert data to database")
    DB = Chroma(persist_directory=vector_dir,embedding_function=OllamaEmbeddings(model=embedding_model))
    DB.add_documents(doc)

def main():
    print("start processing doucment")

    result = process_document(extract_document(document_dir),pre_process_model,document_dir)
    print(result)
    insert_document(result,text_vectorDB_dir,embedding_model)   
    
main()

winsound.Beep(1200,1000)
winsound.Beep(1300,1000)
winsound.Beep(1400,1000)