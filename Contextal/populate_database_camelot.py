import argparse
import os
import shutil
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import camelot 
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_community.vectorstores import Chroma
from typing import List
import uuid

#CHROMA_PATH為Chroma存放資料庫的資料夾目錄
#DATA_PATH為儲存PDF檔案的資料夾目錄
CHROMA_PATH = "camelot"
DATA_PATH = "data2"

# Path to your Tesseract-OCR installation
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Perform OCR on an image.
def ocr_image(image):
    text = pytesseract.image_to_string(image)
    return text

# Extract text or images from a PDF and perform OCR if necessary.
def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    full_text = ""

    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text = page.get_text()
        if text.strip():
            full_text += text
        else:
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes()))
            text = ocr_image(img)
            full_text += text

    return full_text

# Extract tables using Camelot if the PDF is not image-based.
def extract_tables_from_pdf(pdf_path):
    extracted_documents = []
    try:
        # Extract tables using Camelot
        try:
            tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
            if len(tables) == 0:
                tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')
        except Exception as e:
            if 'Ghostscript' in str(e):
                print(f"Error extracting tables using 'lattice' from {pdf_path}: Ghostscript is not installed or not found. Please install Ghostscript and add it to your PATH.")
                tables = []  # Skip this file if Ghostscript is not available
            else:
                print(f"Error extracting tables using 'lattice' from {pdf_path}: {e}")
                tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')

        for i, table in enumerate(tables):
            table_text = table.df.to_json()
            metadata = {"source": pdf_path, "page": table.page, "type": "table", "table_index": i, "document_name": os.path.basename(pdf_path)}
            extracted_documents.append(Document(page_content=table_text, metadata=metadata))
    except Exception as e:
        print(f"Error extracting tables from {pdf_path}: {e}")
    return extracted_documents

# Check if a PDF is a scanned document.
def is_scanned_pdf(pdf_path):
    document = fitz.open(pdf_path)
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text = page.get_text()
        if text.strip():
            return False
    return True

# Main function to populate the database with documents.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    documents = load_documents()
    chunks = split_documents(documents)

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])

    missing_ids = find_missing_documents(existing_ids, chunks)
    
    if missing_ids:
        print(f"🗑️ Deleting missing documents from database: {missing_ids}")
        delete_from_chroma(missing_ids)
    else:
        print("✅ No documents to delete")

    add_to_chroma(chunks)

# Load documents from the specified data directory.
def load_documents():
    documents = []
    # Iterate through the files in the data directory and extract text from PDFs.
    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_file = os.path.join(root, file)
                if is_scanned_pdf(pdf_file):
                    print(f"{pdf_file} is a scanned PDF. Performing OCR.")
                    text = extract_text_from_pdf(pdf_file)
                else:
                    print(f"{pdf_file} is not a scanned PDF. Extracting text directly.")
                    document = fitz.open(pdf_file)
                    text = ""
                    for page_num in range(len(document)):
                        page = document.load_page(page_num)
                        text += page.get_text()

                # Extract tables using Camelot, if applicable
                tables = extract_tables_from_pdf(pdf_file)
                documents.extend(tables)

                doc = Document(page_content=text, metadata={"source": pdf_file, "document_name": os.path.basename(pdf_file)})
                documents.append(doc)
    return documents

# Split the documents into smaller chunks.
def split_documents(documents: list[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=600,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    chunks_with_ids = calculate_chunk_ids(chunks)
    return chunks_with_ids

# Find missing documents in the database.
def find_missing_documents(existing_ids: set, current_chunks: list[Document]) -> List[str]:
    current_ids = {chunk.metadata["id"] for chunk in current_chunks}
    missing_ids = list(existing_ids - current_ids)
    return missing_ids

# Add documents to the Chroma vector store.
def add_to_chroma(chunks: list[Document]):
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

    if new_chunks:
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")

# Delete documents from the Chroma vector store.
def delete_from_chroma(ids_to_delete: list[str]):
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only delete documents that exist in the DB.
    ids_to_delete = [doc_id for doc_id in ids_to_delete if doc_id in existing_ids]

    # Delete the documents.
    if ids_to_delete:
        print(f"🗑️ Deleting documents: {len(ids_to_delete)}")
        db.delete(ids=ids_to_delete)
        db.persist()
    else:
        print("✅ No documents to delete")

# Calculate unique IDs for the chunks.
def calculate_chunk_ids(chunks: List[Document]) -> List[Document]:
    for chunk in chunks:
        # Generate a unique ID for each chunk
        unique_id = str(uuid.uuid4())
        chunk.metadata["id"] = unique_id

    return chunks

# Clear the Chroma vector store database.
def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

if __name__ == "__main__":
    main()