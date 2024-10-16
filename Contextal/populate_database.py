import argparse
import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_community.vectorstores import Chroma
from typing import List
from tqdm import tqdm
from langchain_community.llms.ollama import Ollama

CHROMA_PATH = "chroma"
DATA_PATH = "data2"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    documents = load_documents()
    chunks = split_documents(documents)

    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])

    missing_ids = find_missing_documents(existing_ids, chunks)
    
    if missing_ids:
        print(f"🗑️ Deleting missing documents from database: {missing_ids}")
        delete_from_chroma(missing_ids)
    else:
        print("✅ No documents to delete")

    add_to_chroma(chunks)

def load_documents():
    print("Loading documents...")
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

def split_documents(documents: list[Document]) -> List[Document]:
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=250,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    
    print("Adding context to chunks...")
    contextual_chunks = add_context_to_chunks(chunks, documents)
    
    print("Calculating chunk IDs...")
    chunks_with_ids = calculate_chunk_ids(contextual_chunks)
    return chunks_with_ids

def add_context_to_chunks(chunks: List[Document], full_documents: List[Document]) -> List[Document]:
    model = Ollama(model="llama3.1")
    
    contextual_chunks = []
    for chunk in tqdm(chunks, desc="Processing chunks", unit="chunk"):
        # Find the full document that this chunk belongs to
        full_doc = next(doc for doc in full_documents if doc.metadata['source'] == chunk.metadata['source'])
        
        prompt = f"""<document>
                {full_doc.page_content}
                </document>

                Here is the chunk we want to situate within the whole document:
                <chunk>
                {chunk.page_content}
                </chunk>

                Please give a short succinct context to situate this chunk within the overall document for the purposes of improving 
                search retrieval of the chunk. Answer only with the succinct context and nothing else."""
        
        context = model(prompt)
        
        # Create a new Document with the original content plus the context
        contextual_chunk = Document(
            page_content=f"Context: {context}\n\nContent: {chunk.page_content}",
            metadata=chunk.metadata
        )
        contextual_chunks.append(contextual_chunk)
    
    return contextual_chunks

def find_missing_documents(existing_ids: set, current_chunks: list[Document]) -> List[str]:
    current_ids = {chunk.metadata["id"] for chunk in current_chunks}
    missing_ids = list(existing_ids - current_ids)
    return missing_ids

def add_to_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

    if new_chunks:
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        for i in tqdm(range(0, len(new_chunks), 100), desc="Adding to database", unit="batch"):
            batch = new_chunks[i:i+100]
            batch_ids = new_chunk_ids[i:i+100]
            db.add_documents(batch, ids=batch_ids)
        db.persist()
    else:
        print("✅ No new documents to add")

def delete_from_chroma(ids_to_delete: list[str]):
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    ids_to_delete = [doc_id for doc_id in ids_to_delete if doc_id in existing_ids]

    if ids_to_delete:
        print(f"🗑️ Deleting documents: {len(ids_to_delete)}")
        db.delete(ids=ids_to_delete)
        db.persist()
    else:
        print("✅ No documents to delete")

def calculate_chunk_ids(chunks: List[Document]) -> List[Document]:
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

if __name__ == "__main__":
    main()