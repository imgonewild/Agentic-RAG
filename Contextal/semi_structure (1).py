from typing import Any
from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.llms.ollama import Ollama
from langchain_core.runnables import RunnablePassthrough
from get_embedding_function import get_embedding_function
import uuid
import time
import pandas as pd
import os
import shutil
import glob
from pathlib import Path
import pickle

def get_pdf_files(folder_path: str) -> list:
    """Get all PDF files from the specified folder"""
    pdf_pattern = os.path.join(folder_path, '*.[pP][dD][fF]')
    pdf_files = glob.glob(pdf_pattern)
    
    if not pdf_files:
        raise ValueError(f"No PDF files found in {folder_path}")
    
    print(f"Found {len(pdf_files)} PDF files:")
    for pdf in pdf_files:
        print(f"- {os.path.basename(pdf)}")
    
    return pdf_files

def list_processed_pdfs(persist_dir: str) -> list:
    """List PDFs that have already been processed"""
    try:
        with open(os.path.join(persist_dir, 'docstore.pkl'), 'rb') as f:
            all_doc_data = pickle.load(f)
            return list(all_doc_data.keys())
    except:
        return []

def load_pdf_elements(file_path: str):
    return partition_pdf(
        filename=file_path,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
    )

class Element(BaseModel):
    type: str
    text: Any

def categorize_elements(raw_pdf_elements):
    categorized_elements = []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            categorized_elements.append(Element(type="table", text=str(element)))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            categorized_elements.append(Element(type="text", text=str(element)))
    return categorized_elements

def summarize_elements(text_elements, table_elements):
    prompt_text = """You are an assistant tasked with summarizing tables and text.
    Give a concise summary of the table or text. Table or text chunk: {element}"""
    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = Ollama(temperature=0, model="llama3.1")
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
    
    text_summaries = summarize_chain.batch([e.text for e in text_elements], {"max_concurrency": 5})
    table_summaries = summarize_chain.batch([e.text for e in table_elements], {"max_concurrency": 5})
    
    return text_summaries, table_summaries

def is_valid_vectorstore(persist_dir: str) -> bool:
    """Check if the vectorstore directory contains all necessary files"""
    required_files = ['docstore.pkl']
    chroma_files = ['chroma.sqlite3']
    
    if not os.path.exists(os.path.join(persist_dir, 'docstore.pkl')):
        return False
    
    if not all(os.path.exists(os.path.join(persist_dir, f)) for f in chroma_files):
        return False
    
    return True

def create_retriever(persist_dir: str, pdf_files: list):
    """Create a new retriever and process PDF files"""
    
    vectorstore = Chroma(
        collection_name="summaries",
        embedding_function=get_embedding_function(),
        persist_directory=persist_dir
    )
    
    store = InMemoryStore()
    id_key = "doc_id"
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key
    )
    
    all_doc_data = {}
    # Load existing data if any
    if os.path.exists(os.path.join(persist_dir, 'docstore.pkl')):
        with open(os.path.join(persist_dir, 'docstore.pkl'), 'rb') as f:
            all_doc_data = pickle.load(f)
    
    for pdf_file in pdf_files:
        # Normalize path for storage
        normalized_path = os.path.normpath(pdf_file)
        print(f"Processing {normalized_path}...")
        raw_elements = load_pdf_elements(pdf_file)
        
        categorized_elements = categorize_elements(raw_elements)
        text_elements = [e for e in categorized_elements if e.type == "text"]
        table_elements = [e for e in categorized_elements if e.type == "table"]
        
        text_summaries, table_summaries = summarize_elements(text_elements, table_elements)
        texts = [e.text for e in text_elements]
        tables = [e.text for e in table_elements]
        
        # Add text summaries and raw text
        doc_ids = [str(uuid.uuid4()) for _ in texts]
        summary_texts = [
            Document(
                page_content=s, 
                metadata={
                    id_key: doc_ids[i],
                    "source": normalized_path,
                    "type": "text"
                }
            )
            for i, s in enumerate(text_summaries)
        ]
        retriever.vectorstore.add_documents(summary_texts)
        retriever.docstore.mset(list(zip(doc_ids, texts)))
        
        # Add table summaries and raw tables if they exist
        table_ids = []
        if table_summaries and tables and len(tables) > 0:
            table_ids = [str(uuid.uuid4()) for _ in tables]
            summary_tables = [
                Document(
                    page_content=s, 
                    metadata={
                        id_key: table_ids[i],
                        "source": normalized_path,
                        "type": "table"
                    }
                )
                for i, s in enumerate(table_summaries)
            ]
            retriever.vectorstore.add_documents(summary_tables)
            retriever.docstore.mset(list(zip(table_ids, tables)))
        
        # Store document data using normalized path
        all_doc_data[normalized_path] = {
            'texts': list(zip(doc_ids, texts)),
            'tables': list(zip(table_ids, tables)) if tables and len(tables) > 0 else []
        }
    
    # Save all document data to disk
    with open(os.path.join(persist_dir, 'docstore.pkl'), 'wb') as f:
        pickle.dump(all_doc_data, f)
    
    return retriever

def load_retriever(persist_dir: str, target_pdf_file: str):
    """Load an existing retriever from disk for a specific PDF"""
    
    vectorstore = Chroma(
        collection_name="summaries",
        embedding_function=get_embedding_function(),
        persist_directory=persist_dir
    )
    
    with open(os.path.join(persist_dir, 'docstore.pkl'), 'rb') as f:
        all_doc_data = pickle.load(f)
    
    # Normalize the target path
    normalized_target = os.path.normpath(target_pdf_file)
    print(f"Looking for PDF: {normalized_target}")
    print(f"Available PDFs: {list(all_doc_data.keys())}")
    
    store = InMemoryStore()
    
    # Only load data for the target PDF
    if normalized_target in all_doc_data:
        doc_data = all_doc_data[normalized_target]
        store.mset(doc_data['texts'])
        if doc_data['tables']:
            store.mset(doc_data['tables'])
    else:
        raise ValueError(f"PDF file {normalized_target} not found in the vectorstore")

    print(f"Successfully loaded data for PDF: {os.path.basename(normalized_target)}")
    
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key="doc_id",
        search_kwargs={'filter': {'source': normalized_target}}
    )
    
    return retriever

def run_rag_pipeline(persist_dir: str, question, pdf_file=None):
    """Run RAG pipeline with PDF file filter"""
    if not pdf_file:
        raise ValueError("PDF file must be specified")
    
    # Ensure we're using the normalized path
    pdf_file = os.path.normpath(pdf_file)
    
    template = """
Instruction:
You are an expert safety advisor analyzing a Safety Data Sheet (SDS) to answer the given question. Use only the information provided in the SDS and the context for each chunk. Focus on the facts within the SDS to deliver a complete and precise answer.

Answer the question based on the following context:
Context: {context}
Question: {question}

Important Requirements:
Provide a Complete Answer: When the question requires listing items (e.g., ingredients), provide all items without skipping.
Use Information from Context: Use only the SDS and provided context. Do not invent or assume details that aren't explicitly present.
If Unsure, Admit It: If the information is not available in the SDS or context, clearly state this in the response.

Response Format:
Respond in the following JSON format:
{{
  "answer": "Your comprehensive response here",
  "source": "Relevant section titles from the SDS, separated by commas if multiple"
}}

If the answer cannot be determined based on the provided context, respond with:
{{
  "answer": "Information not available in the provided context",
  "source": "N/A"
}}

Requirements Recap:
Focus on Completeness: Answer questions fully, especially when listing items like ingredients—do not stop at just the first few items.
Stay Fact-Based: Do not introduce any external knowledge or assumptions beyond the provided SDS.
Concise and Direct: Keep the response straightforward and avoid unnecessary elaboration.
Always Use JSON Format.
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    model = Ollama(temperature=0, model="llama3.1")
    
    # Load retriever for specific PDF
    retriever = load_retriever(persist_dir, pdf_file)
    
    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | model
        | StrOutputParser()
    )
    
    return chain.invoke(question)

if __name__ == "__main__":

    persist_dir = "./vector_store"
    data_folder = "./data2"
    
    try:
        start_time = time.time()

        # Get all PDF files from the folder
        pdf_files = get_pdf_files(data_folder)
        
        # Check if we have a valid vectorstore
        if not os.path.exists(persist_dir) or not is_valid_vectorstore(persist_dir):
            print("Processing documents for the first time...")
            
            if os.path.exists(persist_dir):
                shutil.rmtree(persist_dir)
            
            os.makedirs(persist_dir)
            retriever = create_retriever(persist_dir, pdf_files)
        else:
            print("Loading existing vectorstore...")
            processed_pdfs = list_processed_pdfs(persist_dir)
            new_pdfs = [pdf for pdf in pdf_files if os.path.normpath(pdf) not in processed_pdfs]
            
            if new_pdfs:
                print(f"\nFound {len(new_pdfs)} new PDFs to process:")
                for pdf in new_pdfs:
                    print(f"- {os.path.basename(pdf)}")
                
                response = input("\nDo you want to add these new PDFs to the existing vectorstore? (y/n): ")
                if response.lower() == 'y':
                    retriever = create_retriever(persist_dir, new_pdfs)

        # Let user select which PDF to query
        print("\nAvailable PDFs to query:")
        for i, pdf in enumerate(pdf_files, 1):
            print(f"{i}. {os.path.basename(pdf)}")
        print(f"Loading time: {time.time() - start_time:.2f} s")
        
        while True:
            try:
                selection = int(input("\nEnter the number of the PDF you want to query (or 0 to exit): "))
                if selection == 0:
                    print("Exiting...")
                    break
                if 1 <= selection <= len(pdf_files):
                    current_pdf = os.path.normpath(pdf_files[selection - 1])
                    break
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

        # Process questions for selected PDF
        csv_file_path = 'question.csv'
        df = pd.read_csv(csv_file_path)

        document_base_name = Path(current_pdf).stem.split('_')[0]

        results = []
        total_time = 0
        print(f"\nProcessing questions for {os.path.basename(current_pdf)}...")
        
        for index, row in df.iterrows():
            start_time = time.time()

            question = row['Question']
            print(f"\nProcessing Question {index + 1}: {question}")
            try:
                response = run_rag_pipeline(persist_dir, question, current_pdf)
                print(f"Response for Question {index + 1}: {response}\n")
                
                results.append({
                    "Question": question,
                    "Response": response
                })

                processing_time = time.time() - start_time
                total_time += processing_time
                print(f"Processing time: {processing_time:.2f} s")
            except Exception as e:
                print(f"Error processing question {index + 1}: {str(e)}")
                results.append({
                    "Question": question,
                    "Response": f"Error: {str(e)}"
                })

        # Save results
        os.makedirs("./Semi_Structure", exist_ok=True)
        output_filename = f"./Semi_Structure/{document_base_name}.csv"
        output_df = pd.DataFrame(results)
        output_df.to_csv(output_filename, index=False)
        print(f"\nResponses saved to output file: {output_filename}")
        print(f"Avarage processing time: {total_time/43:.2f} s")

    except Exception as e:
        print(f"An error occurred: {e}")