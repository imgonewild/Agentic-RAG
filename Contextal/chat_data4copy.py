import sys
import csv
import os
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from get_embedding_function import get_embedding_function
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import time
import re
import json

print(f"Python version: {sys.version}")

CHROMA_PATH = "camelot"
print(f"CHROMA_PATH: {CHROMA_PATH}")

PROMPT_TEMPLATE = """
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

def main():
    start_time = time.time()

    print("Starting main function")
    document_names = list_document_ids()
    print(f"Retrieved document names: {document_names}")
    if not document_names:
        print("No documents found in the database. Please run populate_database2.py first.")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Contents of {CHROMA_PATH} directory:")
        try:
            print(os.listdir(CHROMA_PATH))
        except FileNotFoundError:
            print(f"{CHROMA_PATH} directory not found")
        return

    print("Available Documents:")
    for idx, doc_name in enumerate(document_names):
        print(f"{idx}: {doc_name}")
    document_index = int(input("Select the document by entering the index -> "))
    document_name = document_names[document_index]

    question = input("Enter question: ")

    process_questions(question, document_name)

    end_time = time.time()
    response_time = end_time - start_time
    print(f"Response Time: {response_time} seconds")

def process_questions(question,document_name):
        response = query_rag(question, document_name)
        print(f"Processed question: {question}")
        print(response)

def list_document_ids():
    print("Entering list_document_ids function")
    embedding_function = get_embedding_function()
    print(f"Embedding function loaded: {type(embedding_function).__name__}")
    try:
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        print("Connected to Chroma database")
        print(f"Chroma collection name: {db._collection.name}")
        print(f"Chroma collection count: {db._collection.count()}")
    except Exception as e:
        print(f"Error connecting to Chroma database: {e}")
        return []

    try:
        all_documents = db._collection.get()['metadatas']
        print(f"Retrieved {len(all_documents)} documents from database")
        if len(all_documents) == 0:
            print("Database is empty. Please check if populate_database2.py was run successfully.")
    except Exception as e:
        print(f"Error retrieving documents from database: {e}")
        return []

    document_ids = [metadata.get("document_name", "Unknown ID") for metadata in all_documents]
    print(f"Extracted {len(document_ids)} document IDs")

    document_names = list(set([doc_id.split(':')[0].replace('data2\\', '') for doc_id in document_ids]))
    print(f"Found {len(document_names)} unique document names")

    return document_names

def extract_chunk_number(doc_id):
    match = re.search(r':(\d+)$', doc_id)
    if match:
        return int(match.group(1))
    return 0

def query_rag(query_text: str, document_name: str):

    print(f"Querying for document: {document_name}")
    embedding_function = get_embedding_function()
    try:
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        print("Connected to Chroma database for querying")
    except Exception as e:
        print(f"Error connecting to Chroma database for querying: {e}")
        return json.dumps({"answer": f"Error connecting to database: {str(e)}", "source": "N/A"})

    document_name = "data2\\" + document_name
    retriever = db.as_retriever(
        search_kwargs={'filter': {'source': {'$eq': document_name}}},
        search_type="mmr",  # Use Maximum Marginal Relevance for diverse results
        k=3
    )

    try:
        retrieved_docs = retriever.get_relevant_documents(query_text)
        print(f"Retrieved {len(retrieved_docs)} relevant documents")
        print(retrieved_docs)
        if len(retrieved_docs) == 0:
            return json.dumps({"answer": "No relevant documents found", "source": "N/A"})
    except Exception as e:
        print(f"Error retrieving relevant documents: {e}")
        return json.dumps({"answer": f"Error retrieving documents: {str(e)}", "source": "N/A"})

    # retrieved_docs_sorted = sorted(retrieved_docs, key=lambda doc: extract_chunk_number(doc.metadata['id']))

    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    model = Ollama(model="llama3.1",format='json')

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    try:
        response_text = rag_chain.invoke(query_text)   
        print("Response received")
        print(f"Raw response: {response_text}")  # Print raw response for debugging
        
        # Try to parse JSON, if fails, attempt to extract answer and source
        try:
            response_dict = json.loads(response_text)
            return json.dumps(response_dict)
        except json.JSONDecodeError:
            # If JSON parsing fails, attempt to extract answer and source
            answer_match = re.search(r'"answer"\s*:\s*"(.+?)"', response_text, re.DOTALL)
            source_match = re.search(r'"source"\s*:\s*"(.+?)"', response_text, re.DOTALL)
            
            if answer_match and source_match:
                answer = answer_match.group(1)
                source = source_match.group(1)
            else:
                # If regex fails, use the entire response as the answer
                answer = response_text.strip()
                source = "Extracted from raw response"
            
            return json.dumps({"answer": answer, "source": source})
    except Exception as e:
        print(f"Error generating response: {e}")
        return json.dumps({"answer": f"Error generating response: {str(e)}", "source": "N/A"})



if __name__ == "__main__":
    main()