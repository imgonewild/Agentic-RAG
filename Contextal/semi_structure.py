# Import necessary libraries
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
import nltk
import os
import pandas as pd
import time

# Step 1: Data Loading and Parsing
def load_pdf_elements(file_path: str):
    return partition_pdf(
        filename=file_path,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
    )

# Step 2: Categorize Elements
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

# Step 3: Summarize Elements
def summarize_elements(text_elements, table_elements):
    prompt_text = """You are an assistant tasked with summarizing tables and text.
    Give a concise summary of the table or text. Table or text chunk: {element}"""
    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = Ollama(temperature=0, model="llama3.1")
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
    
    text_summaries = summarize_chain.batch([e.text for e in text_elements], {"max_concurrency": 5})
    table_summaries = summarize_chain.batch([e.text for e in table_elements], {"max_concurrency": 5})
    
    return text_summaries, table_summaries

# Step 4: Store and Retrieve
def create_retriever(text_summaries, texts, table_summaries, tables):
    vectorstore = Chroma(collection_name="summaries", embedding_function=get_embedding_function(),persist_directory='chroma')
    vectorstore.reset_collection()
    store = InMemoryStore()
    id_key = "doc_id"
    retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key=id_key)
    
    # Add text summaries and raw text
    doc_ids = [str(uuid.uuid4()) for _ in texts]
    summary_texts = [
        Document(page_content=s, metadata={id_key: doc_ids[i]})
        for i, s in enumerate(text_summaries)
    ]
    retriever.vectorstore.add_documents(summary_texts)
    retriever.docstore.mset(list(zip(doc_ids, texts)))
    
    # Add table summaries and raw tables
    table_ids = [str(uuid.uuid4()) for _ in tables]
    summary_tables = [
        Document(page_content=s, metadata={id_key: table_ids[i]})
        for i, s in enumerate(table_summaries)
    ]
    retriever.vectorstore.add_documents(summary_tables)
    retriever.docstore.mset(list(zip(table_ids, tables)))
    
    return retriever

# Step 5: RAG Pipeline
def run_rag_pipeline(retriever, question):
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
Always Use JSON Format."""
    prompt = ChatPromptTemplate.from_template(template)
    model = Ollama(temperature=0, model="llama3.1",format="json")
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    return chain.invoke(question)

# Full Execution Workflow
if __name__ == "__main__":
    
    file_dir_path = "data2"  # Update this path
    file_name = os.listdir(file_dir_path)[0]
    raw_elements = load_pdf_elements(file_dir_path + "\\" + file_name)
    
    categorized_elements = categorize_elements(raw_elements)
    text_elements = [e for e in categorized_elements if e.type == "text"]
    table_elements = [e for e in categorized_elements if e.type == "table"]
    
    text_summaries, table_summaries = summarize_elements(text_elements, table_elements)
    texts = [e.text for e in text_elements]
    tables = [e.text for e in table_elements]
    
    retriever = create_retriever(text_summaries, texts, table_summaries, tables)

    # questions = pd.read_csv('question.csv')

    # results = []

    # for query in questions["Question"]:
    #     print(query + '\n')
    #     result = run_rag_pipeline(retriever=retriever,question=query)
    #     print(result + '\n')
    #     results.append(result)

    # pd.DataFrame({"Quesiont":questions["Question"], "Answer":results}).to_csv(file_name + '_result.csv')
    # print("finsih")

    while True:
        question = input("Enter your question (or type 'quit' to exit): ")
        if question.lower() == 'quit':
            print("Exiting the RAG system. Goodbye!")
            break
        try:
            start_time  = time.time()

            answer = run_rag_pipeline(retriever, question)
            print(f"Answer: {answer}")
            print(f"Context: {retriever.invoke(question)}")
            print(f"response time:{time.time() - start_time:.2f}")
        except Exception as e:
            print(f"An error occurred: {e}")

