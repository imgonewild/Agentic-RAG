import nest_asyncio

import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import camelot 

from llama_index.core import Document

nest_asyncio.apply()

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
)
from llama_index.core.evaluation import (
    DatasetGenerator,
    FaithfulnessEvaluator,
    RelevancyEvaluator
)

from langchain_community.llms.ollama import Ollama
import time

DATA_PATH = "data2"

def ocr_image(image):
    text = pytesseract.image_to_string(image)
    return text

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

                for table in tables:
                    table_content = table.text
                    text = text.replace(table_content, "")
                documents.extend(tables)


                doc = Document(text=text, metadata={"source": pdf_file, "document_name": os.path.basename(pdf_file)})
                documents.append(doc)
    return document

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
            extracted_documents.append(Document(text=table_text, metadata=metadata))
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


gpt4 = Ollama(model="llama3.1")

# Define service context for GPT-4 for evaluation
Settings.llm = gpt4


# Load Data
documents = SimpleDirectoryReader('data2').load_data()
documents = [load_documents()]
print(documents[0].text)
# To evaluate for each chunk size, we will first generate a set of 40 questions from first 20 pages.
eval_documents = documents
data_generator = DatasetGenerator.from_documents(documents)
eval_questions = data_generator.generate_questions_from_nodes(num = 50)

# We will use GPT-4 for evaluating the responses

# Define Faithfulness and Relevancy Evaluators which are based on GPT-4
faithfulness_gpt4 = FaithfulnessEvaluator()
relevancy_gpt4 = RelevancyEvaluator()

# Define function to calculate average response time, average faithfulness and average relevancy metrics for given chunk size
def evaluate_response_time_and_accuracy(chunk_size):
    total_response_time = 0
    total_faithfulness = 0
    total_relevancy = 0

    # create vector index
    llm = Ollama(model="llama3.1")
    Settings.chunk_size = chunk_size
    Settings.chunk_overlap = chunk_size * 0.4
    vector_index = VectorStoreIndex.from_documents(
        eval_documents
    )

    query_engine = vector_index.as_query_engine()
    num_questions = len(eval_questions)

    for question in eval_questions:
        start_time = time.time()
        response_vector = query_engine.query(question)
        elapsed_time = time.time() - start_time
        
        faithfulness_result = faithfulness_gpt4.evaluate_response(
            response=response_vector
        ).passing
        
        relevancy_result = relevancy_gpt4.evaluate_response(
            query=question, response=response_vector
        ).passing

        total_response_time += elapsed_time
        total_faithfulness += faithfulness_result
        total_relevancy += relevancy_result

    average_response_time = total_response_time / num_questions
    average_faithfulness = total_faithfulness / num_questions
    average_relevancy = total_relevancy / num_questions

    return average_response_time, average_faithfulness, average_relevancy

# Iterate over different chunk sizes to evaluate the metrics to help fix the chunk size.
print(documents[0].metadata)
for chunk_size in [300,500,700,1000,1500]:
  avg_time, avg_faithfulness, avg_relevancy = evaluate_response_time_and_accuracy(chunk_size)
  print(f"Chunk size {chunk_size},Chunk overlap {chunk_size * 0.4} - Average Response time: {avg_time:.2f}s, Average Faithfulness: {avg_faithfulness:.4f}, Average Relevancy: {avg_relevancy:.4f}")