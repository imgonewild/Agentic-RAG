from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.embeddings import OllamaEmbeddings
# from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_chroma import Chroma
import re
import os
from PIL import Image
import fitz 
import pytesseract
import io
import camelot

# pip install -U langchain-chroma
class RAGSystem:
    def __init__(self, data_dir_path = "pdf", db_path = "chroma", method=0, filename = 'test') -> None:
        print("inside rag system")
        self.data_directory = data_dir_path
        self.db_path = db_path
        self.model_name = "nomic-embed-text"
        self.llm_model = "llama3.1"
        self.document = ''

        self.method = method
        self.filename = filename
# 
        self._setup_collection() 
        self.model = Ollama(model=self.llm_model)

        self.prompt_template = """
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

    def _clean_text(self, text):
        """ Clean the document text by removing unnecessary headers, footers, and formatting characters. """
        # Remove headers and footers using regex
        text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text)  # Remove "Page X of Y"
        # text = re.sub(r'_{10,}', '', text)  # Remove long sequences of underscores (formatting characters)
        
        # Remove multiple spaces, tabs, and newline characters
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize white spaces
        text = re.sub(r' +', ' ', text).strip()
        return text

    def _setup_collection(self):
        pages = self._load_documents()
        chunks = self._document_splitter(pages)
        chunks = self._get_chunk_ids(chunks)
        self.document = chunks
        vectordb = self._initialize_vectorDB()
        present_in_db = vectordb.get()
        ids_in_db = present_in_db["ids"]
        print(f"Number of existing ids in db: {len(ids_in_db)}")
        # add chunks to db - check if they already exist
        chunks_to_add = [i for i in chunks if i.metadata.get("chunk_id") not in ids_in_db]
        print("finish get chunk")
        if len(chunks_to_add) > 0:
            data_ids = [i.metadata["chunk_id"] for i in chunks_to_add]
            tmp = [[i] for i in data_ids]
            tmp_2 = [[i] for i in chunks_to_add]
            print("load chunk id")
            print(f"add to db:{len(chunks_to_add)}")
            for i in range(len(tmp)):
                print(tmp[i])
                vectordb.add_documents(tmp_2[i], ids = tmp[i])
            # vectordb.add_documents(chunks_to_add, ids = ids)
            print(f"added to db: {len(chunks_to_add)} records")
            # vectordb.persist()
        else:
            print("No records to add")

    def _get_chunk_ids(self, chunks):
        ''''
        for same page number: x
            source_x_0
            source_x_1
            source_x_2
        for same source but page number: x+1
            source_x+1_0
            source_x+1_1
            source_x+1_2
        '''
        prev_page_id = None
        for i in chunks:
            src = i.metadata.get("source")
            page = i.metadata.get("page")
            curr_page_id = f"{src}_{page}"
            if curr_page_id == prev_page_id:
                curr_chunk_index += 1
            else:
                curr_chunk_index = 0
            # final id of chunk
            curr_chunk_id = f"{curr_page_id}_{curr_chunk_index}"
            prev_page_id = curr_page_id
            i.metadata["chunk_id"] = curr_chunk_id
        return chunks        
    
    def _retrieve_context_from_query(self,file_name, query_text):
        vectordb = self._initialize_vectorDB()
        print(file_name)
        context = vectordb.similarity_search_with_score(query_text, k=4,filter= {'source': {'$eq': file_name}})
        return context
    
    def _get_prompt(self, query_text, context):
        # print(f" ***** CONTEXT ******{context} \n")
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in context])
        prompt_template = ChatPromptTemplate.from_template(self.prompt_template)
        prompt = prompt_template.format(context=context_text, question=query_text)
        return prompt

    def answer_query_json(self, file_name ,query_text):
        context = self._retrieve_context_from_query(file_name,query_text)
        for doc, i in context:
            print(doc.metadata)
        prompt = self._get_prompt(query_text,context)
        response_text = self.model.invoke(prompt,format='json')
        formatted_response = f"{response_text}\n"
        return formatted_response
    
    def evaluate_query(self, file_name ,query_text):
        context = self._retrieve_context_from_query(file_name,query_text)
        prompt = self._get_prompt(query_text,context)
        response_text = self.model.invoke(prompt,format='json')
        formatted_response = f"{response_text}\n"
        return formatted_response , context

    def _load_documents(self):
        documents = []
        # Iterate through the files in the data directory and extract text from PDFs.
        for root, dirs, files in os.walk(self.data_directory):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_file = os.path.join(root, file)
                    if self.is_scanned_pdf(pdf_file):
                        print(f"{pdf_file} is a scanned PDF. Performing OCR.")
                        text = self.extract_text_from_pdf(pdf_file)
                    else:
                        print(f"{pdf_file} is not a scanned PDF. Extracting text directly.")
                        document = fitz.open(pdf_file)
                        text = ""
                        for page_num in range(len(document)):
                            page = document.load_page(page_num)
                            text += page.get_text()

                    # Extract tables using Camelot, if applicable
                    tables = self.extract_tables_from_pdf(pdf_file)

                    for table in tables:
                        table_content = table.page_content
                        text = text.replace(table_content, "")
                    documents.extend(tables)


                    doc = Document(page_content=text, metadata={"source": pdf_file, "document_name": os.path.basename(pdf_file)})
                    documents.append(doc)
                    print(doc.metadata)
        return documents

    def is_scanned_pdf(self,pdf_path):
        document = fitz.open(pdf_path)
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text = page.get_text()
            if text.strip():
                return False
        return True

    def ocr_image(self,image):
        text = pytesseract.image_to_string(image)
        return text

    def extract_text_from_pdf(self,pdf_path):
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
                text = self.ocr_image(img)
                full_text += text

        return full_text

    def extract_tables_from_pdf(self,pdf_path):
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

    def _document_splitter(self, documents):
        print("document splitter")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=600,
            length_function=len,
            is_separator_regex=False,
        )


        chunks = splitter.split_documents(documents)
        return chunks

    
    def _get_embedding_func(self):
        embeddings = OllamaEmbeddings(model=self.model_name)
        return embeddings
    
    def _initialize_vectorDB(self):
        return Chroma(
            persist_directory = self.db_path,
            embedding_function = self._get_embedding_func(),
        )