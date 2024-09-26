from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.embeddings import OllamaEmbeddings
# from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_chroma import Chroma
import re
import os

# pip install -U langchain-chroma
class RAGSystem:
    def __init__(self, data_dir_path = "pdf", db_path = "chroma", method=0, filename = 'test') -> None:
        # print("inside rag system")
        self.data_directory = data_dir_path
        self.db_path = db_path
        self.model_name = "nomic-embed-text"
        self.llm_model = "llama3.1"

        self.method = method
        self.filename = filename

        self._setup_collection() 
        self.model = Ollama(model=self.llm_model)
        self.prompt_template = """
            Answer the question based on the following context: {context}. 
            ---
            Answer the question based on the above context: {question}. 
            Identify and include the relevant section title(s) that occur before the information used in your answer.
            Section title should be before the answer.
            Reply in the format: {{"answer": "answer", "source": "section title"}} do not reply other text
            and if the answer contain multiple answers, then combine to one single answer
            and reply in the format:
            {{"answer": "answer 1, answer 2, answer 3, etc", "source": "section title 1, section title 2, section title 3, etc"}}.
            Do not make up answers or use outside information, and if you dont know the answer then reply
            {{"answer": "I dont know", "source": "N/A"}}.
        """

        # self.prompt_template = """
        #     Based on the following context: {context} and answer the question: {question}

        #     Reply in the format: 
        #     {{"answer": "your_answer_here", "source": "your_section_title_here"}}

        #     If the answer contains multiple parts, combine them into a single answer and reply in the format:
        #     {{"answer": "answer1, answer2, answer3, etc.", "source": "source1, source2, source3, etc."}}

        #     If you cannot find the answer, reply:
        #     {{"answer": "The document doesnt have anything about it", "source": "N/A"}}

        #     Moreover, do not provide any additional notes or suggestions.
        # """

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
        vectordb = self._initialize_vectorDB()
        present_in_db = vectordb.get()
        ids_in_db = present_in_db["ids"]
        print(f"Number of existing ids in db: {len(ids_in_db)}")
        # add chunks to db - check if they already exist
        chunks_to_add = [i for i in chunks if i.metadata.get("chunk_id") not in ids_in_db]
        if len(chunks_to_add) > 0:
            vectordb.add_documents(chunks_to_add, ids = [i.metadata["chunk_id"] for i in chunks_to_add])
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
    
    def _retrieve_context_from_query(self, query_text):
        vectordb = self._initialize_vectorDB()
        context = vectordb.similarity_search_with_score(query_text, k=4)
        return context
    
    def _get_prompt(self, query_text):
        context = self._retrieve_context_from_query(query_text)
        # print(f" ***** CONTEXT ******{context} \n")
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in context])
        prompt_template = ChatPromptTemplate.from_template(self.prompt_template)
        prompt = prompt_template.format(context=context_text, question=query_text)
        return prompt

    def answer_query(self, query_text):
        prompt = self._get_prompt(query_text)
        response_text = self.model.invoke(prompt)
        formatted_response = f"{response_text}\n"
        return formatted_response

    def _load_documents(self):
        loader = PyPDFDirectoryLoader(self.data_directory)
        pages = loader.load()
        text = [Document("")]
        # Clean the content of each page
        for page in pages:
            page.page_content = self._clean_text(page.page_content)
            text[0].page_content += page.page_content
            text[0].metadata = page.metadata
        
        return text

    def _document_splitter(self, documents):
        if self.method == 0:
            print("overlap")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=600,
                length_function=len,
                is_separator_regex=False,
            )
        elif self.method == 1:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=250,
                length_function=len,
                is_separator_regex=True,
                separators=[r'[sS][eE][cC][tT][iI][oO][nN]\s([1-9IVX]+)(\.|\:|\ -)'],  # Case-insensitive regex pattern
            )
        # splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=1000,
        #     chunk_overlap=250,
        #     length_function=len,
        #     is_separator_regex=False,
        # )
        # Split the documents into chunks
        chunks = splitter.split_documents(documents)

        #from datetime import datetime
        #current_time = datetime.now().strftime('%m%d %I%M%p').lower()
        # filename = f'{current_time}_chunks.txt'
        os.makedirs("output/" + self.filename )
        with open("output/" + self.filename + "/" + self.data_directory  + "_" + self.filename + "_chunks.txt", 'a', encoding="utf-8") as file:
            for idx, chunk in enumerate(chunks):
                file.write(f"Chunk {idx + 1}:\n")
                file.write(f"Content: {chunk.page_content}\n")
                file.write(f"Metadata: {chunk.metadata}\n")
                file.write("\n" + "-" * 80 + "\n")
        
        return chunks

    
    def _get_embedding_func(self):
        embeddings = OllamaEmbeddings(model=self.model_name)
        return embeddings
    
    def _initialize_vectorDB(self):
        return Chroma(
            persist_directory = self.db_path,
            embedding_function = self._get_embedding_func(),
        )