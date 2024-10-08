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
        print("inside rag system")
        self.data_directory = data_dir_path
        self.db_path = db_path
        self.model_name = "nomic-embed-text"
        self.llm_model = "llama3.1"
        self.document = ''

        self.method = method
        self.filename = filename

        self._setup_collection() 
        self.model = Ollama(model=self.llm_model)

#         self.prompt_template = """
# Instruction: Act as an expert safety advisor analyzing a Safety Data Sheet to address the given question. Use only the provided context, but approach the task with a proactive, problem-solving mindset. Follow these guidelines:
# Context: {context}
# Question: {question}

# Approach:
# Analyze the safety data sheet thoroughly, focusing on the most relevant sections for the question.
# Identify potential hazards, safety concerns, or key information related to the query.
# Formulate a comprehensive response that not only answers the question but also provides actionable advice and recommendations.
# If applicable, suggest preventive measures, best practices, or alternative solutions to mitigate risks.
# Prioritize user safety and regulatory compliance in your recommendations.

# Response Structure:
# Compile your analysis into a concise, actionable response that includes:

# Direct answer to the question
# Key safety considerations
# Practical recommendations or solutions
# Any critical warnings or precautions

# Formatting:
# Provide your response in the following JSON format:
# {{
#   "answer": "Your comprehensive response",
#   "source": "Relevant section title(s) from the Safety Data Sheet"
# }}

# Requirements:
# Maintain the persona of an expert safety advisor throughout your response.
# Use only the information provided in the context, but apply critical thinking and expertise to derive insights and recommendations.
# Prioritize user safety and practical, actionable advice in your recommendations.
# Ensure all key information, analysis, and recommendations are included within the single "answer" field of the response.
# When relevant, mention the importance of following local regulations and consulting with appropriate authorities or experts for complex situations.
# Keep the response concise yet comprehensive, focusing on the most crucial information and advice.
# """

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
    
    def _retrieve_context_from_query(self, query_text):
        vectordb = self._initialize_vectorDB()
        vectordb.as_retriever
        context = vectordb.similarity_search_with_score(query_text, k=4)
        return context
    
    def _get_prompt(self, query_text, context):
        with open("output/" + self.filename + "/" + self.filename + '_context.txt', 'a', encoding="utf-8") as file:
            file.write(context[0][0].page_content + '\n' +
                       context[1][0].page_content + '\n'+
                       context[2][0].page_content + '\n'+
                       context[3][0].page_content + '\n\n')
        # print(f" ***** CONTEXT ******{context} \n")
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in context])
        prompt_template = ChatPromptTemplate.from_template(self.prompt_template)
        prompt = prompt_template.format(context=context_text, question=query_text)
        return prompt

    def answer_query(self, query_text):
        context = self._retrieve_context_from_query(query_text)
        prompt = self._get_prompt(query_text,context)
        response_text = self.model.invoke(prompt)
        formatted_response = f"{response_text}\n"
        return formatted_response , context
    
    ############ testing ollama streaming ##############
    def answer_query_streaming(self, query_text):
        context = self._retrieve_context_from_query(query_text)
        prompt = self._get_prompt(query_text,context)
        response_text = self.model.stream(prompt)
        return response_text
    ####################################################

    #經過測試新的prompt若不指定輸出格式較容易造成輸出格式錯誤的問題
    ########### testing ollama json output #############
    def answer_query_json(self, query_text):
        context = self._retrieve_context_from_query(query_text)
        prompt = self._get_prompt(query_text,context)
        response_text = self.model.invoke(prompt,format='json')
        formatted_response = f"{response_text}\n"
        return formatted_response , context
    ####################################################

    #測試Ollama內的Chat功能看看能不能實現LLM的記憶功能
    ######## testing ollama chat function ##############

    # def answer_query_json(self,query_text,pervious_message):
    #     Ollama.

    ####################################################

    def _load_documents(self):
        print("load document")
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
        print("document splitter")
        if self.method == 0:
            # print("overlap")
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
        with open("output/" + self.filename + "/" + chunks[0].metadata["source"][4:-4] + "_chunks.txt", 'a', encoding="utf-8") as file:
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