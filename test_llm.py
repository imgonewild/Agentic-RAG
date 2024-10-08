from ragas.testset.generator import TestsetGenerator
from langchain_community.llms.ollama import Ollama
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from ragas.testset.evolutions import simple, reasoning, multi_context
from rag_system import RAGSystem
import time
import pandas as pd
import json
from datetime import datetime
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    answer_correctness,
    context_recall,
    context_precision,
)

metrics = [
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    answer_correctness,
]

start_time = time.time()
current_time = datetime.now().strftime('%m%d %I%M%p').lower()

method = 0
# 0 = origin, 1 = rewrite question & split by section
filename = f'{current_time}_{method}'

LLM_model_system = RAGSystem(filename=filename)

model_name = "nomic-embed-text"
llm_model = "llama3.1"

generator = TestsetGenerator.from_langchain(
    generator_llm=Ollama(model=llm_model),
    critic_llm=Ollama(model=llm_model),
    embeddings = OllamaEmbeddings(model=model_name)
)

testset = generator.generate_with_langchain_docs(LLM_model_system.document, test_size=10, distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25})

testset_df = testset.to_pandas()
testset_df.to_csv("testset.csv")

testset_df = pd.read_csv("testset.csv")

testing_question = testset_df["question"].values.tolist()
testing_groundtruth = testset_df['ground_truth'].values.tolist()
LLM_result_Answer = []
LLM_result_Context = []
for i in range(len(testing_question)):
    Answer , Context = LLM_model_system.answer_query_json(testing_question[i])
    LLM_result_Answer.append(json.loads(Answer.lstrip('\n')[:Answer.find('}')+1])["answer"])
    LLM_result_Context.append([Context[i][0].page_content for i in range(4)])

response_dataset = Dataset.from_dict({
    "question" : testing_question,
    "answer" : LLM_result_Answer,
    "contexts" : LLM_result_Context,
    "ground_truth" : testing_groundtruth
})

results = evaluate(response_dataset, metrics,llm=Ollama(model=llm_model),embeddings=OllamaEmbeddings(model=model_name))

results.to_pandas().to_csv("Testing_result.csv")

