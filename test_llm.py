from ragas.testset.generator import TestsetGenerator
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from ragas.testset.evolutions import simple, reasoning, multi_context
from rag_system import RAGSystem
import time
import pandas as pd
import json
import os
import numpy as np
from datetime import datetime
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    answer_correctness
)

start_time = time.time()
current_time = datetime.now().strftime('%m%d %I%M%p').lower()
method = 0
filename = f'{current_time}_{method}'
LLM_model_system = RAGSystem(filename=filename)
model_name = "nomic-embed-text"
llm_model = "llama3.1"
question_dic = "evaluate_question"
new_question = True

def generator_question():
    print("Start generator question")

    generator = TestsetGenerator.from_langchain(
        generator_llm=Ollama(model=llm_model),
        critic_llm=Ollama(model=llm_model),
        embeddings = OllamaEmbeddings(model=model_name)
    )

    testset = generator.generate_with_langchain_docs(LLM_model_system.document, test_size=20, distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25})

    testset_df = testset.to_pandas()
    testset_df.to_csv(question_dic + "/testset.csv")

def evaluate_LLM():
    print("Start evaluate")

    metrics = [
    answer_relevancy,
    answer_correctness,
    ]

    testset_df = pd.read_csv(question_dic + "/testset.csv")

    testing_question = testset_df["question"].values.tolist()
    testing_groundtruth = testset_df['ground_truth'].values.tolist()
    LLM_result_Answer = []
    LLM_result_Context = []
    for i in range(len(testing_question)):
        Answer , Context = LLM_model_system.answer_query_json(testing_question[i])
        LLM_result_Answer.append(json.loads(Answer.lstrip('\n')[:Answer.find('}')+1])["answer"])
        LLM_result_Context.append([Context[i][0].page_content for i in range(4)])
    
    print(np.array(LLM_result_Context).shape)
    pd.DataFrame(LLM_result_Context).to_csv("output/" + filename + "/" + "context_" + os.listdir('pdf')[0] + "_" + str(method) +"_Testing.csv")

    response_dataset = Dataset.from_dict({
        "question" : testing_question,
        "answer" : LLM_result_Answer,
        "contexts" : LLM_result_Context,
        "ground_truth" : testing_groundtruth
    })

    results = evaluate(response_dataset, metrics,llm=Ollama(model=llm_model),embeddings=OllamaEmbeddings(model=model_name))
    results.to_pandas().to_csv("output/" + filename + "/" + "Answer_" + os.listdir('pdf')[0] + "_" + str(method) +"_Testing.csv")

def main():

    if os.listdir(question_dic) == [] or new_question:
        generator_question()
    
    evaluate_LLM()

if __name__ == "__main__":
    print("Start main")
    main()