from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from datasets import Dataset
import pandas as pd
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
model_name = "nomic-embed-text"
llm_model = "llama3.1"

output_csv = pd.read_csv('output.csv')
context = []
for i in range(len(output_csv["Context_1"].tolist())):
    context.append([output_csv["Context_1"].tolist()[i],output_csv["Context_2"].tolist()[i],output_csv["Context_3"].tolist()[i],output_csv["Context_4"].tolist()[i]])
print(context)

response_dataset = Dataset.from_dict({
    "question" : output_csv["Question"].tolist(),
    "answer" : output_csv["Answer"].tolist(),
    "contexts" : context,
})

result = evaluate(response_dataset,metrics,
                  llm=Ollama(model=llm_model),embeddings=OllamaEmbeddings(model=model_name))

result.to_pandas().to_csv("result_csv")
