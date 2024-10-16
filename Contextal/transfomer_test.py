from transformers import pipeline

pipe = pipeline("text-generation", model="RUCKBReasoning/TableLLM-13b")
print(pipe("hi"))