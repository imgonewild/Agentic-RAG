from rag_system import RAGSystem
import time
from datetime import datetime
import pandas as pd
import json
import os

start_time = time.time()
current_time = datetime.now().strftime('%m%d %I%M%p').lower()

method = 0
# 0 = origin, 1 = rewrite question & split by section
filename = f'{current_time}_{method}'
rag_system = RAGSystem(method = method, filename = filename)

def main():
    # questions = pd.read_csv("question/question_1.csv", encoding='UTF-8')["Question"]
    questions = [
        "What is the product identifier?",
        "When was the Safety Data Sheet issued?",
        "What is the product description?",
        "Who is the manufacturer of the product?",
        "What is the emergency telephone number for the product?",
        "What are the hazard classifications associated with the product?",
        "What is the signal word used for the product?",
        "What are the hazard statements related to the product?",
        "What precautions should be taken while handling the product?",
        "What are the first aid measures in case of inhalation?",
        "What are the first aid measures in case of skin contact?",
        "What are the first aid measures in case of eye contact?",
        "What are the first aid measures in case of ingestion?",
        "What are the suitable extinguishing media for a fire involving the product?",
        "What are the special hazards arising from the substance during a fire?",
        "Are there any special fire handling procedures required for the product?",
        "How should accidental releases, spill, or leak of the product be handled?",
        "What precautions should be taken for safe handling and storage of the product?",
        "What are the exposure controls and personal protection measures recommended for the product?",
        "What are the physical and chemical properties of the product?",
        "Is the product stable or reactive? Are there any hazardous reactions to be aware of?",
        "What are the possible hazardous decomposition products of the product?",
        "What is the toxicological information available for the product?",
        "What are the safety precautions and handling instructions for using the product?",
        "What are the potential hazards associated with the product?",
        "What are the specific health effects or risks associated with exposure to the product?",
        "What are the emergency procedures in case of inhalation of the product?",
        "What are the emergency procedures in case of skin contact with the product?",
        "What are the emergency procedures in case of eye contact with the product?",
        "What are the emergency procedures in case of ingestion of the product?",
        "What is the waste disposal method for the product?",
        "Are there any specific storage requirements or conditions for safely storing the product?",
        "What is the shelf life of the product?",
        "What are the composition and ingredients of the product?",
        "Are there any known hazardous reactions or incompatibilities with other substances or materials?",
        "What is the odor of the product?",
        "What is the boiling point of the product?",
        "Is the product flammable?",
        "What is the solubility of the product?",
        "Is the product stable under certain conditions, and are there any potential decomposition products?",
        "What are the recommended engineering controls, personal protective equipment (PPE), and respiratory protection for handling the product?",
        "Are there any specific environmental precautions or considerations for the disposal of the product or its containers?",
        "Can the specific chemical identity or exact percentage of the composition be disclosed for the product?"
    ]

    # questions = ["What are the first aid measures in case of skin contact?"]

    cnt = 1
    with open("output/" + filename + "/" + filename + '.txt', 'w', encoding="utf-8") as file:
        Answer = pd.DataFrame({"Question":[],"Answer":[],"Source":[],"Context_1":[],"Context_2":[],"Context_3":[],"Context_4":[]})

        for question in questions:
            if(method):
                question = "Based on the following Safety Data Sheet, " + question

            response , context = rag_system.answer_query_json(question)
            file.write(f"{cnt}: {question}\n")
            file.write(f"{response}\n")

            print(str(cnt) + ": " + question)  # Convert cnt to string
            print(response)   
            try:
                j = json.loads(response.lstrip('\n')[:response.find('}')+1])
                Answer = Answer._append({"Question":question,"Answer":j["answer"],"Source":j["source"]
                                         ,"Context_1":context[0][0].page_content,"Context_2":context[1][0].page_content
                                         ,"Context_3":context[2][0].page_content,"Context_4":context[3][0].page_content},ignore_index=True)
            except json.JSONDecodeError:
                try:
                    response = response.lstrip('\n')[:response.find('}')+1]
                    Answer = Answer._append({"Question":question,"Answer":[i.split('answer') for i in response.split('source')][0][1][3:-3],
                                         "Source":[i.split('answer') for i in response.split('source')][1][0][3:-1]
                                         ,"Context_1":context[0][0].page_content,"Context_2":context[1][0].page_content
                                         ,"Context_3":context[2][0].page_content,"Context_4":context[3][0].page_content},ignore_index=True)
                except Exception as err:
                    print(err)
                    Answer = Answer._append({"Question":question,"Answer":response,"Source":err                                         
                                            ,"Context_1":context[0][0].page_content,"Context_2":context[1][0].page_content
                                            ,"Context_3":context[2][0].page_content,"Context_4":context[3][0].page_content},ignore_index=True)
            cnt+=1
        Answer.to_csv("output/" + filename + "/" + os.listdir('pdf')[0] + "_" + filename +".csv",encoding='utf-8-sig')
    print("\n--- Total %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    print("Start main")
    main()