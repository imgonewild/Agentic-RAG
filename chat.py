from rag_system import RAGSystem
import time
from datetime import datetime
start_time = time.time()
current_time = datetime.now().strftime('%m%d %I%M%p').lower()

method = 0
# 0 = origin, 1 = rewrite question & split by section
filename = f'{current_time}_{method}'
rag_system = RAGSystem(method = method, filename = filename)

while True:
    for i in rag_system.answer_query_streaming(input('Question:')):
            print(i,end="")
    print()
    print('finish')
        