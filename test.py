import camelot
import os

table_doc = camelot.read_pdf('pdf/' + os.listdir('pdf')[0])
print(len(table_doc))
# print(table_doc)

# for doc in table_doc:
#         camelot.plot(doc, kind='grid').show()
# input()
