import camelot
from camelot.core import TableList
import os
import camelot.image_processing
from unstructured.partition.auto import partition
from langchain_community.document_loaders import UnstructuredPDFLoader
from llmsherpa.readers import LayoutPDFReader
import openparse
import cv2

# pdf_path = 'data2/' + os.listdir('data2')[0]
# api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# tables = TableList(camelot.read_pdf(pdf_path, pages='all', flavor='lattice'))
# print(tables[0].parsing_report)
# camelot.plot(tables[0],kind='').show()
# input()

# loader = UnstructuredPDFLoader(pdf_path)
# print(loader.load())

# elements = partition(filename=pdf_path,strategy='hi_res')
# tables = [el for el in elements if el.category == "Table"]
# Texts = [el for el in elements if el.category == "NarrativeText"]
# Titles = [el for el in elements if el.category == "Title"]

# print(Titles[0])
# print(tables[0])

# # print(LayoutPDFReader(api_url).read_pdf(pdf_path))


# parser = openparse.DocumentParser(
#     table_args={
#         "parsing_algorithm": "pymupdf",
#         "table_output_format": "markdown"
#     }
# )
# parsed_doc = parser.parse(pdf_path)

# for node in parsed_doc.nodes:
#     print(node.text)

