import camelot
from camelot.core import TableList
import os
from unstructured.partition.auto import partition
from langchain_community.document_loaders import UnstructuredPDFLoader
from llmsherpa.readers import LayoutPDFReader
import openparse

pdf_path = 'data2/' + os.listdir('data2')[0]
api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
# tables = TableList(camelot.read_pdf(pdf_path, pages='all', flavor='lattice'))
# for table in tables:
#     table_text = table.df.to_json(index=False)
#     print(table_text)

# loader = UnstructuredPDFLoader(pdf_path)
# print(loader.load())

# elements = partition(filename=pdf_path,strategy='hi_res')
# tables = [el for el in elements if el.category == "Table"]
# Texts = [el for el in elements if el.category == "NarrativeText"]
# Titles = [el for el in elements if el.category == "Title"]

# print(Titles[0])
# print(tables[0])

# # print(LayoutPDFReader(api_url).read_pdf(pdf_path))


parser = openparse.DocumentParser(
    table_args={
        "parsing_algorithm": "pymupdf",
        "table_output_format": "markdown"
    }
)
parsed_doc = parser.parse(pdf_path)

for node in parsed_doc.nodes:
    print(node.text)

