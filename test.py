import pdfplumber
import os

with pdfplumber.open("pdf/" + os.listdir("pdf")[0]) as pdf:
    for page in pdf.pages:
        
            print(page.extract_text())
    
    print()
