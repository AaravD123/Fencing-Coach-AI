import os
import json
from PyPDF2 import PdfReader

# Path to the PDF
pdf_path = "../fencing-handbook/PRXFE_USAFencingRules.pdf"
output_path = "../data/fencing_rulebook_chunks.json"

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text_pages = [page.extract_text() for page in reader.pages if page.extract_text()]
    return text_pages

def split_into_chunks(text_pages, chunk_size=500):
    chunks = []
    for page_num, page in enumerate(text_pages):
        words = page.split()
        for i in range(0, len(words), chunk_size):
            chunk_text = " ".join(words[i:i+chunk_size])
            chunks.append({
                "page": page_num + 1,
                "text": chunk_text
            })
    return chunks

def save_chunks(chunks, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

if __name__ == "__main__":
    pages = extract_text_from_pdf(pdf_path)
    chunks = split_into_chunks(pages)
    save_chunks(chunks, output_path)
    print(f"Saved {len(chunks)} chunks to {output_path}")
