import pdfplumber
import re
import json
from config import CHUNK_SIZE, PDF_PATH, CHUNKS_FILE


def extract_text(pdf_path):
    print("---Extracting text from PDF---")
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def clean_text(text):
    print("---Cleaning text---")
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z0-9.?,;:()\- ]', '', text)
    return text


def chunk_text(text, chunk_size):
    print("---Creating chunks---")
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)

    return chunks


def save_chunks(chunks, path):
    print("---Saving chunks---")
    data = [{"id": i, "text": c} for i, c in enumerate(chunks)]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    raw = extract_text(PDF_PATH)
    clean = clean_text(raw)
    chunks = chunk_text(clean, CHUNK_SIZE)
    save_chunks(chunks, CHUNKS_FILE)

    print(f"Done! Total chunks: {len(chunks)}")