import pdfplumber
import docx
from bs4 import BeautifulSoup
import csv
import requests

def extract_text_from_pdf(path: str) -> str:
    out = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            out.append(page.extract_text() or "")
    return "\n\n".join(out)

def extract_text_from_docx(path: str) -> str:
    doc = docx.Document(path)
    return "\n\n".join([p.text for p in doc.paragraphs])

def extract_text_from_html(path_or_url: str) -> str:
    if path_or_url.startswith('http'):
        r = requests.get(path_or_url, timeout=20)
        soup = BeautifulSoup(r.text, 'html.parser')
    else:
        with open(path_or_url, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
    for s in soup(['script','style']):
        s.decompose()
    return soup.get_text(separator='\n')

def extract_text_from_csv(path: str) -> str:
    rows = []
    with open(path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for r in reader:
            rows.append(' | '.join(r))
    return '\n'.join(rows)

def extract_text_from_path(path: str, filename: str) -> str:
    ext = filename.lower().split('.')[-1]
    if ext in ('pdf',):
        return extract_text_from_pdf(path)
    if ext in ('docx','doc'):
        return extract_text_from_docx(path)
    if ext in ('html','htm'):
        return extract_text_from_html(path)
    if ext in ('csv',):
        return extract_text_from_csv(path)
    try:
        return extract_text_from_html(path)
    except Exception:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
