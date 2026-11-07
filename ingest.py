import re
from pathlib import Path
from pypdf import PdfReader

def clean_text(text: str) -> str:
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    text = re.sub(r'(Page \d+ of \d+|Confidential|RBI/.*?\d{4})', '', text, flags=re.I)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def extract_pdfs(raw_pdfs_dir: str = "data/raw_pdfs", out_dir: str = "data/extracted"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    pdf_files = list(Path(raw_pdfs_dir).glob("*.pdf"))
    for pdf_path in pdf_files:
        reader = PdfReader(pdf_path)
        for i, page in enumerate(reader.pages):
            raw = page.extract_text() or ""
            cleaned = clean_text(raw)
            if len(cleaned) < 100:
                continue
            out_file = Path(out_dir) / f"{pdf_path.stem}_p{i+1:04d}.txt"
            out_file.write_text(cleaned, encoding="utf-8")
            meta = {
                "source_pdf": pdf_path.name,
                "page": i + 1,
                "doc_family": "KYC" if any(x in pdf_path.name.upper() for x in ["KYC", "AML", "MD"]) else "PSL",
                "year": re.search(r'20\d{2}', pdf_path.name).group(0) if re.search(r'20\d{2}', pdf_path.name) else "unknown"
            }
            (out_file.with_suffix(".meta")).write_text(str(meta))
    print(f"Extracted pages from {len(pdf_files)} PDFs")