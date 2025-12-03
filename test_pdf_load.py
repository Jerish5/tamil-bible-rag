import pypdf
import os
import sys

PDF_PATH = "Viviliya Thedal.pdf"
OUTPUT_FILE = "pdf_test_output.txt"

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    if os.path.exists(PDF_PATH):
        f.write(f"File found: {PDF_PATH}\n")
        try:
            reader = pypdf.PdfReader(PDF_PATH)
            num_pages = len(reader.pages)
            f.write(f"Successfully loaded {num_pages} pages.\n")
            
            pages_to_check = [0, 10, 20, 50]
            for p in pages_to_check:
                if p < num_pages:
                    text = reader.pages[p].extract_text()
                    f.write(f"\n--- Page {p+1} content snippet ---\n")
                    f.write(text[:500] + "\n")
                    if not text.strip():
                        f.write("(Empty page)\n")
        except Exception as e:
            f.write(f"Error loading PDF: {e}\n")
    else:
        f.write(f"File NOT found: {PDF_PATH}\n")

print(f"Output written to {OUTPUT_FILE}")
