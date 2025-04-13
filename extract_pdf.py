import PyPDF2
import sys

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
        return text

if __name__ == "__main__":
    pdf_path = "Task-Circuit Quantization Paper.pdf"
    extracted_text = extract_text_from_pdf(pdf_path)
    
    # Save to a text file
    with open("extracted_pdf_content.txt", "w", encoding="utf-8") as f:
        f.write(extracted_text)
    
    print("PDF content extracted and saved to extracted_pdf_content.txt")
