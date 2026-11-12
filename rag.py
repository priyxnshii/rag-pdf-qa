from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdf(file_path):
    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text

def load_pdf_with_pages(file_path):
    reader = PdfReader(file_path)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()

        if page_text:
            pages.append({
                "page": page_number,
                "text": page_text
            })

    return pages

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )

    chunks = splitter.split_text(text)

    return chunks

def split_pages_into_chunks(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 100
    )

    chunks = []

    for page in pages:
        page_chunks = splitter.split_text(
            page["text"]
        )

        for chunk in page_chunks:
            chunks.append({
                "text": chunk,
                "page": page["page"]
            })

    return chunks

if __name__ == "__main__":

    # 1. Load PDF
    pdf_text = load_pdf("data/sample.pdf")
    print("Total characters: ", len(pdf_text))

    # 2. Split into chunks
    chunks = split_text(pdf_text)
    print("Total chunks: ", len(chunks))

    print("\n==== FIRST 3 CHUNKS =====\n")

    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- CHUNK {i + 1} ----")
        print(chunk)

    print("\n==== CONCLUSION CHUNKS ====\n")

    pages = load_pdf_with_pages("data/sample.pdf")
    page_chunks = split_pages_into_chunks(pages)

    for i, chunk in enumerate(page_chunks):
        if "CONCLUSION" in chunk["text"].upper():
            print(f"\nChunk index: {i}")
            print(f"Physical PDF page: {chunk['page']}")
            print(chunk["text"])