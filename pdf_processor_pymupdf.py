import fitz  # PyMuPDF
import pickle
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

class TextElement:
    """Simple text element class to match the structure expected by load_and_store_vectors.py"""
    def __init__(self, text, page_number, chunk_index=0):
        self.text = text
        self.metadata = SimpleMetadata(page_number)
        self.chunk_index = chunk_index

class SimpleMetadata:
    """Simple metadata class to store page information"""
    def __init__(self, page_number):
        self.page_number = page_number

def clean_text(text):
    """Clean and normalize text extracted from PDF"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove page headers/footers if they follow a pattern
    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)  # Remove standalone page numbers
    # Remove excessive newlines
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    return text.strip()

def extract_and_chunk_pdf(pdf_path, chunk_size=1000, chunk_overlap=200):
    """
    Extract text from PDF and chunk it into elements
    
    Args:
        pdf_path: Path to PDF file
        chunk_size: Maximum characters per chunk
        chunk_overlap: Characters to overlap between chunks
        
    Returns:
        List of TextElement objects
    """
    doc = fitz.open(pdf_path)
    elements = []
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]  # Split on paragraphs first, then sentences
    )
    
    total_pages = len(doc)
    print(f"📄 Processing PDF: {total_pages} total pages")
    
    for page_num in range(total_pages):
        page = doc[page_num]
        text = page.get_text()
        
        # Clean the extracted text
        cleaned_text = clean_text(text)
        
        if cleaned_text.strip():  # Only process pages with content
            # Page numbering - adjust this if you need to skip front matter
            actual_page_num = page_num + 1  # PDF page number (1-indexed)
            # actual_page_num = page_num - 2  # Uncomment and adjust if you want to skip title pages
            
            # Chunk the page text
            chunks = text_splitter.split_text(cleaned_text)
            
            for chunk_index, chunk in enumerate(chunks):
                if chunk.strip():  # Only include non-empty chunks
                    element = TextElement(
                        text=chunk.strip(),
                        page_number=actual_page_num,
                        chunk_index=chunk_index
                    )
                    elements.append(element)
            
            print(f"✅ Processed page {actual_page_num}: {len(chunks)} chunks, {len(cleaned_text)} characters")
        else:
            print(f"⏭️  Skipped empty page {page_num + 1}")
    
    doc.close()
    print(f"\n✅ Total elements created: {len(elements)}")
    return elements

def main():
    """Main processing function - just chunk and save to pickle"""
    # Configuration
    pdf_path = "MCDP_1.pdf"  # Update with your PDF path
    output_pkl = "partitioned_output_pymupdf.pkl"
    
    print("🚀 Starting PDF chunking with PyMuPDF...")
    
    # Check if the output pickle file already exists
    if os.path.exists(output_pkl):
        print(f"⚠️  {output_pkl} already exists.")
        response = input("Do you want to overwrite it? (y/n): ").strip().lower()
        if response != 'y':
            print("✅ Skipping chunking process - using existing file.")
            
            # Show info about existing file
            try:
                with open(output_pkl, "rb") as f:
                    existing_elements = pickle.load(f)
                print(f"📊 Existing file contains {len(existing_elements)} elements")
                print(f"🔄 Next step: Update PKL_PATH in load_and_store_vectors.py and run it")
            except Exception as e:
                print(f"❌ Error reading existing file: {e}")
                print("You may want to delete the corrupted file and re-run.")
            return
        else:
            print("🔄 Overwriting existing file...")
    
    # Extract and chunk PDF
    elements = extract_and_chunk_pdf(
        pdf_path,
        chunk_size=1000,  # Adjust based on your needs
        chunk_overlap=200
    )
    
    # Save to pickle file
    print(f"\n💾 Saving {len(elements)} elements to {output_pkl}...")
    with open(output_pkl, "wb") as f:
        pickle.dump(elements, f)
    
    print(f"🎉 Chunking complete! Elements saved to {output_pkl}")
    print(f"📊 Summary:")
    print(f"   - Total elements: {len(elements)}")
    print(f"   - Processing method: PyMuPDF with accurate page numbers")
    print(f"   - Output format: Compatible with load_and_store_vectors.py")
    print(f"\n🔄 Next step: Update PKL_PATH in load_and_store_vectors.py and run it")

if __name__ == "__main__":
    main() 