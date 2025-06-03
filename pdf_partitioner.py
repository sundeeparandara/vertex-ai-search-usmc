"""
📄 PURPOSE OF THIS SCRIPT

This script loads a PDF file and breaks it down into smaller, structured parts 
that can be used for AI processing later.

It does 3 main things:
1. Extracts all the content from the PDF — including:
   - 📝 Text blocks (like paragraphs or headings)
   - 📊 Tables
   - 🖼️ Images

2. Sorts the content into 3 separate lists:
   - `text_chunks` for plain text
   - `tables` for structured data
   - `images` for visual elements

3. Prints a preview of the content so you can understand what was extracted

🧠 Why this matters:
Before we can summarize, embed, or search our documents, we need to 
"chop them up" into smaller pieces. This step gives us clean parts that 
we can later feed into Gemini or a vector database.

💡 This script is a local test version of Step 1.4 in the RAG pipeline.

🔧 REQUIREMENT: POPPLER must be installed (used internally by pdf2image)

📌 Windows Users:
- If you see the error: "PDFInfoNotInstalledError: Is poppler installed and in PATH?"
- You need to install Poppler manually:

1. Download from:
   https://github.com/oschwartz10612/poppler-windows/releases

2. Unzip the folder (e.g., to C:\poppler)

3. Add the bin folder to your PATH:
   Example: C:\poppler\poppler-xx.x.0\Library\bin

4. Restart your terminal and run the script again

📌 Linux Users (Debian/Ubuntu):
- You can install Poppler with:
  sudo apt update && sudo apt install poppler-utils

📌 macOS Users (Homebrew):
- Install with:
  brew install poppler

Once Poppler is installed and in your PATH, this script will work correctly.

🔧  OCR NOTE: Tesseract Installation Required for Image-Based PDFs

This script uses the `unstructured` library to extract text, tables, and images from PDFs.

If your PDF contains scanned pages or image-based content (like scanned tables),
the library will try to run OCR (Optical Character Recognition) using Tesseract.

❗ If Tesseract is not installed, you'll get an error like:
    TesseractNotFoundError: tesseract is not installed or it's not in your PATH

✅ HOW TO FIX THIS:

📌 WINDOWS USERS:
1. Download the Windows installer:
   https://github.com/UB-Mannheim/tesseract/wiki

2. During installation:
   - ✅ Make sure "Add to PATH" is checked
   - ✅ Install to default location (e.g., C:\Program Files\Tesseract-OCR)

3. If you forgot to add to PATH:
   - Go to Environment Variables → System → Path → Add:
     C:\Program Files\Tesseract-OCR

4. Restart your terminal or IDE

📌 LINUX (Ubuntu/Debian):
Run this in your terminal:
    sudo apt update && sudo apt install tesseract-ocr

📌 macOS (with Homebrew):
    brew install tesseract

After installing, try running:
    tesseract --version

If you see version info, you're good to go ✅

"""



from unstructured.partition.pdf import partition_pdf
import os
import base64
import pickle
import logging

# Enable logging for unstructured and pdf2image
# logging.basicConfig(level=logging.DEBUG)
# logging.getLogger("unstructured").setLevel(logging.DEBUG)
# logging.getLogger("pdf2image").setLevel(logging.DEBUG)
# logging.getLogger("PIL").setLevel(logging.INFO)  # Pillow image logs


PDF_PATH = "./MCDP 1.pdf"  # Change this to your PDF file
PKL_PATH = "./partitioned_output.pkl"  # Where we save the output

def save_elements_pickle(elements, filename):
    with open(filename, "wb") as f:
        pickle.dump(elements, f)
    print(f"📦 Saved partitioned elements to: {filename}")

def load_elements_pickle(filename):
    with open(filename, "rb") as f:
        elements = pickle.load(f)
    print(f"📂 Loaded cached elements from: {filename}")
    return elements

def main():
    if not os.path.exists(PDF_PATH):
        print(f"❌ PDF file not found: {PDF_PATH}")
        return

    # Check for existing .pkl
    if os.path.exists(PKL_PATH):
        elements = load_elements_pickle(PKL_PATH)
    else:
        print(f"📄 Partitioning PDF: {PDF_PATH}")
        elements = partition_pdf(
            filename=PDF_PATH,
            extract_images_in_pdf=True,
            infer_table_structure=True
        )
        save_elements_pickle(elements, PKL_PATH)

    """
    📚 HOW partition_pdf() WORKS  Element Detection Explained

    This script uses `unstructured.partition_pdf()` to break a PDF into 'elements'.

    Each element is a logical block of content such as:
    - 📝 Paragraphs or headings (CompositeElement)
    - 📊 Tables (Table)
    - 🖼️ Images (Image)

    The size and boundaries of each element are NOT based on a fixed token or character count.

    Instead, they are determined by:
    - Font size and layout
    - Whitespace between blocks
    - Structural hints from the PDF (e.g., tags or metadata)
    - Optional ML layout models (used internally for visual PDFs)

    So each element is as large or small as it naturally appears on the page.
    For example:
    - A full paragraph becomes one element
    - A table stays grouped as one table element
    - An image on its own page becomes one image element

    🧠 Why this matters:
    This gives us semantically meaningful chunks — better for summarization and embedding later on.

    """


    print("\n✅ Partitioning complete!")
    print(f"🔢 Total elements extracted: {len(elements)}\n")

    text_chunks = []
    tables = []
    images = []

    for el in elements:
        el_type = type(el).__name__
        if "Table" in el_type:
            tables.append(el)
        elif "CompositeElement" in el_type:
            text_chunks.append(el)
        elif "Image" in el_type:
            images.append(el)

    # Display text
    print(f"📝 Found {len(text_chunks)} text blocks:")
    for i, chunk in enumerate(text_chunks[:3]):  # Show first 3
        print(f"\n--- Text Block {i+1} ---")
        print(chunk.text.strip()[:1000])  # Preview first 1000 chars

    # Display tables
    print(f"\n📊 Found {len(tables)} tables:")
    for i, tbl in enumerate(tables[:1]):
        print(f"\n--- Table {i+1} ---")
        print(tbl.text.strip()[:1000])

    # Display images (base64 previews)
    print(f"\n🖼️ Found {len(images)} images:")
    for i, img in enumerate(images[:1]):
        print(f"\n--- Image {i+1} ---")
        print(f"MIME type: {img.metadata.image_mime_type}")
        if img.image_data:
            encoded = base64.b64encode(img.image_data.getvalue()).decode("utf-8")
            print(f"Base64 Preview: {encoded[:200]}...")

if __name__ == "__main__":
    main()
