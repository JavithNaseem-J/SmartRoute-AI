"""
Build Vector Database from Documents
=====================================
This script processes documents (PDF, TXT) and creates embeddings for RAG.

Usage:
    python scripts/build_vectordb.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.retriever import DocumentRetriever
from src.utils.logger import logger


def main():
    """Build vector database from documents"""
    
    # Paths
    docs_dir = Path("data/documents")
    persist_dir = Path("data/embeddings")
    
    print("=" * 60)
    print("  BUILDING VECTOR DATABASE FOR RAG")
    print("=" * 60)
    print()
    
    # Check documents exist
    if not docs_dir.exists():
        print(f"❌ Documents directory not found: {docs_dir}")
        return
    
    # List documents
    pdf_files = list(docs_dir.glob("**/*.pdf"))
    txt_files = list(docs_dir.glob("**/*.txt"))
    
    print(f"📁 Documents directory: {docs_dir}")
    print(f"📄 PDF files found: {len(pdf_files)}")
    for f in pdf_files:
        print(f"   - {f.name}")
    print(f"📄 TXT files found: {len(txt_files)}")
    for f in txt_files:
        print(f"   - {f.name}")
    print()
    
    if len(pdf_files) + len(txt_files) == 0:
        print("❌ No documents found! Add PDF or TXT files to data/documents/")
        return
    
    # Build vector database
    print("🔄 Processing documents...")
    print()
    
    try:
        vectordb = DocumentRetriever.create_vectordb(
            docs_dir=docs_dir,
            persist_dir=persist_dir,
            chunk_size=800,
            chunk_overlap=150
        )
        
        print()
        print("=" * 60)
        print("  ✅ VECTOR DATABASE CREATED SUCCESSFULLY!")
        print("=" * 60)
        print()
        print(f"📦 Saved to: {persist_dir}")
        print()
        print("🚀 Next steps:")
        print("   1. Run the API: python api/main.py")
        print("   2. Or Dashboard: streamlit run dashboard/app.py")
        print("   3. Ask questions about your documents!")
        print()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
