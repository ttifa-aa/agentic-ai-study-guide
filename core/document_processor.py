"""
Document processing module for loading and processing various academic file formats.
Handles PDF, DOCX, TXT, and PPTX files with appropriate loaders and metadata tagging.
"""

import os  # for file path operations and extension extraction
import tempfile  # for creating temporary files when processing uploads
from typing import List, Optional, Dict, Any  # type hints for function signatures
from dataclasses import dataclass  # for creating structured data classes

# langchain document loaders for different file formats
from langchain_community.document_loaders import (
    PyPDFLoader,           # for pdf textbooks, lecture slides, question papers
    TextLoader,            # for plain text notes and documents
    Docx2txtLoader,        # for microsoft word documents (lab manuals, assignments)
    UnstructuredPowerPointLoader  # for powerpoint lecture slides
)
from langchain_core.documents import Document  # base document class for langchain

from config.settings import config, ContentType  # import configuration and content types


@dataclass
class ProcessedDocument:
    """Represents a processed document with metadata."""
    filename: str                    # original filename of the uploaded document
    content_type: str                # type of content (lecture notes, textbook, etc.)
    chunks: List[Document]           # document chunks after splitting
    num_chunks: int                  # total number of chunks created
    file_size_kb: float              # file size in kilobytes
    processing_time_seconds: float   # time taken to process in seconds


def get_file_extension(filename: str) -> str:
    """
    Extract file extension from filename in lowercase.
    
    Args:
        filename (str): Name of the file
    
    Returns:
        str: Lowercase file extension including the dot (e.g., '.pdf')
    """
    # os.path.splitext splits filename into (name, extension) tuple
    # [1] gets the extension part, .lower() converts to lowercase for consistency
    ext = os.path.splitext(filename)[1].lower()
    # example: "document.PDF" -> ".pdf"
    return ext  # return the normalized extension


def is_supported_file(filename: str) -> bool:
    """
    Check if file extension is supported by the system.
    
    Args:
        filename (str): Name of the file to check
    
    Returns:
        bool: True if file extension is supported, False otherwise
    """
    ext = get_file_extension(filename)  # get normalized extension
    # check if extension is in the supported set from config
    return ext in config.SUPPORTED_EXTENSIONS


def get_loader_for_file(filepath: str):
    """
    Get appropriate document loader based on file extension.
    
    Args:
        filepath (str): Path to the file to be loaded
    
    Returns:
        Document loader instance for the specific file type
    
    Raises:
        ValueError: If file extension is not supported
    """
    ext = get_file_extension(filepath)  # get normalized extension
    
    # map file extensions to their corresponding loader classes
    # this is similar to the dictionary approach you used in app.py
    loader_map = {
        ".pdf": PyPDFLoader,           # pdf files use pypdfloader
        ".docx": Docx2txtLoader,       # word documents use docx2txtloader
        ".doc": Docx2txtLoader,        # older .doc files also use docx2txtloader
        ".txt": TextLoader,            # plain text files use textloader
        ".pptx": UnstructuredPowerPointLoader  # powerpoint files use unstructuredpowerpointloader
    }
    
    if ext not in loader_map:
        # raise error for unsupported file types
        raise ValueError(f"Unsupported file extension: {ext}")
    
    # return loader instance initialized with the file path
    return loader_map[ext](filepath)


def process_uploaded_file(
    uploaded_file,  # streamlit uploadedfile object
    content_type: str,  # selected content type from ui
    chunk_size: int = None,  # optional custom chunk size
    chunk_overlap: int = None  # optional custom chunk overlap
) -> List[Document]:
    """
    Process an uploaded file and return document chunks with metadata.
    this function handles the entire pipeline from temp file to tagged chunks.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        content_type (str): Content type selected by user
        chunk_size (int, optional): Custom chunk size for splitting
        chunk_overlap (int, optional): Custom overlap for splitting
    
    Returns:
        List[Document]: List of document chunks with metadata
    """
    import time  # for measuring processing time
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    start_time = time.time()  # record start time for performance measurement
    
    # use config defaults if not specified
    chunk_size = chunk_size or config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP
    
    # create temporary file to store uploaded content
    # using tempfile.namedtemporaryfile creates a secure temporary file
    # delete=False keeps the file after closing for processing
    suffix = os.path.splitext(uploaded_file.name)[1]  # get file extension for temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        # write uploaded content to temporary file
        f.write(uploaded_file.getvalue())
        # getvalue() returns the binary content of the uploaded file
        temp_path = f.name  # store path to temporary file for later use
    
    try:
        # get appropriate loader for this file type
        loader = get_loader_for_file(temp_path)
        
        # load the document using the selected loader
        # loader.load() returns a list of document objects
        raw_docs = loader.load()
        
        # tag each document chunk with metadata for tracking and retrieval
        for doc in raw_docs:
            # add content type metadata - helps the assistant understand what kind of content this is
            doc.metadata["content_type"] = content_type
            
            # add source filename - useful for attribution and cross-referencing
            doc.metadata["source"] = uploaded_file.name
            
            # add file size in characters - useful for relevance scoring
            doc.metadata["size"] = len(doc.page_content)
            
            # add processing timestamp
            doc.metadata["processed_at"] = time.time()
        
        # create text splitter for chunking
        # recursivecharactersplitter tries to split on natural boundaries like paragraphs
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,        # size of each chunk in characters
            chunk_overlap=chunk_overlap,  # overlap between consecutive chunks
            separators=["\n\n", "\n", ". ", " ", ""]  # preferred split points in order
            # tries to split on double newline (paragraphs) first, then single newline, etc.
        )
        
        # split documents into chunks
        chunks = splitter.split_documents(raw_docs)
        
        # add chunk-specific metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i  # position in document
            chunk.metadata["total_chunks"] = len(chunks)  # total chunks for this document
        
        return chunks  # return the processed and tagged chunks
        
    finally:
        # clean up temporary file
        # this runs even if an error occurs during processing
        if os.path.exists(temp_path):
            os.unlink(temp_path)  # delete the temporary file
    
    # note: processing time could be logged here for performance monitoring
    # elapsed_time = time.time() - start_time


def process_multiple_files(
    uploaded_files: List,  # list of streamlit uploadedfile objects
    content_types: List[str],  # parallel list of content types
    chunk_size: int = None,
    chunk_overlap: int = None
) -> List[Document]:
    """
    Process multiple uploaded files and combine their chunks.
    
    Args:
        uploaded_files (List): List of uploaded file objects
        content_types (List[str]): List of content types corresponding to files
        chunk_size (int, optional): Custom chunk size
        chunk_overlap (int, optional): Custom chunk overlap
    
    Returns:
        List[Document]: Combined list of document chunks from all files
    """
    all_chunks = []  # initialize empty list for all chunks
    
    # process each file individually
    for uploaded_file, content_type in zip(uploaded_files, content_types):
        # zip pairs files with their content types for parallel iteration
        try:
            # process individual file
            chunks = process_uploaded_file(
                uploaded_file,
                content_type,
                chunk_size,
                chunk_overlap
            )
            all_chunks.extend(chunks)  # add chunks to combined list
            
        except Exception as e:
            # log error but continue processing other files
            print(f"Error processing {uploaded_file.name}: {str(e)}")
            # in production, this would use proper logging
            continue
    
    return all_chunks  # return combined chunks from all files


def extract_document_metadata(filepath: str) -> Dict[str, Any]:
    """
    Extract metadata from document without full processing.
    useful for previewing documents before full ingestion.
    
    Args:
        filepath (str): Path to the document
    
    Returns:
        Dict[str, Any]: Document metadata including size, pages, etc.
    """
    metadata = {
        "filename": os.path.basename(filepath),  # extract filename from path
        "extension": get_file_extension(filepath),  # get file extension
        "size_bytes": os.path.getsize(filepath),  # get file size in bytes
        "size_mb": os.path.getsize(filepath) / (1024 * 1024),  # convert to megabytes
        "supported": is_supported_file(filepath)  # check if format is supported
    }
    
    # add format-specific metadata if possible
    if metadata["extension"] == ".pdf":
        try:
            import PyPDF2
            with open(filepath, 'rb') as f:
                pdf = PyPDF2.PdfReader(f)
                metadata["pages"] = len(pdf.pages)  # number of pages in pdf
                # try to extract pdf metadata if available
                if pdf.metadata:
                    metadata["title"] = pdf.metadata.get('/Title', 'Unknown')
                    metadata["author"] = pdf.metadata.get('/Author', 'Unknown')
        except Exception:
            # if pdf parsing fails, just continue with basic metadata
            pass
    
    return metadata  # return extracted metadata dictionary


def validate_document_content(chunks: List[Document]) -> bool:
    """
    Validate that document chunks contain meaningful content.
    filters out empty or near-empty chunks that won't be useful.
    
    Args:
        chunks (List[Document]): List of document chunks to validate
    
    Returns:
        bool: True if content appears valid, False otherwise
    """
    if not chunks:  # check if chunks list is empty
        return False
    
    # check each chunk for minimum content
    for chunk in chunks:
        content = chunk.page_content.strip()  # remove whitespace
        # check if content has at least 10 meaningful characters
        # this filters out chunks that are just whitespace or very short
        if len(content) >= 10:
            return True  # found at least one valid chunk
    
    return False  # no valid chunks found


def get_document_stats(chunks: List[Document]) -> Dict[str, Any]:
    """
    Calculate statistics about processed document chunks.
    
    Args:
        chunks (List[Document]): List of processed document chunks
    
    Returns:
        Dict[str, Any]: Statistics about the document collection
    """
    stats = {
        "total_chunks": len(chunks),  # total number of chunks
        "total_characters": sum(len(chunk.page_content) for chunk in chunks),  # sum of all content lengths
        "avg_chunk_size": 0,  # will calculate below
        "content_types": {},  # distribution by content type
        "sources": {}  # distribution by source file
    }
    
    if chunks:
        # calculate average chunk size
        stats["avg_chunk_size"] = stats["total_characters"] / len(chunks)
        
        # count chunks by content type
        for chunk in chunks:
            # get content type from metadata (default to "unknown" if missing)
            ctype = chunk.metadata.get("content_type", "unknown")
            stats["content_types"][ctype] = stats["content_types"].get(ctype, 0) + 1
            
            # count chunks by source file
            source = chunk.metadata.get("source", "unknown")
            stats["sources"][source] = stats["sources"].get(source, 0) + 1
    
    return stats  # return calculated statistics


def clean_document_text(text: str) -> str:
    """
    Clean and normalize extracted document text.
    removes excessive whitespace and normalizes line endings.
    
    Args:
        text (str): Raw extracted text
    
    Returns:
        str: Cleaned and normalized text
    """
    # replace multiple newlines with double newline (paragraph break)
    import re
    
    # replace 3 or more newlines with 2 newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # replace multiple spaces with single space
    text = re.sub(r' {2,}', ' ', text)
    
    # remove spaces at beginning and end of lines
    lines = text.split('\n')
    lines = [line.strip() for line in lines]
    
    # join back with newlines
    text = '\n'.join(lines)
    
    # remove any non-printable characters
    text = ''.join(char for char in text if char.isprintable() or char in '\n\t')
    
    return text.strip()  # return cleaned text


def estimate_processing_time(file_size_mb: float) -> float:
    """
    Estimate processing time based on file size.
    helps provide user feedback for large uploads.
    
    Args:
        file_size_mb (float): File size in megabytes
    
    Returns:
        float: Estimated processing time in seconds
    """
    # rough estimate: ~2 seconds per MB for PDF processing
    # this is a heuristic based on typical performance
    base_time = file_size_mb * 2.0
    
    # add overhead for small files
    overhead = 3.0
    
    # cap at reasonable maximum
    return min(base_time + overhead, 60.0)  # max 60 seconds estimate