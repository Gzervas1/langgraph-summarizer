from langchain_text_splitters import RecursiveCharacterTextSplitter
import asyncio
from src.logger import get_logger

logger = get_logger("text-chunker")

def chunk_text_sync(text, chunk_size=2000, chunk_overlap=500):
    """
    Splits the input text into smaller chunks of specified size with overlap.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The size of each chunk.
        chunk_overlap (int): The number of overlapping characters between consecutive chunks.

    Returns:
        list: A list of text chunks.
    """
    logger.debug("Starting chunk_text function.")
    if not isinstance(text, str):
        logger.error("Input must be a string.")
        raise ValueError("Input must be a string.")
    
    if not isinstance(chunk_size, int) or not isinstance(chunk_overlap, int):
        logger.error("Chunk size and overlap must be integers.")
        raise ValueError("Chunk size and overlap must be integers.")
    
    if chunk_size <= 0 or chunk_overlap < 0:
        logger.error("Chunk size must be positive and overlap must be non-negative.")
        raise ValueError("Chunk size must be positive and overlap must be non-negative.")
    
    if chunk_overlap >= chunk_size:
        logger.error("Overlap must be less than chunk size.")
        raise ValueError("Overlap must be less than chunk size.")
    
    total_words = len(text.split())
    logger.info(f"Total words in document: {total_words}")  

    logger.debug(
        f"Initializing RecursiveCharacterTextSplitter with chunk_size={chunk_size}, "
        f"chunk_overlap={chunk_overlap}, encoding_name='gpt-4o', model_name='gpt-4o'."
    )

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="gpt-4o",
        model_name="gpt-4o",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    logger.debug("Splitting text into chunks.")
    split_docs = text_splitter.create_documents([text])
    num_docs = len(split_docs)
    total_tokens = sum(len(doc.page_content.split()) for doc in split_docs)
    logger.info(f"Chunk size: {chunk_size} -> {num_docs} chunks, {total_tokens} words total")
    logger.debug("chunk_text function completed successfully.")

    return split_docs


async def chunk_text(text, chunk_size=20000, chunk_overlap=2000):
    """
    Async wrapper for chunk_text_sync that runs the entire blocking operation in a separate thread.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The size of each chunk.
        chunk_overlap (int): The number of overlapping characters between consecutive chunks.

    Returns:
        list: A list of text chunks.
    """
    # Run the entire synchronous chunking operation in a separate thread
    return await asyncio.to_thread(chunk_text_sync, text, chunk_size, chunk_overlap)