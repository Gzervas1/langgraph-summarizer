from src.logger import get_logger
import os
import asyncio
import docx
import pypandoc

logger = get_logger("file_loader")

async def read_text_content(file_path):
    """
    Opens, reads, and returns the text content of a file (DOC, DOCX, RTF, TXT, ODT, Markdown, Plain text).
    
    Args:
        file_path (str): Path to the input file.
    
    Returns:
        str: The text content of the file.
    
    Raises:
        ValueError: If the file format is unsupported.
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(file_path):
        logger.error(f"The file '{file_path}' does not exist.")
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    
    file_extension = os.path.splitext(file_path)[1].lower()
    logger.info(f"Reading file '{file_path}' with extension '{file_extension}'.")

    if file_extension == ".txt":
        logger.debug("Opening TXT file.")
        return await asyncio.to_thread(
            lambda: open(file_path, "r", encoding="utf-8").read()
        )
    elif file_extension == ".docx":
        logger.debug("Opening DOCX file.")
        def read_docx():
            doc = docx.Document(file_path)
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        content = await asyncio.to_thread(read_docx)
        logger.info("Successfully read DOCX file.")
        return content
    
    elif file_extension == ".pdf":   
        logger.error("PDF Files not currently supported")
        raise ValueError("PDF Files not currently supported")
            
    elif file_extension in [".rtf", ".odt", ".md"]:
        logger.debug(f"Converting {file_extension} file to plain text using pypandoc.")
        content = await asyncio.to_thread(
            lambda: pypandoc.convert_file(file_path, "plain")
        )
        logger.info(f"Successfully converted {file_extension} file to plain text.")
        return content
    else:
        logger.error(f"Unsupported file format: {file_extension}")
        raise ValueError(f"Unsupported file format: {file_extension}")