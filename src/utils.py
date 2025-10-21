import base64
import asyncio
import io
import docx
from src.logger import get_logger
logger = get_logger("utils")

def get_document_bytes(text: str) -> bytes:
    """Helper function to decode base64 encoded document bytes."""
    return base64.b64decode(text)


def detect_file_type(file_bytes: bytes) -> str:
    """Detect file type from byte content."""
    # PDF signature
    if file_bytes.startswith(b'%PDF'):
        return 'pdf'
    
    # DOCX signature (ZIP with specific structure)
    if (file_bytes.startswith(b'PK\x03\x04') and 
        b'word/' in file_bytes[:2048] or b'[Content_Types].xml' in file_bytes[:2048]):
        return 'docx'
    
    # DOC signature
    if (file_bytes.startswith(b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1') or
        file_bytes.startswith(b'\xdb\xa5-\x00\x00\x00')):
        return 'doc'
    
    # RTF signature
    if file_bytes.startswith(b'{\\rtf'):
        return 'rtf'
    
    # Try to decode as text
    try:
        file_bytes.decode('utf-8')
        return 'txt'
    except UnicodeDecodeError:
        pass
    
    # Default to PDF for unknown binary files
    return 'pdf'

async def process_docx_bytes(file_bytes: bytes) -> str:
    """Process DOCX file from bytes."""
    
    def read_docx_from_bytes():
        file_like = io.BytesIO(file_bytes)
        doc = docx.Document(file_like)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    
    return await asyncio.to_thread(read_docx_from_bytes)


async def process_text_bytes(file_bytes: bytes) -> str:
    """Process text file from bytes."""
    def decode_text():
        try:
            return file_bytes.decode('utf-8')
        except UnicodeDecodeError:
            return file_bytes.decode('latin-1')
    
    return await asyncio.to_thread(decode_text)


