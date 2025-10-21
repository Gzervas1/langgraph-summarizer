"""Main entrypoint for the Document Summarization implementation.

This module defines the core structure and functionality of the document
summarization graph using Map-Reduce pattern. It includes the main graph definition,
state management, and key functions for processing documents, chunking text,
generating summaries, and producing final consolidated summaries.
"""

from typing import Dict, List, Literal, Optional, Annotated, TypedDict
from datetime import datetime, timezone
import operator
import asyncio
import base64

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
# Use a local helper for splitting lists of Documents to avoid import resolution issues.
# The original `split_list_of_docs` from langchain is replaced by the implementation
# below (defined later in this module).
from pydantic import BaseModel
from dotenv import load_dotenv

from src.prompts import (
    LANGGRAPH_SUMMMARIZER_SYSTEM_PROMPT, 
    LANGGRAPH_SUMMARIZER_COMBINATION_PROMPT, 
    LANGGRAPH_SUMMARIZER_PROMPT
)
from src.utils import get_document_bytes, detect_file_type, process_docx_bytes, process_text_bytes
from src.logger import get_logger
from src.file_loader import read_text_content
from src.chunker import chunk_text
from src.llm import get_llm

# Load environment and setup
load_dotenv()
logger = get_logger("langgraph_summarizer")
llm = get_llm()


class InputState(TypedDict):
    """Input state for document processing."""
    file_path: Optional[str]
    file_bytes: Optional[str]  # Base64 encoded bytes
    chunk_size: Optional[int]
    chunk_overlap: Optional[int]


class State(InputState):
    """Complete state for the legal document summarization agent."""
    # Processing state
    file_content: str
    chunks: List[str]
    
    # Summarization state  
    summaries: Annotated[List[str], operator.add]
    collapsed_summaries: List[Document]
    final_summary: str
    
    # Configuration
    token_max: int

class ChunkState(TypedDict):
    """State for individual chunk processing."""
    content: str

class OutputState(TypedDict):
    """Output state for the summarization process."""
    final_summary: str


async def load_file(state: State) -> Dict[str, str]:
    """Load document content from file path or bytes.
    
    This function handles both file path and byte-based document loading,
    using the appropriate processing method based on file type detection.
    
    Args:
        state (State): The current state containing file information.
        config (RunnableConfig): Configuration for the loading process.
        
    Returns:
        Dict[str, str]: Dictionary containing the loaded file content.
    """
    try:
        if state.get("file_path"):
            logger.info(f"Loading file: {state['file_path']}")
            file_content = await read_text_content(state["file_path"])
            
        elif state.get("file_bytes"):
            logger.info("Loading file content from bytes")
            
            # Decode the base64 bytes
            file_bytes = get_document_bytes(state['file_bytes'])
            
            # Detect file type
            file_type = detect_file_type(file_bytes)
            logger.info(f"Detected file type: {file_type}")
            
            if file_type == 'docx':
                logger.info("Processing DOCX file from bytes")
                file_content = await process_docx_bytes(file_bytes)
                
            elif file_type == 'txt':
                logger.info("Processing text file from bytes")
                file_content = await process_text_bytes(file_bytes)

            else:
                logger.error(f"Unsupported file type: {file_type}")
                raise ValueError(f"Unsupported file type: {file_type}")
            
        else:
            raise ValueError("No file path or file bytes provided")
            
        logger.info(f"File loaded successfully. Content length: {len(file_content)}")
        return {"file_content": file_content}
        
    except Exception as e:
        logger.error(f"Failed to load file: {e}")
        raise RuntimeError(f"Failed to load file: {e}")


async def chunk_file(state: State) -> Dict[str, List[str]]:
    """Split document content into manageable chunks for processing.
    
    This function uses configurable chunk size and overlap parameters
    to split the document into processable segments.
    
    Args:
        state (State): The current state containing file content and chunking parameters.
        config (RunnableConfig): Configuration for the chunking process.
        
    Returns:
        Dict[str, List[str]]: Dictionary containing the list of document chunks.
    """
    try:
        logger.info("Chunking file content")
        
        chunk_size = state.get("chunk_size", 2000)
        chunk_overlap = state.get("chunk_overlap", 500)
        
        chunks = await chunk_text(
            state["file_content"], 
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        
        logger.info(f"Chunked into {len(chunks)} chunks")
        return {"chunks": chunks}
        
    except Exception as e:
        logger.error(f"Failed to chunk file: {e}")
        raise RuntimeError(f"Failed to chunk file: {e}")


async def map_summaries(state: State) -> List[Send]:
    """Map chunks to parallel summary generation tasks.
    
    This function creates Send operations for each chunk to enable
    parallel processing of document segments.
    
    Args:
        state (State): The current state containing document chunks.
        config (RunnableConfig): Configuration for the mapping process.
        
    Returns:
        List[Send]: List of Send operations for parallel chunk processing.
    """
    try:
        logger.info(f"Mapping {len(state['chunks'])} contents to summary generation")
        return [
            Send("generate_summary", {"content": content}) 
            for content in state["chunks"]
        ]
    except Exception as e:
        logger.error(f"Failed to map summaries: {e}")
        raise RuntimeError(f"Failed to map summaries: {e}")


async def generate_summary(state: ChunkState) -> Dict[str, List[str]]:
    """Generate summary for a single document chunk.
    
    This function processes individual chunks using the map chain
    to create focused summaries of document segments.
    
    Args:
        state (ChunkState): The state containing chunk content.
        config (RunnableConfig): Configuration for the summary generation.
        
    Returns:
        Dict[str, List[str]]: Dictionary containing the generated summary.
    """
    try:
        logger.debug("Generating summary for a document chunk")
        
        # Build map chain
        prompt = ChatPromptTemplate([
            ("system", LANGGRAPH_SUMMMARIZER_SYSTEM_PROMPT),
            ("human", LANGGRAPH_SUMMARIZER_PROMPT),
        ])
        map_chain = prompt | llm | StrOutputParser()
        
        response = await map_chain.ainvoke({"text": state["content"]})
        logger.debug("Summary generated successfully")
        
        return {"summaries": [response]}
        
    except Exception as e:
        logger.error(f"Failed to generate summary: {e}")
        return {"summaries": [f"Error processing chunk: {str(e)}"]}


async def collect_summaries(state: State) -> Dict[str, List[Document]]:
    """Collect all chunk summaries into Document objects.
    
    This function aggregates individual chunk summaries and converts
    them into Document objects for further processing.
    
    Args:
        state (State): The current state containing all chunk summaries.
        config (RunnableConfig): Configuration for the collection process.
        
    Returns:
        Dict[str, List[Document]]: Dictionary containing collected summary documents.
    """
    try:
        logger.info(f"Collecting {len(state['summaries'])} summaries into collapsed_summaries")
        
        valid_summaries = []
        for summary in state.get("summaries", []):
            if summary and isinstance(summary, str) and summary.strip():
                valid_summaries.append(Document(page_content=summary.strip()))
            elif summary is not None:
                logger.warning(f"Invalid summary type or content: {type(summary)}")
        
        logger.info(f"Created {len(valid_summaries)} valid documents from summaries")
        return {"collapsed_summaries": valid_summaries}
        
    except Exception as e:
        logger.error(f"Failed to collect summaries: {e}")
        raise RuntimeError(f"Failed to collect summaries: {e}")
    
def length_function(documents: List[Document]) -> int:
    """Calculate total character length for documents.
    
    This function provides a simple token estimation based on character count
    for determining if summary collapse is needed.
    
    Args:
        documents (List[Document]): List of documents to measure.
        
    Returns:
        int: Total character count across all documents.
    """
    try:
        total_chars = 0
        for doc in documents:
            if doc and hasattr(doc, 'page_content') and doc.page_content:
                total_chars += len(str(doc.page_content))
        return total_chars
    except Exception as e:
        logger.error(f"Failed to calculate token length: {e}")
        raise RuntimeError(f"Failed to calculate token length: {e}")


def split_list_of_docs(
    documents: List[Document], 
    length_fn, 
    max_tokens: int
) -> List[List[Document]]:
    """
    Split a list of Document objects into sublists where each sublist's
    combined length (as computed by length_fn) does not exceed max_tokens,
    using a simple greedy batching approach.

    Args:
        documents: List of Document objects to split.
        length_fn: Callable that accepts a list of Document and returns an int length.
        max_tokens: Maximum allowed tokens/characters per batch.

    Returns:
        List of Document sublists.
    """
    if not documents:
        return []

    batches: List[List[Document]] = []
    current_batch: List[Document] = []
    for doc in documents:
        # If current batch is empty, start with this doc regardless of size.
        if not current_batch:
            current_batch.append(doc)
            continue

        # Check if adding this doc would exceed max_tokens.
        tentative_batch = current_batch + [doc]
        if length_fn(tentative_batch) <= max_tokens:
            current_batch.append(doc)
        else:
            # Close current batch and start a new one with the current doc.
            batches.append(current_batch)
            current_batch = [doc]

    # Append the final batch if present.
    if current_batch:
        batches.append(current_batch)

    return batches


async def should_collapse(state: State) -> Literal["collapse_summaries", "generate_final_summary"]:
    """Determine if summaries need collapsing based on token count.
    
    This function evaluates whether the current summaries exceed token limits
    and need to be collapsed before final summary generation.
    
    Args:
        state (State): The current state containing summary documents.
        config (RunnableConfig): Configuration for the decision process.
        
    Returns:
        Literal: The next node to route to based on token count.
    """
    try:
        collapsed_summaries = state.get("collapsed_summaries", [])
        if not collapsed_summaries:
            logger.warning("No collapsed_summaries found. Proceeding to final summary")
            return "generate_final_summary"
        
        token_max = state.get("token_max", 10000)
        num_tokens = length_function(collapsed_summaries)
        
        logger.info(f"Checking if collapsed_summaries ({num_tokens} chars) exceed token_max ({token_max})")
        
        if num_tokens > token_max:
            logger.info("Token count exceeds token_max. Collapsing further")
            return "collapse_summaries"
        else:
            logger.info("Token count within limit. Proceeding to generate final summary")
            return "generate_final_summary"
            
    except Exception as e:
        logger.error(f"Failed to determine if should collapse: {e}")
        raise RuntimeError(f"Failed to determine if should collapse: {e}")



async def collapse_summaries(state: State) -> Dict[str, List[Document]]:
    """Collapse summaries to reduce token count.
    
    This function reduces the number of summary documents by combining
    them when token limits are exceeded.
    
    Args:
        state (State): The current state containing summary documents.
        config (RunnableConfig): Configuration for the collapse process.
        
    Returns:
        Dict[str, List[Document]]: Dictionary containing collapsed summary documents.
    """
    try:
        logger.info("Collapsing summaries to fit within token limits")
        
        # Build reduce chain
        prompt = ChatPromptTemplate([
            ("system", LANGGRAPH_SUMMMARIZER_SYSTEM_PROMPT),
            ("human", LANGGRAPH_SUMMARIZER_COMBINATION_PROMPT),
        ])
        reduce_chain = prompt | llm | StrOutputParser()
        
        token_max = state.get("token_max", 10000)
        doc_lists = split_list_of_docs(
            state["collapsed_summaries"], 
            length_function, 
            token_max
        )
        
        logger.debug(f"Split collapsed_summaries into {len(doc_lists)} lists")
        
        results = []
        for idx, doc_list in enumerate(doc_lists):
            try:
                logger.debug(f"Collapsing doc list {idx+1}/{len(doc_lists)}")
                
                docs_content = "\n\n".join(
                    doc.page_content for doc in doc_list if doc.page_content
                )
                
                collapsed_result = await reduce_chain.ainvoke({"docs": docs_content})
                results.append(Document(page_content=collapsed_result))
                
                logger.debug(f"Doc list {idx+1} collapsed successfully")
                
            except Exception as e:
                logger.error(f"Failed to collapse a doc list: {e}")
                raise RuntimeError(f"Failed to collapse a doc list: {e}")
        
        logger.info("All doc lists collapsed successfully")
        return {"collapsed_summaries": results}
        
    except Exception as e:
        logger.error(f"Failed to collapse summaries: {e}")
        raise RuntimeError(f"Failed to collapse summaries: {e}")


async def generate_final_summary(state: State) -> Dict[str, str]:
    """Generate the final consolidated summary.
    
    This function creates the final summary by combining all processed
    summary documents into a cohesive legal document summary.
    
    Args:
        state (State): The current state containing all summary documents.
        config (RunnableConfig): Configuration for the final generation process.
        
    Returns:
        Dict[str, str]: Dictionary containing the final summary.
    """
    try:
        logger.info("Generating final summary from collapsed summaries")
        
        # Build reduce chain
        prompt = ChatPromptTemplate([
            ("system", LANGGRAPH_SUMMMARIZER_SYSTEM_PROMPT),
            ("human", LANGGRAPH_SUMMARIZER_COMBINATION_PROMPT),
        ])
        reduce_chain = prompt | llm | StrOutputParser()
        
        docs_content = "\n\n".join(
            doc.page_content for doc in state["collapsed_summaries"] 
            if doc.page_content
        )
        
        response = await reduce_chain.ainvoke({"docs": docs_content})
        
        logger.info("Final summary generated successfully")
        logger.debug(f"Final summary content length: {len(response)}")
        
        return {"final_summary": response}
        
    except Exception as e:
        logger.error(f"Failed to generate final summary: {e}")
        raise RuntimeError(f"Failed to generate final summary: {e}")


# Define the graph
builder = StateGraph(State, input=InputState, output=OutputState)

# Add nodes
builder.add_node("load_file", load_file)
builder.add_node("chunk_file", chunk_file)
builder.add_node("generate_summary", generate_summary)
builder.add_node("collect_summaries", collect_summaries)
builder.add_node("collapse_summaries", collapse_summaries)
builder.add_node("generate_final_summary", generate_final_summary)

# Add edges with conditional routing
builder.add_edge(START, "load_file")
builder.add_edge("load_file", "chunk_file")
builder.add_conditional_edges("chunk_file", map_summaries, ["generate_summary"])
builder.add_edge("generate_summary", "collect_summaries")
builder.add_conditional_edges("collect_summaries", should_collapse)
builder.add_conditional_edges("collapse_summaries", should_collapse)
builder.add_edge("generate_final_summary", END)

# Compile the graph
graph = builder.compile(
    interrupt_before=[],  # Add nodes here if you want to update state before execution
    interrupt_after=[],   # Add nodes here if you want to update state after execution
)
graph.name = "DocumentSummarizationGraph"