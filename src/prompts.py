LANGGRAPH_SUMMMARIZER_SYSTEM_PROMPT = '''
You are an expert summarization assistant. 
Your task is to generate concise and informative summaries of provided documents, ensuring that all critical information is retained while unnecessary details are omitted.
'''

LANGGRAPH_SUMMARIZER_COMBINATION_PROMPT = '''
You are provided with multiple text segments that are parts of the summary of a document.

Documents: 
"""
{docs}
"""

## Instructions for Summary Compilation:
- Combine the provided text segments into a single, coherent summary.
- Ensure that the final summary is well-structured, clear, and concise.
- Maintain the original meaning and key points from the segments.
- Avoid redundancy and ensure smooth transitions between different parts of the summary.
- The final summary should be at least 500 words in length.
'''

LANGGRAPH_SUMMARIZER_PROMPT = '''
You are provided with a text segment from a document.
Document Segment:
"""
{text}
"""

## Instructions for Summarization:
- Read the provided text segment carefully.
- Generate a concise summary that captures the main ideas and key points of the segment.
- Ensure that the summary is clear, coherent, and free of unnecessary details.  
- The summary should be approximately 200-300 words in length.
- Use formal language and maintain a neutral tone.
'''
