#!/usr/bin/env python3
"""Test script to invoke the summarizer and identify issues."""

import asyncio
from src.langgraph_summarizer import graph

async def test_summarizer():
    """Test the summarizer with a sample file."""
    print("Testing summarizer with file path...")

    # Test with a simple text input
    test_input = {
        "file_path": r"your_test_file.txt",
        "chunk_size": 2000,
        "chunk_overlap": 500,
    }

    try:
        print("Invoking graph...")
        result = await graph.ainvoke(test_input)
        print("\n" + "="*80)
        print("SUCCESS!")
        print("="*80)
        print(f"\nFinal Summary:\n{result.get('final_summary', 'No summary found')}")

    except Exception as e:
        print("\n" + "="*80)
        print("ERROR OCCURRED!")
        print("="*80)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_summarizer())
