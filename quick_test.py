#!/usr/bin/env python3
"""
Quick test of the SOAP evaluation system with minimal setup
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def quick_test():
    """Run a quick test using the main.py CLI interface."""
    
    print("🚀 Quick Test of DeepScribe SOAP Evaluation System")
    print("=" * 55)
    
    # Check if API key is set
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("⚠️  No API key found!")
        print("Please set one of the following environment variables:")
        print("   export GEMINI_API_KEY='your-gemini-api-key'")
        print("   export OPENAI_API_KEY='your-openai-api-key'")
        print("\nYou can get a free Gemini API key at: https://makersuite.google.com/app/apikey")
        return
    
    # Test commands to run
    test_commands = [
        {
            "name": "Test 1: Single JSON file (Fast)",
            "command": f"python main.py --source test_single_example.json --samples 1 --evaluation-mode deterministic --output results/quick_test_1.jsonl",
            "description": "Tests basic functionality with one example using fast deterministic evaluation"
        },
        {
            "name": "Test 2: CSV file (Medium)",
            "command": f"python main.py --source test_data_csv_format.csv --samples 3 --evaluation-mode deterministic --output results/quick_test_2.jsonl",
            "description": "Tests CSV loading and batch processing with 3 examples"
        },
        {
            "name": "Test 3: Comprehensive evaluation (Slow)",
            "command": f"python main.py --source test_single_example.json --samples 1 --evaluation-mode comprehensive --output results/quick_test_3.jsonl",
            "description": "Tests full LLM evaluation (takes 30-60 seconds)"
        }
    ]
    
    print("Available test commands:")
    print("-" * 25)
    
    for i, test in enumerate(test_commands, 1):
        print(f"\n{i}. {test['name']}")
        print(f"   Description: {test['description']}")
        print(f"   Command: {test['command']}")
    
    print(f"\n4. Dashboard Test")
    print(f"   Description: Generate interactive dashboard from results")
    print(f"   Command: python main.py --dashboard results/quick_test_*.jsonl --open")
    
    print("\n" + "=" * 55)
    print("To run a test, copy and paste the command above.")
    print("Or run them all in sequence:")
    print("\nfor i in {1..3}; do echo \"Running Test $i...\"; done")
    
    # Show expected output format
    print("\n📋 Expected Output Format:")
    print("-" * 25)
    print("""
{
  "conversation": "Doctor: Hello, what brings you in today? Patient: ...",
  "referenced_soap": "SUBJECTIVE: Patient reports...",
  "evaluation_metrics": {
    "deterministic_metrics": {
      "entity_coverage": 85.0,
      "section_completeness": 100.0,
      "format_validity": 95.0
    },
    "llm_metrics": {
      "content_fidelity": {"f1": 0.82, "precision": 0.89, "recall": 0.76},
      "medical_correctness": {"accuracy": 0.91}
    }
  },
  "compared_on": "ground_truth",
  "source_name": "test_file"
}
    """)

if __name__ == "__main__":
    quick_test()