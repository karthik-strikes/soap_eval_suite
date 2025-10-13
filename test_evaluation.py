#!/usr/bin/env python3
"""
Simple test script to evaluate SOAP notes using the DeepScribe system
"""

import asyncio
import json
from pathlib import Path

# Add the project root to Python path
import sys
sys.path.append(str(Path(__file__).parent))

from data.loader import UniversalDataLoader, DSPyFieldDetector
from core.integration import SimpleSOAPIntegration
from utils.model_setup import setup_dspy_model

async def test_simple_evaluation():
    """Test the evaluation system with simple synthetic data."""
    
    print("🚀 Starting DeepScribe SOAP Evaluation Test")
    print("=" * 50)
    
    # 1. Setup DSPy model (using Gemini as default)
    print("📋 Setting up DSPy model...")
    model_name = "gemini/gemini-2.5-pro"
    success = setup_dspy_model(model_name, max_tokens=4000, temperature=0.1)
    
    if not success:
        print("❌ Failed to setup DSPy model. Check your API key.")
        return
    
    print(f"✅ DSPy model configured: {model_name}")
    
    # 2. Initialize components
    print("\n📋 Initializing evaluation components...")
    detector = DSPyFieldDetector()
    loader = UniversalDataLoader(detector)
    
    # 3. Test with single example first
    print("\n📋 Testing with single example...")
    
    try:
        # Load single test case
        with open("test_single_example.json", "r") as f:
            single_test = json.load(f)
        
        # Create integration pipeline for evaluation only
        integration = SimpleSOAPIntegration(
            soap_engine="dspy",
            evaluation_mode="deterministic",  # Start with fast evaluation
            storage_mode="both",
            storage_file="results/test_results_single.jsonl",
            batch_size=1
        )
        
        print("✅ Integration pipeline ready")
        
        # Prepare data in the expected format
        test_data = [{
            'transcript': single_test['patient_convo'],
            'reference_notes': single_test['soap_notes'],
            'patient_metadata': single_test['patient_metadata'],
            'ground_truth': single_test['soap_notes']  # Using same as reference for this test
        }]
        
        print("\n🔄 Running evaluation...")
        
        # Run evaluation only (not generation)
        results = await integration.process_evaluation_only_async(test_data, "test_single")
        
        if results:
            result = results[0]
            print("\n✅ Evaluation Complete!")
            print("-" * 30)
            
            # Display key metrics
            if 'evaluation_metrics' in result:
                metrics = result['evaluation_metrics']
                
                if 'deterministic_metrics' in metrics:
                    det = metrics['deterministic_metrics']
                    print(f"📊 Entity Coverage: {det.get('entity_coverage', 0):.1f}%")
                    print(f"📊 Section Completeness: {det.get('section_completeness', 0):.1f}%")
                    print(f"📊 Format Validity: {det.get('format_validity', 0):.1f}%")
                
                if 'missing_entities' in metrics.get('details', {}):
                    missing = metrics['details']['missing_entities']
                    if missing:
                        print(f"⚠️  Missing Entities: {missing[:3]}...")  # Show first 3
                    else:
                        print("✅ No missing entities detected")
            
            print(f"\n💾 Results saved to: results/test_results_single.jsonl")
        
    except Exception as e:
        print(f"❌ Single test failed: {e}")
        return
    
    # 4. Test with multiple examples
    print("\n" + "=" * 50)
    print("📋 Testing with multiple examples...")
    
    try:
        # Test with CSV data
        normalized_data, field_mapping = await loader.load_and_normalize(
            source="test_data_csv_format.csv",
            max_samples=3
        )
        
        print(f"✅ Loaded {len(normalized_data)} samples from CSV")
        print(f"📋 Field mapping confidence: {field_mapping.confidence_score:.2f}")
        
        # Create integration for comprehensive evaluation
        integration_full = SimpleSOAPIntegration(
            soap_engine="dspy",
            evaluation_mode="comprehensive",  # Full evaluation with LLM
            storage_mode="both",
            storage_file="results/test_results_batch.jsonl",
            batch_size=3
        )
        
        print("\n🔄 Running comprehensive evaluation (this may take 30-60 seconds)...")
        
        # Run evaluation
        results = await integration_full.process_evaluation_only_async(normalized_data, "test_batch")
        
        print(f"\n✅ Batch Evaluation Complete! Processed {len(results)} samples")
        print("-" * 40)
        
        # Summary statistics
        if results:
            entity_scores = []
            completeness_scores = []
            
            for result in results:
                if 'evaluation_metrics' in result:
                    metrics = result['evaluation_metrics']
                    if 'deterministic_metrics' in metrics:
                        det = metrics['deterministic_metrics']
                        entity_scores.append(det.get('entity_coverage', 0))
                        completeness_scores.append(det.get('section_completeness', 0))
            
            if entity_scores:
                avg_entity = sum(entity_scores) / len(entity_scores)
                avg_completeness = sum(completeness_scores) / len(completeness_scores)
                
                print(f"📊 Average Entity Coverage: {avg_entity:.1f}%")
                print(f"📊 Average Section Completeness: {avg_completeness:.1f}%")
                print(f"📊 Success Rate: {len([r for r in results if 'error' not in r])}/{len(results)} ({100*len([r for r in results if 'error' not in r])/len(results):.1f}%)")
        
        print(f"\n💾 Results saved to: results/test_results_batch.jsonl")
        
    except Exception as e:
        print(f"❌ Batch test failed: {e}")
        return
    
    print("\n" + "=" * 50)
    print("🎉 All tests completed successfully!")
    print("\nNext steps:")
    print("1. Check the results files in the 'results/' directory")
    print("2. Try running with --auto-dashboard to see visual results")
    print("3. Experiment with different evaluation modes:")
    print("   - 'deterministic': Fast rule-based evaluation")
    print("   - 'llm_only': Deep LLM analysis (slower)")
    print("   - 'comprehensive': Both deterministic and LLM")

def main():
    """Main test function."""
    try:
        asyncio.run(test_simple_evaluation())
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have set your API key (GEMINI_API_KEY)")
        print("2. Check that all dependencies are installed: pip install -r requirements.txt")
        print("3. Ensure you're running from the project root directory")

if __name__ == "__main__":
    main()