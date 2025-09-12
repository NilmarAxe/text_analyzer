#!/usr/bin/env python3
"""
Strategic analysis execution script.
Convenience script for running text analysis with common configurations.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.text_analyzer import TextAnalyzer


def main():
    """Main execution function for analysis script."""
    parser = argparse.ArgumentParser(
        description="Strategic Text Analysis - Quick execution script"
    )
    
    parser.add_argument(
        "input_file",
        help="Input text file to analyze"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output file path"
    )
    
    parser.add_argument(
        "--format", "-f",
        choices=["txt", "json", "csv", "html"],
        default="txt",
        help="Output format (default: txt)"
    )
    
    parser.add_argument(
        "--top-count", "-n",
        type=int,
        default=10,
        help="Number of top words to display (default: 10)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    try:
        # Create analyzer with configuration
        config = {
            'top_words_count': args.top_count,
            'detailed_analysis': args.verbose
        }
        
        with TextAnalyzer(config) as analyzer:
            print(f"Analyzing: {args.input_file}")
            
            # Execute analysis
            result = analyzer.analyze_file(
                args.input_file,
                args.output,
                args.format
            )
            
            if result.success:
                print(f"✓ Analysis completed successfully!")
                print(f"  Total words: {result.analysis_metrics['word_statistics']['total_words']:,}")
                print(f"  Unique words: {result.analysis_metrics['word_statistics']['unique_words']:,}")
                print(f"  Processing time: {result.execution_time:.3f}s")
                
                if args.output:
                    print(f"  Report saved: {args.output}")
                
                return 0
            else:
                print(f"✗ Analysis failed: {result.error_message}")
                return 1
                
    except Exception as e:
        print(f"✗ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())