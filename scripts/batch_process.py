#!/usr/bin/env python3
"""
Strategic batch processing script.
Efficient batch analysis with progress tracking and comprehensive reporting.
"""

import sys
import glob
import argparse
from pathlib import Path
from typing import List

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.text_analyzer import TextAnalyzer


def collect_files(patterns: List[str], recursive: bool = False) -> List[Path]:
    """Collect files from patterns with optional recursion."""
    files = []
    
    for pattern in patterns:
        if recursive:
            # Use ** for recursive globbing
            if not pattern.startswith("**/"):
                pattern = f"**/{pattern}"
            matched = glob.glob(pattern, recursive=True)
        else:
            matched = glob.glob(pattern)
        
        files.extend(Path(f) for f in matched if Path(f).is_file())
    
    # Remove duplicates while preserving order
    seen = set()
    unique_files = []
    for file in files:
        if file not in seen:
            seen.add(file)
            unique_files.append(file)
    
    return unique_files


def display_progress(current: int, total: int, filename: str) -> None:
    """Display progress information."""
    percentage = (current / total) * 100
    bar_length = 40
    filled_length = int(bar_length * current // total)
    bar = "█" * filled_length + "░" * (bar_length - filled_length)
    
    print(f"\r[{bar}] {percentage:5.1f}% ({current:3d}/{total:3d}) {filename[:30]:<30}", end="")
    sys.stdout.flush()


def main():
    """Main batch processing function."""
    parser = argparse.ArgumentParser(
        description="Strategic Text Analysis - Batch Processing Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_process.py "*.txt" --output-dir results/
  python batch_process.py "documents/*.txt" "articles/*.md" --recursive
  python batch_process.py "**/*.txt" --format json --comparison
        """
    )
    
    parser.add_argument(
        "patterns",
        nargs="+",
        help="File patterns to process (supports wildcards)"
    )
    
    parser.add_argument(
        "--output-dir", "-d",
        type=str,
        default="batch_results",
        help="Output directory for results (default: batch_results)"
    )
    
    parser.add_argument(
        "--format", "-f",
        choices=["txt", "json", "csv", "html"],
        default="txt",
        help="Output format (default: txt)"
    )
    
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Search recursively in subdirectories"
    )
    
    parser.add_argument(
        "--comparison", "-c",
        action="store_true",
        help="Generate comparison report"
    )
    
    parser.add_argument(
        "--top-count", "-n",
        type=int,
        default=10,
        help="Number of top words per file (default: 10)"
    )
    
    parser.add_argument(
        "--parallel", "-p",
        action="store_true",
        help="Enable parallel processing (experimental)"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )
    
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing even if some files fail"
    )
    
    args = parser.parse_args()
    
    try:
        # Collect files
        if not args.quiet:
            print("Collecting files...")
        
        files = collect_files(args.patterns, args.recursive)
        
        if not files:
            print("✗ No files found matching the patterns")
            return 1
        
        if not args.quiet:
            print(f"Found {len(files)} files to process")
            print(f"Output directory: {args.output_dir}")
            print(f"Format: {args.format}")
            print()
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure analyzer
        config = {
            'top_words_count': args.top_count,
            'detailed_analysis': False,  # Faster for batch processing
            'memory_limit_mb': 512  # Conservative for batch processing
        }
        
        with TextAnalyzer(config) as analyzer:
            successful_files = []
            failed_files = []
            total_words = 0
            total_unique_words = 0
            total_time = 0
            
            # Process files
            for i, file_path in enumerate(files, 1):
                if not args.quiet:
                    display_progress(i, len(files), file_path.name)
                
                try:
                    # Generate output filename
                    output_filename = f"{file_path.stem}_analysis.{args.format}"
                    output_path = output_dir / output_filename
                    
                    # Analyze file
                    result = analyzer.analyze_file(file_path, output_path, args.format)
                    
                    if result.success:
                        successful_files.append((file_path, result))
                        stats = result.analysis_metrics.get('word_statistics', {})
                        total_words += stats.get('total_words', 0)
                        total_unique_words += stats.get('unique_words', 0)
                        total_time += result.execution_time
                    else:
                        failed_files.append((file_path, result.error_message))
                        if not args.continue_on_error:
                            print(f"\n✗ Error processing {file_path}: {result.error_message}")
                            return 1
                
                except Exception as e:
                    failed_files.append((file_path, str(e)))
                    if not args.continue_on_error:
                        print(f"\n✗ Error processing {file_path}: {e}")
                        return 1
            
            if not args.quiet:
                print()  # New line after progress bar
            
            # Display results summary
            print("\n" + "="*60)
            print("BATCH PROCESSING RESULTS")
            print("="*60)
            print(f"Total Files: {len(files)}")
            print(f"Successful: {len(successful_files)}")
            print(f"Failed: {len(failed_files)}")
            print(f"Success Rate: {len(successful_files)/len(files)*100:.1f}%")
            print()
            
            if successful_files:
                print("AGGREGATE STATISTICS:")
                print("-" * 25)
                print(f"Total Words (All Files): {total_words:,}")
                print(f"Total Unique Words: {total_unique_words:,}")
                print(f"Average Words per File: {total_words/len(successful_files):,.0f}")
                print(f"Total Processing Time: {total_time:.2f}s")
                print(f"Average Time per File: {total_time/len(successful_files):.2f}s")
                print()
                
                # Show top files by word count
                successful_files.sort(key=lambda x: x[1].analysis_metrics.get('word_statistics', {}).get('total_words', 0), reverse=True)
                print("TOP 5 LARGEST FILES:")
                print("-" * 20)
                for i, (file_path, result) in enumerate(successful_files[:5], 1):
                    stats = result.analysis_metrics.get('word_statistics', {})
                    words = stats.get('total_words', 0)
                    unique = stats.get('unique_words', 0)
                    print(f"{i}. {file_path.name}: {words:,} words, {unique:,} unique")
                print()
            
            if failed_files:
                print("FAILED FILES:")
                print("-" * 15)
                for file_path, error in failed_files[:10]:  # Show first 10
                    print(f"✗ {file_path.name}: {error}")
                
                if len(failed_files) > 10:
                    print(f"... and {len(failed_files) - 10} more")
                print()
            
            # Generate comparison report if requested
            if args.comparison and len(successful_files) >= 2:
                print("Generating comparison report...")
                results_only = [result for _, result in successful_files]
                comparison_path = output_dir / f"comparison_report.{args.format}"
                
                comparison = analyzer.generate_comparison_report(
                    results_only,
                    comparison_path
                )
                
                if 'error' not in comparison:
                    print(f"✓ Comparison report saved: {comparison_path}")
                    
                    # Display comparison highlights
                    summary = comparison.get('summary_statistics', {})
                    print("\nCOMPARISON HIGHLIGHTS:")
                    print("-" * 22)
                    print(f"Global Vocabulary: {summary.get('global_vocabulary_size', 0):,} words")
                    print(f"Average Vocabulary Richness: {summary.get('average_vocabulary_richness', 0):.4f}")
                    print(f"Average Complexity Score: {summary.get('average_complexity_score', 0):.3f}")
                else:
                    print(f"✗ Failed to generate comparison report: {comparison['error']}")
            
            # Final status
            if successful_files:
                print(f"\n✓ Batch processing completed!")
                print(f"  Results saved in: {output_dir}")
                return 0
            else:
                print(f"\n✗ No files were processed successfully")
                return 1
    
    except KeyboardInterrupt:
        print(f"\n\n⚠ Batch processing interrupted by user")
        return 130
    except Exception as e:
        print(f"\n✗ Batch processing failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())