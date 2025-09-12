"""
Strategic CLI main interface for Text Analyzer.
Command-line entry point with comprehensive error handling and user interaction.
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import traceback

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from config import (
    get_cli_logger, 
    initialize_config, 
    PerformanceLogger, 
    LoggedOperation
)
from ..core import TextAnalyzer, AnalysisResult
from ..utils import (
    FormattingHelper, 
    SystemHelper, 
    handle_exception,
    InvalidArgumentError,
    CommandExecutionError
)
from .argument_parser import ArgumentParser, ParsedArguments, parse_command_line


class CLIInterface:
    """
    Strategic CLI interface implementing INTJ systematic approach.
    Comprehensive user interaction with error handling and performance monitoring.
    """
    
    def __init__(self):
        # Initialize configuration
        if not initialize_config():
            print("ERROR: Failed to initialize configuration system", file=sys.stderr)
            sys.exit(1)
        
        self.logger = get_cli_logger()
        self.performance_logger = PerformanceLogger()
        self.analyzer = None
        
        # CLI state
        self.verbose = False
        self.quiet = False
        self.debug = False
    
    def run(self, args: Optional[List[str]] = None) -> int:
        """
        Main CLI execution method.
        
        Args:
            args: Optional command-line arguments
            
        Returns:
            Exit code (0 for success, non-zero for error)
        """
        try:
            # Parse arguments
            parsed_args = parse_command_line(args)
            
            # Configure logging and verbosity
            self._configure_verbosity(parsed_args)
            
            # Handle special commands
            if self._handle_special_commands(parsed_args):
                return 0
            
            # Initialize analyzer
            self.analyzer = self._create_analyzer(parsed_args)
            
            # Execute main analysis workflow
            return self._execute_analysis_workflow(parsed_args)
            
        except InvalidArgumentError as e:
            self._print_error(f"Invalid arguments: {e}")
            return 1
        except KeyboardInterrupt:
            self._print_info("Analysis interrupted by user")
            return 130  # Standard exit code for SIGINT
        except Exception as e:
            return self._handle_unexpected_error(e)
        finally:
            self._cleanup()
    
    def _configure_verbosity(self, args: ParsedArguments) -> None:
        """Configure verbosity levels based on arguments."""
        self.verbose = args.verbose
        self.quiet = args.quiet
        
        if hasattr(args, 'debug'):
            self.debug = args.debug
    
    def _handle_special_commands(self, args: ParsedArguments) -> bool:
        """
        Handle special commands that don't require full analysis.
        
        Returns:
            True if special command was handled
        """
        # System information
        if hasattr(args, 'system_info') and args.system_info:
            self._display_system_info()
            return True
        
        # Preview mode
        if args.preview_only:
            return self._handle_preview_mode(args)
        
        # Optimization analysis
        if args.optimize_config:
            return self._handle_optimization_mode(args)
        
        return False
    
    def _display_system_info(self) -> None:
        """Display comprehensive system information."""
        if not self.quiet:
            print("TEXT ANALYZER - SYSTEM INFORMATION")
            print("=" * 50)
        
        try:
            # Create temporary analyzer to get system status
            temp_analyzer = TextAnalyzer()
            status = temp_analyzer.get_system_status()
            
            if status.get('health_status') == 'error':
                self._print_error(f"System status error: {status.get('error', 'Unknown')}")
                return
            
            # Display system information
            sys_info = status['system_info']
            memory_status = status['memory_status']
            
            print(f"System Platform: {sys_info.get('platform', 'Unknown')}")
            print(f"Python Version: {sys_info.get('python_version', 'Unknown')}")
            print(f"CPU Cores: {sys_info.get('cpu_count', 'Unknown')}")
            print(f"Total Memory: {FormattingHelper.format_bytes(sys_info.get('memory_total', 0))}")
            print(f"Available Memory: {FormattingHelper.format_bytes(sys_info.get('memory_available', 0))}")
            print(f"Memory Usage: {memory_status.get('system_memory_percent', 0):.1f}%")
            print()
            
            # Display configuration
            print("ANALYZER CONFIGURATION:")
            print("-" * 25)
            config = status['configuration']
            print(f"Top Words Count: {config['top_words_count']}")
            print(f"Memory Limit: {config['memory_limit_mb']} MB")
            print(f"Detailed Analysis: {config['detailed_analysis_enabled']}")
            print()
            
            # Display supported formats
            print("SUPPORTED FORMATS:")
            print("-" * 20)
            formats = status['supported_formats']
            print(f"Input Extensions: {', '.join(formats['input_extensions'])}")
            print(f"Output Formats: {', '.join(formats['output_formats'])}")
            print(f"Encodings: {', '.join(formats['encodings'][:5])}...")
            print()
            
            # Display health status
            health = status['health_status']
            health_color = '✓' if health == 'healthy' else '⚠' if health == 'warning' else '✗'
            print(f"System Health: {health_color} {health.upper()}")
            
        except Exception as e:
            self._print_error(f"Failed to retrieve system information: {e}")
    
    def _handle_preview_mode(self, args: ParsedArguments) -> bool:
        """Handle preview mode execution."""
        if not args.input_file:
            self._print_error("No input file specified for preview")
            return True
        
        try:
            analyzer = TextAnalyzer()
            preview_lines = getattr(args, 'preview_lines', 10)
            preview = analyzer.get_analysis_preview(args.input_file, preview_lines)
            
            if not preview.get('file_preview', {}).get('success', False):
                self._print_error(f"Preview failed: {preview.get('error', 'Unknown error')}")
                return True
            
            # Display preview information
            self._display_file_preview(preview, args.input_file)
            return True
            
        except Exception as e:
            self._print_error(f"Preview mode failed: {e}")
            return True
    
    def _handle_optimization_mode(self, args: ParsedArguments) -> bool:
        """Handle configuration optimization mode."""
        if not args.input_file:
            self._print_error("No input file specified for optimization")
            return True
        
        try:
            analyzer = TextAnalyzer()
            recommendations = analyzer.optimize_for_file(args.input_file)
            
            if 'error' in recommendations:
                self._print_error(f"Optimization failed: {recommendations['error']}")
                return True
            
            # Display optimization recommendations
            self._display_optimization_recommendations(recommendations, args.input_file)
            return True
            
        except Exception as e:
            self._print_error(f"Optimization mode failed: {e}")
            return True
    
    def _create_analyzer(self, args: ParsedArguments) -> TextAnalyzer:
        """Create and configure text analyzer based on arguments."""
        config = {}
        
        # Analysis configuration
        if args.case_sensitive:
            config['case_sensitive'] = True
        
        if args.remove_stop_words:
            config['remove_stop_words'] = True
            if hasattr(args, 'stop_words_language'):
                config['stop_words_language'] = args.stop_words_language
        
        if args.min_word_length is not None:
            config['min_word_length'] = args.min_word_length
        
        if args.max_word_length is not None:
            config['max_word_length'] = args.max_word_length
        
        if hasattr(args, 'word_pattern') and args.word_pattern:
            config['word_pattern'] = args.word_pattern
        
        # Performance configuration
        if hasattr(args, 'memory_limit') and args.memory_limit:
            config['memory_limit_mb'] = args.memory_limit
        
        if hasattr(args, 'detailed_analysis') and args.detailed_analysis:
            config['detailed_analysis'] = True
        
        # Output configuration
        config['top_words_count'] = args.top_count
        config['output_formats'] = [args.format]
        
        return TextAnalyzer(config)
    
    def _execute_analysis_workflow(self, args: ParsedArguments) -> int:
        """Execute the main analysis workflow."""
        with LoggedOperation("CLI Analysis Workflow", self.logger):
            try:
                if args.batch_mode:
                    return self._execute_batch_analysis(args)
                else:
                    return self._execute_single_analysis(args)
            
            except Exception as e:
                error_info = handle_exception(e, self.logger, "analysis_workflow")
                self._print_error(f"Analysis failed: {error_info['error_message']}")
                return 1
    
    def _execute_single_analysis(self, args: ParsedArguments) -> int:
        """Execute analysis for a single file."""
        if not args.input_file:
            self._print_error("No input file specified")
            return 1
        
        self._print_info(f"Analyzing: {args.input_file}")
        
        try:
            # Execute analysis
            result = self.analyzer.analyze_file(
                args.input_file, 
                args.output_file, 
                args.format
            )
            
            if not result.success:
                self._print_error(f"Analysis failed: {result.error_message}")
                return 1
            
            # Display results
            self._display_analysis_results(result, args)
            
            # Performance summary
            if self.verbose:
                self._display_performance_summary(result)
            
            self._print_success(f"Analysis completed successfully in {result.execution_time:.3f}s")
            return 0
            
        except Exception as e:
            self._print_error(f"Single file analysis failed: {e}")
            return 1
    
    def _execute_batch_analysis(self, args: ParsedArguments) -> int:
        """Execute batch analysis for multiple files."""
        # Collect all input files for batch processing
        input_files = self._collect_batch_files(args)
        
        if not input_files:
            self._print_error("No input files found for batch processing")
            return 1
        
        self._print_info(f"Starting batch analysis: {len(input_files)} files")
        
        try:
            # Execute batch analysis
            results = self.analyzer.analyze_multiple_files(
                input_files,
                args.output_directory,
                args.format
            )
            
            # Display batch results
            self._display_batch_results(results, args)
            
            # Generate comparison report if requested
            if args.generate_comparison:
                self._generate_comparison_report(results, args)
            
            return 0
            
        except Exception as e:
            self._print_error(f"Batch analysis failed: {e}")
            return 1
    
    def _collect_batch_files(self, args: ParsedArguments) -> List[Path]:
        """Collect all files for batch processing."""
        # Implementation would collect files from various sources
        # For now, return single file as list
        if args.input_file:
            return [args.input_file]
        return []
    
    def _display_file_preview(self, preview: Dict[str, Any], filepath: Path) -> None:
        """Display file preview information."""
        file_preview = preview['file_preview']
        quick_analysis = preview['quick_analysis']
        
        if not self.quiet:
            print(f"\nFILE PREVIEW: {filepath}")
            print("=" * 50)
            print(f"File Size: {FormattingHelper.format_bytes(file_preview['file_size'])}")
            print(f"Encoding: {file_preview['encoding']}")
            print(f"Total Lines: {FormattingHelper.format_number(file_preview['total_lines'])}")
            print()
            
            print("CONTENT PREVIEW:")
            print("-" * 20)
            for i, line in enumerate(file_preview['preview_lines'], 1):
                print(f"{i:3}: {line}")
            
            if file_preview['is_truncated']:
                print(f"... ({file_preview['total_lines'] - len(file_preview['preview_lines'])} more lines)")
            
            print()
            print("QUICK ANALYSIS:")
            print("-" * 15)
            print(f"Preview Words: {FormattingHelper.format_number(quick_analysis['preview_word_count'])}")
            print(f"Estimated Total: {FormattingHelper.format_number(quick_analysis['estimated_total_words'])}")
            
            if quick_analysis['top_words_preview']:
                print("\nTop Words (Preview):")
                for word, count in quick_analysis['top_words_preview']:
                    print(f"  {word}: {count}")
    
    def _display_optimization_recommendations(self, recommendations: Dict[str, Any], filepath: Path) -> None:
        """Display optimization recommendations."""
        if not self.quiet:
            print(f"\nOPTIMIZATION ANALYSIS: {filepath}")
            print("=" * 50)
            
            # Performance recommendations
            perf = recommendations.get('performance_optimization', {})
            print("PERFORMANCE RECOMMENDATIONS:")
            print("-" * 30)
            print(f"Estimated Memory Usage: {perf.get('estimated_memory_usage_mb', 0):.1f} MB")
            print(f"Estimated Processing Time: {perf.get('estimated_processing_time_seconds', 0):.1f}s")
            print(f"Recommended Batch Size: {perf.get('recommended_batch_size', 1)}")
            
            if perf.get('should_use_streaming', False):
                print("⚠ Recommendation: Use streaming mode for large file")
            
            print()
            
            # Analysis configuration recommendations
            analysis = recommendations.get('analysis_configuration', {})
            print("ANALYSIS CONFIGURATION:")
            print("-" * 25)
            print(f"Optimal Top Words Count: {analysis.get('optimal_top_words_count', 10)}")
            
            if analysis.get('memory_efficient_mode', False):
                print("⚠ Recommendation: Enable memory-efficient mode")
            
            print()
            
            # File characteristics
            characteristics = recommendations.get('file_characteristics', {})
            print("FILE CHARACTERISTICS:")
            print("-" * 20)
            print(f"Size Category: {characteristics.get('size_category', 'unknown').title()}")
            print(f"Complexity: {characteristics.get('complexity_category', 'unknown').title()}")
            print(f"Language: {characteristics.get('estimated_language', 'unknown').title()}")
    
    def _display_analysis_results(self, result: AnalysisResult, args: ParsedArguments) -> None:
        """Display analysis results to console."""
        if self.quiet:
            return
        
        print(f"\nANALYSIS RESULTS: {result.file_info.get('path', 'Unknown')}")
        print("=" * 60)
        
        # Basic statistics
        metrics = result.analysis_metrics.get('word_statistics', {})
        print(f"Total Words: {FormattingHelper.format_number(metrics.get('total_words', 0))}")
        print(f"Unique Words: {FormattingHelper.format_number(metrics.get('unique_words', 0))}")
        print(f"Vocabulary Richness: {metrics.get('vocabulary_richness', 0):.4f}")
        
        # Top words
        print(f"\nTOP {args.top_count} MOST FREQUENT WORDS:")
        print("-" * 40)
        
        total_words = metrics.get('total_words', 1)
        for i, (word, count) in enumerate(result.word_frequencies.most_common(args.top_count), 1):
            percentage = FormattingHelper.format_percentage(count, total_words)
            print(f"{i:2}. {word:<15} | {count:>6} ({percentage})")
    
    def _display_performance_summary(self, result: AnalysisResult) -> None:
        """Display performance summary information."""
        if not self.verbose:
            return
        
        print(f"\nPERFORMANCE SUMMARY:")
        print("-" * 20)
        print(f"Execution Time: {result.execution_time:.3f}s")
        
        processing_stats = result.processing_stats
        if processing_stats:
            words_per_second = processing_stats.get('total_words', 0) / result.execution_time if result.execution_time > 0 else 0
            print(f"Processing Rate: {words_per_second:,.0f} words/second")
            print(f"Operations Applied: {', '.join(processing_stats.get('operations_applied', []))}")
    
    def _display_batch_results(self, results: Dict[str, AnalysisResult], args: ParsedArguments) -> None:
        """Display batch analysis results."""
        if self.quiet:
            return
        
        successful_results = {k: v for k, v in results.items() if v.success}
        failed_results = {k: v for k, v in results.items() if not v.success}
        
        print(f"\nBATCH ANALYSIS RESULTS:")
        print("=" * 30)
        print(f"Total Files: {len(results)}")
        print(f"Successful: {len(successful_results)}")
        print(f"Failed: {len(failed_results)}")
        
        if successful_results and self.verbose:
            print("\nSUCCESSFUL ANALYSES:")
            print("-" * 20)
            for filepath, result in successful_results.items():
                metrics = result.analysis_metrics.get('word_statistics', {})
                words = metrics.get('total_words', 0)
                unique = metrics.get('unique_words', 0)
                print(f"{Path(filepath).name}: {words:,} words, {unique:,} unique")
        
        if failed_results:
            print("\nFAILED ANALYSES:")
            print("-" * 15)
            for filepath, result in failed_results.items():
                print(f"{Path(filepath).name}: {result.error_message}")
    
    def _generate_comparison_report(self, results: Dict[str, AnalysisResult], args: ParsedArguments) -> None:
        """Generate and display comparison report."""
        try:
            successful_results = [r for r in results.values() if r.success]
            
            if len(successful_results) < 2:
                self._print_warning("Need at least 2 successful analyses for comparison report")
                return
            
            comparison = self.analyzer.generate_comparison_report(
                successful_results,
                args.output_directory / "comparison_report.txt" if args.output_directory else None
            )
            
            if 'error' not in comparison:
                self._print_success("Comparison report generated successfully")
                
                if not self.quiet:
                    summary = comparison['summary_statistics']
                    print(f"\nCOMPARISON SUMMARY:")
                    print("-" * 20)
                    print(f"Files Compared: {summary['total_files_analyzed']}")
                    print(f"Total Words (All Files): {FormattingHelper.format_number(summary['total_words_all_files'])}")
                    print(f"Global Vocabulary: {FormattingHelper.format_number(summary['global_vocabulary_size'])}")
                    print(f"Average Complexity: {summary['average_complexity_score']:.3f}")
            
        except Exception as e:
            self._print_error(f"Comparison report generation failed: {e}")
    
    def _print_info(self, message: str) -> None:
        """Print informational message."""
        if not self.quiet:
            print(f"INFO: {message}")
    
    def _print_success(self, message: str) -> None:
        """Print success message."""
        if not self.quiet:
            print(f"✓ {message}")
    
    def _print_warning(self, message: str) -> None:
        """Print warning message."""
        if not self.quiet:
            print(f"⚠ WARNING: {message}")
    
    def _print_error(self, message: str) -> None:
        """Print error message."""
        print(f"✗ ERROR: {message}", file=sys.stderr)
    
    def _handle_unexpected_error(self, error: Exception) -> int:
        """Handle unexpected errors with comprehensive reporting."""
        self._print_error(f"Unexpected error: {error}")
        
        if self.debug or self.verbose:
            print(f"\nDEBUG INFORMATION:", file=sys.stderr)
            print("-" * 20, file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        
        self.logger.error(f"Unexpected error in CLI: {error}", exc_info=True)
        return 2
    
    def _cleanup(self) -> None:
        """Cleanup resources before exit."""
        try:
            if self.analyzer:
                self.analyzer.cleanup_resources()
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")


def main() -> int:
    """
    Main entry point for CLI application.
    
    Returns:
        Exit code
    """
    cli = CLIInterface()
    return cli.run()


if __name__ == "__main__":
    sys.exit(main())