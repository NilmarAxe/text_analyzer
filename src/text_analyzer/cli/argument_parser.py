"""
Strategic command-line argument parsing system.
Comprehensive CLI interface with systematic validation and help system.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from config import AnalysisConfig, OutputConfig
from ..utils import CLIValidator, InvalidArgumentError


@dataclass
class ParsedArguments:
    """Structured container for parsed CLI arguments."""
    input_file: Path
    output_file: Optional[Path] = None
    format: str = 'txt'
    top_count: int = 10
    verbose: bool = False
    quiet: bool = False
    case_sensitive: bool = False
    remove_stop_words: bool = False
    min_word_length: Optional[int] = None
    max_word_length: Optional[int] = None
    encoding: Optional[str] = None
    preview_only: bool = False
    batch_mode: bool = False
    output_directory: Optional[Path] = None
    config_file: Optional[Path] = None
    generate_comparison: bool = False
    optimize_config: bool = False


class ArgumentParser:
    """
    Strategic CLI argument parser with comprehensive validation.
    """
    
    def __init__(self):
        self.parser = self._create_parser()
        self.validator = CLIValidator()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create comprehensive argument parser with all options."""
        parser = argparse.ArgumentParser(
            prog='text-analyzer',
            description='Strategic Text Analysis System - Comprehensive text frequency and statistical analysis',
            epilog='Examples:\n'
                   '  text-analyzer document.txt\n'
                   '  text-analyzer document.txt --output report.html --format html\n'
                   '  text-analyzer *.txt --batch --output-dir results/\n'
                   '  text-analyzer document.txt --top-count 20 --remove-stop-words',
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        # Core arguments
        self._add_core_arguments(parser)
        
        # Output arguments
        self._add_output_arguments(parser)
        
        # Analysis configuration arguments
        self._add_analysis_arguments(parser)
        
        # Advanced arguments
        self._add_advanced_arguments(parser)
        
        # Utility arguments
        self._add_utility_arguments(parser)
        
        return parser
    
    def _add_core_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add core required and primary arguments."""
        parser.add_argument(
            'input',
            nargs='*',
            help='Input text file(s) to analyze. Supports wildcards for batch processing.'
        )
        
        parser.add_argument(
            '--input-file', '-i',
            type=str,
            help='Explicitly specify input file (alternative to positional argument)'
        )
        
        parser.add_argument(
            '--version', '-V',
            action='version',
            version='Text Analyzer 1.0.0 - Strategic Analysis System'
        )
    
    def _add_output_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add output-related arguments."""
        output_group = parser.add_argument_group('Output Options')
        
        output_group.add_argument(
            '--output', '-o',
            type=str,
            help='Output file path for the analysis report'
        )
        
        output_group.add_argument(
            '--format', '-f',
            choices=['txt', 'json', 'csv', 'html'],
            default='txt',
            help='Output format for the report (default: txt)'
        )
        
        output_group.add_argument(
            '--output-dir', '-d',
            type=str,
            help='Output directory for batch processing or multiple format outputs'
        )
        
        output_group.add_argument(
            '--all-formats',
            action='store_true',
            help='Generate reports in all supported formats'
        )
        
        output_group.add_argument(
            '--no-console',
            action='store_true',
            help='Suppress console output (only save to file)'
        )
    
    def _add_analysis_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add analysis configuration arguments."""
        analysis_group = parser.add_argument_group('Analysis Configuration')
        
        analysis_group.add_argument(
            '--top-count', '-n',
            type=int,
            default=AnalysisConfig.DEFAULT_TOP_WORDS_COUNT,
            help=f'Number of top frequent words to display (default: {AnalysisConfig.DEFAULT_TOP_WORDS_COUNT})'
        )
        
        analysis_group.add_argument(
            '--case-sensitive',
            action='store_true',
            help='Enable case-sensitive analysis (default: case-insensitive)'
        )
        
        analysis_group.add_argument(
            '--remove-stop-words',
            action='store_true',
            help='Remove common stop words from analysis'
        )
        
        analysis_group.add_argument(
            '--stop-words-language',
            choices=['english'],
            default='english',
            help='Language for stop words removal (default: english)'
        )
        
        analysis_group.add_argument(
            '--min-word-length',
            type=int,
            help=f'Minimum word length to include (default: {AnalysisConfig.MIN_WORD_LENGTH})'
        )
        
        analysis_group.add_argument(
            '--max-word-length',
            type=int,
            help=f'Maximum word length to include (default: {AnalysisConfig.MAX_WORD_LENGTH})'
        )
        
        analysis_group.add_argument(
            '--word-pattern',
            type=str,
            help='Custom regex pattern for word extraction (advanced users)'
        )
    
    def _add_advanced_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add advanced configuration arguments."""
        advanced_group = parser.add_argument_group('Advanced Options')
        
        advanced_group.add_argument(
            '--encoding',
            type=str,
            choices=AnalysisConfig.SUPPORTED_ENCODINGS,
            help='Force specific file encoding (auto-detect if not specified)'
        )
        
        advanced_group.add_argument(
            '--memory-limit',
            type=int,
            help='Memory limit in MB for processing large files'
        )
        
        advanced_group.add_argument(
            '--config',
            type=str,
            help='Configuration file path (JSON format)'
        )
        
        advanced_group.add_argument(
            '--optimize-config',
            action='store_true',
            help='Analyze file and suggest optimal configuration settings'
        )
        
        advanced_group.add_argument(
            '--detailed-analysis',
            action='store_true',
            help='Enable detailed statistical analysis (slower but more comprehensive)'
        )
        
        advanced_group.add_argument(
            '--performance-profile',
            action='store_true',
            help='Enable performance profiling and detailed timing information'
        )
    
    def _add_utility_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add utility and convenience arguments."""
        utility_group = parser.add_argument_group('Utility Options')
        
        utility_group.add_argument(
            '--preview',
            action='store_true',
            help='Show file preview and quick analysis without full processing'
        )
        
        utility_group.add_argument(
            '--preview-lines',
            type=int,
            default=10,
            help='Number of lines to show in preview mode (default: 10)'
        )
        
        utility_group.add_argument(
            '--batch',
            action='store_true',
            help='Enable batch processing mode for multiple files'
        )
        
        utility_group.add_argument(
            '--comparison-report',
            action='store_true',
            help='Generate comparison report when processing multiple files'
        )
        
        utility_group.add_argument(
            '--system-info',
            action='store_true',
            help='Display system information and analyzer status'
        )
        
        # Verbosity control
        verbosity_group = utility_group.add_mutually_exclusive_group()
        verbosity_group.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose output with detailed processing information'
        )
        
        verbosity_group.add_argument(
            '--quiet', '-q',
            action='store_true',
            help='Suppress all non-essential output'
        )
        
        verbosity_group.add_argument(
            '--debug',
            action='store_true',
            help='Enable debug mode with extensive logging'
        )
    
    def parse_arguments(self, args: Optional[List[str]] = None) -> ParsedArguments:
        """
        Parse and validate command-line arguments.
        
        Args:
            args: Optional argument list (uses sys.argv if None)
            
        Returns:
            ParsedArguments object with validated arguments
            
        Raises:
            InvalidArgumentError: If arguments are invalid
            SystemExit: If help is requested or critical errors occur
        """
        try:
            # Parse arguments
            parsed_args = self.parser.parse_args(args)
            
            # Handle special cases
            if parsed_args.system_info:
                return self._create_system_info_args()
            
            # Validate input files
            input_files = self._resolve_input_files(parsed_args)
            if not input_files and not parsed_args.system_info:
                self.parser.error("No input files specified")
            
            # Create structured arguments object
            structured_args = self._create_structured_args(parsed_args, input_files)
            
            # Validate arguments
            self._validate_arguments(structured_args)
            
            return structured_args
            
        except argparse.ArgumentError as e:
            raise InvalidArgumentError('argument_parsing', str(e), 'Argument parsing failed')
        except Exception as e:
            raise InvalidArgumentError('validation', str(e), 'Argument validation failed')
    
    def _resolve_input_files(self, args: argparse.Namespace) -> List[Path]:
        """Resolve input files from various argument sources."""
        input_files = []
        
        # From positional arguments
        if args.input:
            for file_pattern in args.input:
                if '*' in file_pattern or '?' in file_pattern:
                    # Handle wildcards
                    import glob
                    matched_files = glob.glob(file_pattern)
                    input_files.extend(Path(f) for f in matched_files)
                else:
                    input_files.append(Path(file_pattern))
        
        # From --input-file flag
        if args.input_file:
            input_files.append(Path(args.input_file))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_files = []
        for file_path in input_files:
            if file_path not in seen:
                seen.add(file_path)
                unique_files.append(file_path)
        
        return unique_files
    
    def _create_structured_args(self, args: argparse.Namespace, 
                               input_files: List[Path]) -> ParsedArguments:
        """Create structured arguments object from parsed namespace."""
        # Handle batch mode detection
        batch_mode = args.batch or len(input_files) > 1
        
        # Determine primary input file
        primary_input = input_files[0] if input_files else None
        
        # Handle output configuration
        output_file = None
        output_directory = None
        
        if args.output:
            output_path = Path(args.output)
            if batch_mode or args.output_dir:
                output_directory = output_path
            else:
                output_file = output_path
        
        if args.output_dir:
            output_directory = Path(args.output_dir)
        
        return ParsedArguments(
            input_file=primary_input,
            output_file=output_file,
            format=args.format,
            top_count=args.top_count,
            verbose=getattr(args, 'verbose', False),
            quiet=getattr(args, 'quiet', False),
            case_sensitive=args.case_sensitive,
            remove_stop_words=args.remove_stop_words,
            min_word_length=args.min_word_length,
            max_word_length=args.max_word_length,
            encoding=args.encoding,
            preview_only=getattr(args, 'preview', False),
            batch_mode=batch_mode,
            output_directory=output_directory,
            config_file=Path(args.config) if getattr(args, 'config', None) else None,
            generate_comparison=getattr(args, 'comparison_report', False),
            optimize_config=getattr(args, 'optimize_config', False)
        )
    
    def _create_system_info_args(self) -> ParsedArguments:
        """Create arguments object for system info display."""
        return ParsedArguments(
            input_file=Path('/dev/null'),  # Dummy path
            preview_only=True  # Use preview mode to avoid file processing
        )
    
    def _validate_arguments(self, args: ParsedArguments) -> None:
        """Validate structured arguments for consistency and correctness."""
        validation_data = {
            'input_file': args.input_file,
            'output_file': args.output_file,
            'top_count': args.top_count,
            'format': args.format,
            'verbose': args.verbose
        }
        
        try:
            self.validator.validate_cli_arguments(validation_data)
        except Exception as e:
            raise InvalidArgumentError('validation', str(validation_data), str(e))
        
        # Additional custom validations
        if args.min_word_length is not None and args.max_word_length is not None:
            if args.min_word_length >= args.max_word_length:
                raise InvalidArgumentError(
                    'word_length', 
                    f'min={args.min_word_length}, max={args.max_word_length}',
                    'Minimum word length must be less than maximum'
                )
        
        if args.top_count <= 0:
            raise InvalidArgumentError(
                'top_count', args.top_count, 'Must be a positive integer'
            )
        
        # Validate file access if not in preview mode
        if not args.preview_only and args.input_file:
            if not args.input_file.exists():
                raise InvalidArgumentError(
                    'input_file', str(args.input_file), 'File does not exist'
                )
    
    def print_help(self) -> None:
        """Print comprehensive help information."""
        self.parser.print_help()
    
    def print_usage(self) -> None:
        """Print usage information."""
        self.parser.print_usage()
    
    def get_configuration_template(self) -> Dict[str, Any]:
        """
        Generate configuration file template.
        
        Returns:
            Dictionary with default configuration structure
        """
        return {
            "analysis": {
                "top_words_count": AnalysisConfig.DEFAULT_TOP_WORDS_COUNT,
                "case_sensitive": AnalysisConfig.CASE_SENSITIVE,
                "min_word_length": AnalysisConfig.MIN_WORD_LENGTH,
                "max_word_length": AnalysisConfig.MAX_WORD_LENGTH,
                "remove_stop_words": False,
                "stop_words_language": "english",
                "word_pattern": AnalysisConfig.WORD_PATTERN
            },
            "output": {
                "default_format": "txt",
                "report_width": OutputConfig.REPORT_WIDTH,
                "generate_all_formats": False,
                "include_statistics": True
            },
            "performance": {
                "memory_limit_mb": 512,
                "enable_optimization": True,
                "detailed_analysis": True,
                "performance_profiling": False
            },
            "files": {
                "supported_extensions": AnalysisConfig.SUPPORTED_EXTENSIONS,
                "encoding_preference": AnalysisConfig.SUPPORTED_ENCODINGS,
                "max_file_size_mb": AnalysisConfig.MAX_FILE_SIZE // (1024 * 1024)
            }
        }
    
    def generate_examples(self) -> str:
        """
        Generate comprehensive usage examples.
        
        Returns:
            String with detailed usage examples
        """
        examples = """
USAGE EXAMPLES:

Basic Analysis:
  text-analyzer document.txt
  text-analyzer document.txt --output report.txt

Multiple Formats:
  text-analyzer document.txt --format json --output analysis.json
  text-analyzer document.txt --format html --output report.html
  text-analyzer document.txt --all-formats --output-dir results/

Analysis Configuration:
  text-analyzer document.txt --top-count 20
  text-analyzer document.txt --case-sensitive --remove-stop-words
  text-analyzer document.txt --min-word-length 3 --max-word-length 15

Batch Processing:
  text-analyzer *.txt --batch --output-dir batch_results/
  text-analyzer file1.txt file2.txt --comparison-report
  text-analyzer documents/*.txt --batch --format json

Preview and Optimization:
  text-analyzer document.txt --preview
  text-analyzer document.txt --optimize-config
  text-analyzer large_file.txt --memory-limit 1024

Advanced Usage:
  text-analyzer document.txt --encoding utf-8 --detailed-analysis
  text-analyzer document.txt --config analysis_config.json
  text-analyzer document.txt --performance-profile --verbose

Utility Commands:
  text-analyzer --system-info
  text-analyzer document.txt --preview --preview-lines 20
        """
        
        return examples.strip()
    
    def validate_config_file(self, config_path: Path) -> Dict[str, Any]:
        """
        Validate and load configuration file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Validated configuration dictionary
            
        Raises:
            InvalidArgumentError: If configuration is invalid
        """
        try:
            import json
            
            if not config_path.exists():
                raise InvalidArgumentError(
                    'config_file', str(config_path), 'Configuration file not found'
                )
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Validate structure
            template = self.get_configuration_template()
            self._validate_config_structure(config_data, template)
            
            return config_data
            
        except json.JSONDecodeError as e:
            raise InvalidArgumentError(
                'config_file', str(config_path), f'Invalid JSON format: {e}'
            )
        except Exception as e:
            raise InvalidArgumentError(
                'config_file', str(config_path), f'Configuration validation failed: {e}'
            )
    
    def _validate_config_structure(self, config: Dict[str, Any], 
                                  template: Dict[str, Any], 
                                  path: str = '') -> None:
        """Recursively validate configuration structure."""
        for key, value in config.items():
            current_path = f"{path}.{key}" if path else key
            
            if key not in template:
                # Allow additional keys but warn
                continue
            
            if isinstance(template[key], dict) and isinstance(value, dict):
                self._validate_config_structure(value, template[key], current_path)
            elif not isinstance(value, type(template[key])):
                expected_type = type(template[key]).__name__
                actual_type = type(value).__name__
                raise InvalidArgumentError(
                    'config_structure', 
                    f'{current_path}: {actual_type}',
                    f'Expected {expected_type}'
                )


def create_argument_parser() -> ArgumentParser:
    """Factory function to create argument parser."""
    return ArgumentParser()


def parse_command_line(args: Optional[List[str]] = None) -> ParsedArguments:
    """
    Convenience function to parse command line arguments.
    
    Args:
        args: Optional argument list
        
    Returns:
        ParsedArguments object
    """
    parser = create_argument_parser()
    return parser.parse_arguments(args)