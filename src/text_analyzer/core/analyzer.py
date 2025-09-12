"""
Core Text Analyzer - Strategic analysis orchestration system.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from collections import Counter

from config import (
    get_analyzer_logger, 
    AnalysisConfig, 
    SystemConfig, 
    LoggedOperation, 
    PerformanceLogger,
    initialize_config
)
from .file_handler import FileHandler
from .text_processor import TextProcessor, WordAnalysis
from .report_generator import ReportGenerator
from ..utils import (
    PerformanceMonitor,
    SystemHelper,
    timing_decorator,
    handle_exception,
    InsufficientDataError,
    AnalysisConfigurationError
)

@dataclass
class AnalysisResult:
    """Comprehensive analysis result container."""
    file_info: Dict[str, Any]
    word_frequencies: Counter
    analysis_metrics: Dict[str, Any]
    processing_stats: Dict[str, Any]
    word_length_distribution: Dict[int, int]
    character_distribution: Dict[str, int]
    success: bool
    error_message: Optional[str] = None
    execution_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        # Convert Counter to dict for serialization
        result['word_frequencies'] = dict(self.word_frequencies)
        result['character_distribution'] = dict(self.character_distribution)
        return result

class TextAnalyzer:
    """
    Strategic Text Analysis System - Main orchestrator.
    Implements comprehensive analysis pipeline with systematic error handling.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Initialize configuration system
        if not initialize_config():
            raise AnalysisConfigurationError("system", "Failed to initialize configuration")
        
        self.logger = get_analyzer_logger()
        self.performance_logger = PerformanceLogger()
        self.performance_monitor = PerformanceMonitor()
        
        # Initialize core components
        self.file_handler = FileHandler()
        self.text_processor = TextProcessor(config)
        self.report_generator = ReportGenerator()
        
        # Analysis configuration
        self.config = self._load_analysis_config(config or {})
        
        # System monitoring
        self.system_info = SystemHelper.get_system_info()
        
        self.logger.info("TextAnalyzer initialized successfully")
        self.logger.debug(f"System info: CPU cores={self.system_info.get('cpu_count')}, "
                         f"Memory={FormattingHelper.format_bytes(self.system_info.get('memory_total', 0))}")
    
    def _load_analysis_config(self, user_config: Dict[str, Any]) -> Dict[str, Any]:
        """Load and merge analysis configuration."""
        default_config = {
            'top_words_count': AnalysisConfig.DEFAULT_TOP_WORDS_COUNT,
            'min_word_length': AnalysisConfig.MIN_WORD_LENGTH,
            'max_word_length': AnalysisConfig.MAX_WORD_LENGTH,
            'case_sensitive': AnalysisConfig.CASE_SENSITIVE,
            'remove_stop_words': False,
            'generate_statistics': True,
            'enable_optimization': True,
            'memory_limit_mb': SystemConfig.MAX_MEMORY_USAGE // (1024 * 1024),
            'output_formats': ['txt'],
            'detailed_analysis': True
        }
        
        merged_config = {**default_config, **user_config}
        
        # Validate configuration
        self._validate_configuration(merged_config)
        
        return merged_config
    
    def _validate_configuration(self, config: Dict[str, Any]) -> None:
        """Validate analysis configuration parameters."""
        # Validate top words count
        if config['top_words_count'] <= 0:
            raise AnalysisConfigurationError(
                'top_words_count', config['top_words_count'], 'Must be positive integer'
            )
        
        # Validate word length constraints
        if config['min_word_length'] >= config['max_word_length']:
            raise AnalysisConfigurationError(
                'word_length', 
                f"min={config['min_word_length']}, max={config['max_word_length']}", 
                'Minimum must be less than maximum'
            )
        
        # Validate memory limit
        available_memory = SystemHelper.get_system_info().get('memory_available', 0) // (1024 * 1024)
        if config['memory_limit_mb'] > available_memory:
            self.logger.warning(f"Memory limit ({config['memory_limit_mb']} MB) exceeds available memory ({available_memory} MB)")
        
        self.logger.debug("Configuration validation successful")
    
    @timing_decorator("file_analysis")
    def analyze_file(self, filepath: Union[str, Path], 
                    output_path: Optional[Union[str, Path]] = None,
                    output_format: str = 'txt') -> AnalysisResult:
        """
        Comprehensive file analysis with full pipeline execution.
        
        Args:
            filepath: Path to file for analysis
            output_path: Optional output file path
            output_format: Output format for report
            
        Returns:
            AnalysisResult with comprehensive analysis data
            
        Raises:
            Various exceptions for different failure modes
        """
        self.performance_monitor.start_monitoring()
        
        with LoggedOperation(f"File Analysis: {filepath}", self.logger):
            try:
                # Check memory availability
                if not SystemHelper.is_memory_available(self.config['memory_limit_mb']):
                    raise AnalysisConfigurationError(
                        'memory', 
                        f"Insufficient memory (required: {self.config['memory_limit_mb']} MB)"
                    )
                
                # Step 1: File reading and validation
                self.logger.info(f"Reading file: {filepath}")
                text_content, file_info = self.file_handler.read_file(filepath)
                self.performance_monitor.checkpoint("file_read")
                
                # Log file processing metrics
                self.performance_logger.log_file_processing_time(
                    str(filepath), 
                    self.performance_monitor.checkpoints.get('file_read', 0),
                    file_info.size
                )
                
                # Step 2: Text processing and analysis
                self.logger.info("Processing text content")
                word_analysis = self.text_processor.analyze_text(text_content)
                self.performance_monitor.checkpoint("text_processed")
                
                # Validate sufficient data for analysis
                if word_analysis.processing_stats.total_words < 1:
                    raise InsufficientDataError(
                        'words', 1, word_analysis.processing_stats.total_words
                    )
                
                # Step 3: Generate analysis metrics
                analysis_metrics = self._generate_analysis_metrics(word_analysis)
                self.performance_monitor.checkpoint("metrics_generated")
                
                # Step 4: Create comprehensive result
                result = self._create_analysis_result(
                    file_info, word_analysis, analysis_metrics
                )
                self.performance_monitor.checkpoint("result_created")
                
                # Step 5: Generate report if requested
                if output_path:
                    self.logger.info(f"Generating {output_format} report")
                    self._generate_output_report(result, output_path, output_format)
                    self.performance_monitor.checkpoint("report_generated")
                
                # Log final performance metrics
                perf_report = self.performance_monitor.get_performance_report()
                self.performance_logger.log_analysis_metrics(
                    word_analysis.processing_stats.total_words,
                    word_analysis.processing_stats.unique_words,
                    perf_report['total_execution_time']
                )
                
                result.execution_time = perf_report['total_execution_time']
                
                self.logger.info(f"Analysis complete: {word_analysis.processing_stats.total_words} words, "
                               f"{word_analysis.processing_stats.unique_words} unique "
                               f"({result.execution_time:.3f}s)")
                
                return result
                
            except Exception as e:
                error_info = handle_exception(e, self.logger, "file_analysis")
                return AnalysisResult(
                    file_info={},
                    word_frequencies=Counter(),
                    analysis_metrics={},
                    processing_stats={},
                    word_length_distribution={},
                    character_distribution={},
                    success=False,
                    error_message=error_info['error_message'],
                    execution_time=self.performance_monitor.checkpoints.get('file_read', 0)
                )
    
    def _generate_analysis_metrics(self, word_analysis: WordAnalysis) -> Dict[str, Any]:
        """Generate comprehensive analysis metrics."""
        stats = word_analysis.processing_stats
        frequencies = word_analysis.word_frequencies
        
        if not frequencies:
            return {'error': 'No word frequencies available'}
        
        # Basic metrics
        total_words = stats.total_words
        unique_words = stats.unique_words
        vocabulary_richness = unique_words / total_words if total_words > 0 else 0
        
        # Word length statistics
        word_lengths = [len(word) for word in frequencies.keys()]
        avg_word_length = sum(word_lengths) / len(word_lengths) if word_lengths else 0
        min_word_length = min(word_lengths) if word_lengths else 0
        max_word_length = max(word_lengths) if word_lengths else 0
        
        # Frequency statistics
        frequency_values = list(frequencies.values())
        most_common_word = frequencies.most_common(1)[0] if frequencies else ('', 0)
        
        # Hapax legomena (words appearing once)
        hapax_count = sum(1 for count in frequency_values if count == 1)
        
        # Text complexity metrics
        lexical_diversity = unique_words / total_words if total_words > 0 else 0
        repetition_rate = 1 - lexical_diversity
        
        # Processing efficiency metrics
        processing_rate = total_words / stats.processing_time if stats.processing_time > 0 else 0
        
        return {
            'word_statistics': {
                'total_words': total_words,
                'unique_words': unique_words,
                'vocabulary_richness': vocabulary_richness,
                'lexical_diversity': lexical_diversity,
                'repetition_rate': repetition_rate,
                'hapax_legomena_count': hapax_count,
                'hapax_percentage': (hapax_count / unique_words * 100) if unique_words > 0 else 0
            },
            'word_length_statistics': {
                'average_length': avg_word_length,
                'minimum_length': min_word_length,
                'maximum_length': max_word_length,
                'length_range': max_word_length - min_word_length
            },
            'frequency_statistics': {
                'most_common_word': most_common_word[0],
                'most_common_count': most_common_word[1],
                'most_common_percentage': (most_common_word[1] / total_words * 100) if total_words > 0 else 0,
                'average_frequency': sum(frequency_values) / len(frequency_values) if frequency_values else 0
            },
            'processing_statistics': {
                'processing_time': stats.processing_time,
                'processing_rate_words_per_second': processing_rate,
                'operations_applied': stats.operations_applied,
                'compression_ratio': stats.processed_length / stats.original_length if stats.original_length > 0 else 0
            },
            'text_characteristics': {
                'estimated_reading_time_minutes': total_words / 200,  # Assuming 200 WPM
                'complexity_score': self._calculate_complexity_score(word_analysis),
                'text_length_category': self._categorize_text_length(total_words)
            }
        }
    
    def _calculate_complexity_score(self, word_analysis: WordAnalysis) -> float:
        """Calculate text complexity score (0.0 to 1.0)."""
        stats = word_analysis.processing_stats
        frequencies = word_analysis.word_frequencies
        
        if not frequencies:
            return 0.0
        
        # Factors contributing to complexity
        vocabulary_richness = stats.unique_words / stats.total_words if stats.total_words > 0 else 0
        avg_word_length = sum(len(word) * count for word, count in frequencies.items()) / stats.total_words
        
        # Normalize factors
        vocab_score = min(vocabulary_richness * 2, 1.0)  # Higher vocabulary = more complex
        length_score = min(avg_word_length / 10, 1.0)    # Longer words = more complex
        
        # Weighted combination
        complexity_score = (vocab_score * 0.6) + (length_score * 0.4)
        return min(max(complexity_score, 0.0), 1.0)
    
    def _categorize_text_length(self, word_count: int) -> str:
        """Categorize text by length."""
        if word_count < 100:
            return 'very_short'
        elif word_count < 500:
            return 'short'
        elif word_count < 2000:
            return 'medium'
        elif word_count < 10000:
            return 'long'
        else:
            return 'very_long'
    
    def _create_analysis_result(self, file_info: Any, 
                               word_analysis: WordAnalysis, 
                               analysis_metrics: Dict[str, Any]) -> AnalysisResult:
        """Create comprehensive analysis result object."""
        return AnalysisResult(
            file_info=file_info._asdict() if hasattr(file_info, '_asdict') else dict(file_info),
            word_frequencies=word_analysis.word_frequencies,
            analysis_metrics=analysis_metrics,
            processing_stats=word_analysis.processing_stats.__dict__ if hasattr(word_analysis.processing_stats, '__dict__') else dict(word_analysis.processing_stats),
            word_length_distribution=word_analysis.word_lengths,
            character_distribution=word_analysis.character_frequencies,
            success=True,
            execution_time=0.0  # Will be set later
        )
    
    def _generate_output_report(self, result: AnalysisResult, 
                               output_path: Union[str, Path], 
                               output_format: str) -> None:
        """Generate and save output report."""
        try:
            # Prepare data for report generation
            report_data = {
                'file_info': result.file_info,
                'word_frequencies': result.word_frequencies,
                'processing_stats': result.processing_stats,
                'word_length_distribution': result.word_length_distribution,
                'character_distribution': result.character_distribution,
                'top_count': self.config['top_words_count']
            }
            
            # Generate report
            report_result = self.report_generator.generate_report(
                report_data, output_format, output_path
            )
            
            if report_result['success']:
                self.logger.info(f"Report saved: {report_result.get('output_file', output_path)}")
            else:
                self.logger.error(f"Report generation failed: {report_result.get('error', 'Unknown error')}")
        
        except Exception as e:
            self.logger.error(f"Failed to generate output report: {e}")
    
    @timing_decorator("batch_analysis")
    def analyze_multiple_files(self, filepaths: List[Union[str, Path]],
                              output_directory: Optional[Union[str, Path]] = None,
                              output_format: str = 'txt') -> Dict[str, AnalysisResult]:
        """
        Efficiently analyze multiple files with batch processing.
        
        Args:
            filepaths: List of file paths to analyze
            output_directory: Optional output directory for reports
            output_format: Output format for reports
            
        Returns:
            Dictionary mapping filepath to AnalysisResult
        """
        results = {}
        
        with LoggedOperation(f"Batch Analysis ({len(filepaths)} files)", self.logger):
            for i, filepath in enumerate(filepaths):
                try:
                    self.logger.info(f"Analyzing file {i+1}/{len(filepaths)}: {filepath}")
                    
                    # Determine output path if directory provided
                    output_path = None
                    if output_directory:
                        filename = Path(filepath).stem + f"_analysis.{output_format}"
                        output_path = Path(output_directory) / filename
                    
                    # Analyze file
                    result = self.analyze_file(filepath, output_path, output_format)
                    results[str(filepath)] = result
                    
                    if result.success:
                        self.logger.info(f"Successfully analyzed: {filepath}")
                    else:
                        self.logger.error(f"Analysis failed for {filepath}: {result.error_message}")
                
                except Exception as e:
                    self.logger.error(f"Failed to analyze {filepath}: {e}")
                    results[str(filepath)] = AnalysisResult(
                        file_info={},
                        word_frequencies=Counter(),
                        analysis_metrics={},
                        processing_stats={},
                        word_length_distribution={},
                        character_distribution={},
                        success=False,
                        error_message=str(e)
                    )
        
        # Generate batch summary
        successful_analyses = sum(1 for result in results.values() if result.success)
        self.logger.info(f"Batch analysis complete: {successful_analyses}/{len(filepaths)} successful")
        
        return results
    
    def get_analysis_preview(self, filepath: Union[str, Path], 
                           lines: int = 10) -> Dict[str, Any]:
        """
        Get quick preview of file analysis without full processing.
        
        Args:
            filepath: Path to file
            lines: Number of preview lines
            
        Returns:
            Dictionary with preview information
        """
        try:
            # Get file preview
            file_preview = self.file_handler.get_file_preview(filepath, lines)
            
            if not file_preview['success']:
                return file_preview
            
            # Quick text analysis on preview
            preview_text = '\n'.join(file_preview['preview_lines'])
            quick_analysis = self.text_processor.analyze_text(preview_text)
            
            # Generate preview metrics
            preview_info = {
                'file_preview': file_preview,
                'quick_analysis': {
                    'preview_word_count': quick_analysis.processing_stats.total_words,
                    'preview_unique_words': quick_analysis.processing_stats.unique_words,
                    'estimated_total_words': int(
                        (quick_analysis.processing_stats.total_words / lines) * file_preview['total_lines']
                    ) if lines > 0 else 0,
                    'top_words_preview': quick_analysis.word_frequencies.most_common(5)
                },
                'recommendations': self.text_processor.optimize_processing_config(preview_text)
            }
            
            return preview_info
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'filepath': str(filepath)
            }
    
    def generate_comparison_report(self, results: List[AnalysisResult],
                                 output_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Generate comparative analysis report for multiple files.
        
        Args:
            results: List of analysis results to compare
            output_path: Optional output path for comparison report
            
        Returns:
            Comparison analysis dictionary
        """
        if not results or not any(r.success for r in results):
            return {'error': 'No successful results to compare'}
        
        successful_results = [r for r in results if r.success]
        
        with LoggedOperation("Comparison Report Generation", self.logger):
            # Aggregate statistics
            total_words_all = sum(r.analysis_metrics.get('word_statistics', {}).get('total_words', 0) 
                                for r in successful_results)
            
            total_unique_words_all = sum(r.analysis_metrics.get('word_statistics', {}).get('unique_words', 0) 
                                       for r in successful_results)
            
            # Calculate averages
            avg_vocab_richness = sum(
                r.analysis_metrics.get('word_statistics', {}).get('vocabulary_richness', 0) 
                for r in successful_results
            ) / len(successful_results)
            
            avg_complexity = sum(
                r.analysis_metrics.get('text_characteristics', {}).get('complexity_score', 0) 
                for r in successful_results
            ) / len(successful_results)
            
            # Merge word frequencies for global analysis
            combined_frequencies = Counter()
            for result in successful_results:
                combined_frequencies.update(result.word_frequencies)
            
            # Generate comparison data
            comparison_data = {
                'summary_statistics': {
                    'total_files_analyzed': len(successful_results),
                    'total_words_all_files': total_words_all,
                    'total_unique_words_all_files': total_unique_words_all,
                    'average_vocabulary_richness': avg_vocab_richness,
                    'average_complexity_score': avg_complexity,
                    'global_vocabulary_size': len(combined_frequencies)
                },
                'file_comparisons': [
                    {
                        'filepath': result.file_info.get('path', 'Unknown'),
                        'word_count': result.analysis_metrics.get('word_statistics', {}).get('total_words', 0),
                        'unique_words': result.analysis_metrics.get('word_statistics', {}).get('unique_words', 0),
                        'vocabulary_richness': result.analysis_metrics.get('word_statistics', {}).get('vocabulary_richness', 0),
                        'complexity_score': result.analysis_metrics.get('text_characteristics', {}).get('complexity_score', 0),
                        'processing_time': result.execution_time
                    }
                    for result in successful_results
                ],
                'global_top_words': combined_frequencies.most_common(20),
                'processing_summary': {
                    'total_processing_time': sum(r.execution_time for r in successful_results),
                    'average_processing_time': sum(r.execution_time for r in successful_results) / len(successful_results),
                    'fastest_file': min(successful_results, key=lambda r: r.execution_time).file_info.get('path', 'Unknown'),
                    'slowest_file': max(successful_results, key=lambda r: r.execution_time).file_info.get('path', 'Unknown')
                }
            }
            
            # Generate report if output path provided
            if output_path:
                try:
                    report_result = self.report_generator.generate_report(
                        comparison_data, 'txt', output_path
                    )
                    comparison_data['report_generated'] = report_result.get('success', False)
                    comparison_data['report_path'] = report_result.get('output_file')
                except Exception as e:
                    comparison_data['report_error'] = str(e)
            
            self.logger.info(f"Comparison report generated for {len(successful_results)} files")
            return comparison_data
    
    def optimize_for_file(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Analyze file and provide optimization recommendations.
        
        Args:
            filepath: Path to file for optimization analysis
            
        Returns:
            Dictionary with optimization recommendations
        """
        with LoggedOperation(f"Optimization Analysis: {filepath}", self.logger):
            try:
                # Get file preview for quick analysis
                preview = self.get_analysis_preview(filepath, 50)  # Larger sample for better optimization
                
                if not preview.get('file_preview', {}).get('success', False):
                    return {'error': 'Could not analyze file for optimization'}
                
                file_size = preview['file_preview']['file_size']
                total_lines = preview['file_preview']['total_lines']
                estimated_words = preview['quick_analysis']['estimated_total_words']
                
                recommendations = {
                    'processing_recommendations': preview.get('recommendations', {}),
                    'performance_optimization': {
                        'estimated_memory_usage_mb': (file_size / 1024 / 1024) * 2,  # Rough estimate
                        'estimated_processing_time_seconds': estimated_words / 10000,  # Rough estimate
                        'recommended_batch_size': min(max(estimated_words // 1000, 1), 100),
                        'should_use_streaming': file_size > 50 * 1024 * 1024,  # > 50MB
                        'parallel_processing_recommended': estimated_words > 100000
                    },
                    'analysis_configuration': {
                        'optimal_top_words_count': min(max(estimated_words // 100, 10), 50),
                        'memory_efficient_mode': file_size > SystemConfig.MAX_MEMORY_USAGE // 2,
                        'detailed_analysis_recommended': estimated_words < 50000
                    },
                    'file_characteristics': {
                        'size_category': self._categorize_file_size(file_size),
                        'complexity_category': preview.get('recommendations', {}).get('processing_complexity', 'unknown'),
                        'estimated_language': preview.get('recommendations', {}).get('estimated_language', 'unknown')
                    }
                }
                
                self.logger.info(f"Optimization analysis complete for {filepath}")
                return recommendations
                
            except Exception as e:
                error_info = handle_exception(e, self.logger, "optimization_analysis")
                return {'error': error_info['error_message']}
    
    def _categorize_file_size(self, size_bytes: int) -> str:
        """Categorize file by size."""
        if size_bytes < 1024:  # < 1KB
            return 'tiny'
        elif size_bytes < 10 * 1024:  # < 10KB
            return 'small'
        elif size_bytes < 1024 * 1024:  # < 1MB
            return 'medium'
        elif size_bytes < 10 * 1024 * 1024:  # < 10MB
            return 'large'
        else:
            return 'very_large'
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status and health metrics.
        
        Returns:
            Dictionary with system status information
        """
        try:
            memory_status = SystemHelper.check_memory_usage()
            
            status = {
                'system_info': self.system_info,
                'memory_status': memory_status,
                'configuration': {
                    'top_words_count': self.config['top_words_count'],
                    'memory_limit_mb': self.config['memory_limit_mb'],
                    'output_formats': self.config['output_formats'],
                    'detailed_analysis_enabled': self.config['detailed_analysis']
                },
                'component_status': {
                    'file_handler_ready': bool(self.file_handler),
                    'text_processor_ready': bool(self.text_processor),
                    'report_generator_ready': bool(self.report_generator)
                },
                'supported_formats': {
                    'input_extensions': AnalysisConfig.SUPPORTED_EXTENSIONS,
                    'output_formats': self.report_generator.get_supported_formats(),
                    'encodings': AnalysisConfig.SUPPORTED_ENCODINGS
                },
                'performance_limits': {
                    'max_file_size_mb': AnalysisConfig.MAX_FILE_SIZE // (1024 * 1024),
                    'memory_limit_mb': self.config['memory_limit_mb'],
                    'processing_timeout_seconds': 3600  # 1 hour default
                },
                'health_status': 'healthy' if memory_status.get('system_memory_percent', 100) < 90 else 'warning'
            }
            
            return status
            
        except Exception as e:
            return {
                'error': f'Could not retrieve system status: {e}',
                'health_status': 'error'
            }
    
    def cleanup_resources(self) -> None:
        """Clean up system resources and temporary files."""
        try:
            # Clean up temporary files
            cleaned_files = SystemHelper.cleanup_temp_files("text_analyzer_*")
            
            # Reset performance monitors
            self.performance_monitor = PerformanceMonitor()
            
            # Log cleanup
            self.logger.info(f"Resource cleanup complete: {cleaned_files} temporary files removed")
            
        except Exception as e:
            self.logger.error(f"Resource cleanup failed: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup."""
        self.cleanup_resources()
        
        if exc_type is not None:
            self.logger.error(f"Analysis session ended with exception: {exc_val}")
        else:
            self.logger.info("Analysis session completed successfully")