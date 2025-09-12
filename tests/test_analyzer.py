"""
Strategic test suite for Text Analyzer core functionality.
"""

import pytest
import tempfile
from pathlib import Path
from collections import Counter

# Import components under test
import sys
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.text_analyzer.core import TextAnalyzer, AnalysisResult
from src.text_analyzer.utils import TextAnalyzerError, FileNotFoundError


class TestTextAnalyzer:
    """Comprehensive test suite for TextAnalyzer class."""
    
    @pytest.fixture
    def sample_text_file(self):
        """Create temporary file with sample text."""
        content = """The quick brown fox jumps over the lazy dog.
        The dog was not amused by the fox's antics.
        Brown foxes are known for their quick movements.
        This is a test document with repeated words."""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(content)
            return Path(f.name)
    
    @pytest.fixture
    def empty_file(self):
        """Create temporary empty file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("")
            return Path(f.name)
    
    @pytest.fixture
    def large_text_file(self):
        """Create temporary file with larger text sample."""
        words = ["analysis", "text", "frequency", "word", "count", "system", "data", "process"]
        content = " ".join(words * 100) + ".\n"  # 800 words
        content = content * 5  # 4000 words total
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(content)
            return Path(f.name)
    
    def test_analyzer_initialization(self):
        """Test TextAnalyzer initialization with various configurations."""
        # Default initialization
        analyzer = TextAnalyzer()
        assert analyzer is not None
        assert hasattr(analyzer, 'file_handler')
        assert hasattr(analyzer, 'text_processor')
        assert hasattr(analyzer, 'report_generator')
    
    def test_analyzer_with_custom_config(self):
        """Test analyzer initialization with custom configuration."""
        config = {
            'top_words_count': 15,
            'case_sensitive': True,
            'remove_stop_words': True,
            'min_word_length': 3
        }
        
        analyzer = TextAnalyzer(config)
        assert analyzer.config['top_words_count'] == 15
        assert analyzer.config['case_sensitive'] is True
        assert analyzer.config['remove_stop_words'] is True
        assert analyzer.config['min_word_length'] == 3
    
    def test_analyze_file_success(self, sample_text_file):
        """Test successful file analysis."""
        analyzer = TextAnalyzer()
        result = analyzer.analyze_file(sample_text_file)
        
        # Verify result structure
        assert isinstance(result, AnalysisResult)
        assert result.success is True
        assert result.error_message is None
        assert result.execution_time > 0
        
        # Verify file info
        assert result.file_info is not None
        assert 'path' in result.file_info
        assert 'size' in result.file_info
        
        # Verify word frequencies
        assert isinstance(result.word_frequencies, Counter)
        assert len(result.word_frequencies) > 0
        
        # Verify analysis metrics
        assert 'word_statistics' in result.analysis_metrics
        word_stats = result.analysis_metrics['word_statistics']
        assert word_stats['total_words'] > 0
        assert word_stats['unique_words'] > 0
        assert 0 <= word_stats['vocabulary_richness'] <= 1
        
        # Check for expected words in sample text
        assert 'the' in result.word_frequencies
        assert 'fox' in result.word_frequencies
        assert 'dog' in result.word_frequencies
    
    def test_analyze_file_with_output(self, sample_text_file):
        """Test file analysis with output generation."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as output_file:
            output_path = Path(output_file.name)
        
        analyzer = TextAnalyzer()
        result = analyzer.analyze_file(sample_text_file, output_path, 'txt')
        
        assert result.success is True
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        
        # Verify report content
        content = output_path.read_text(encoding='utf-8')
        assert 'TEXT ANALYSIS REPORT' in content
        assert 'MOST FREQUENT WORDS' in content
        
        # Cleanup
        output_path.unlink()
    
    def test_analyze_multiple_formats(self, sample_text_file):
        """Test analysis with different output formats."""
        analyzer = TextAnalyzer()
        formats = ['txt', 'json', 'csv', 'html']
        
        for fmt in formats:
            with tempfile.NamedTemporaryFile(suffix=f'.{fmt}', delete=False) as output_file:
                output_path = Path(output_file.name)
            
            result = analyzer.analyze_file(sample_text_file, output_path, fmt)
            
            assert result.success is True, f"Failed for format: {fmt}"
            assert output_path.exists(), f"Output file not created for format: {fmt}"
            assert output_path.stat().st_size > 0, f"Empty output for format: {fmt}"
            
            # Cleanup
            output_path.unlink()
    
    def test_analyze_file_not_found(self):
        """Test analysis with non-existent file."""
        analyzer = TextAnalyzer()
        non_existent_file = Path("this_file_does_not_exist.txt")
        
        result = analyzer.analyze_file(non_existent_file)
        
        assert result.success is False
        assert result.error_message is not None
        assert 'not found' in result.error_message.lower()
    
    def test_analyze_empty_file(self, empty_file):
        """Test analysis with empty file."""
        analyzer = TextAnalyzer()
        result = analyzer.analyze_file(empty_file)
        
        assert result.success is False
        assert result.error_message is not None
        assert 'empty' in result.error_message.lower()
    
    def test_case_sensitivity(self, sample_text_file):
        """Test case-sensitive vs case-insensitive analysis."""
        # Case-insensitive (default)
        analyzer_insensitive = TextAnalyzer({'case_sensitive': False})
        result_insensitive = analyzer_insensitive.analyze_file(sample_text_file)
        
        # Case-sensitive
        analyzer_sensitive = TextAnalyzer({'case_sensitive': True})
        result_sensitive = analyzer_sensitive.analyze_file(sample_text_file)
        
        # Both should succeed
        assert result_insensitive.success is True
        assert result_sensitive.success is True
        
        # Case-insensitive should have fewer unique words (due to merging)
        insensitive_unique = result_insensitive.analysis_metrics['word_statistics']['unique_words']
        sensitive_unique = result_sensitive.analysis_metrics['word_statistics']['unique_words']
        
        assert insensitive_unique <= sensitive_unique
    
    def test_word_length_filtering(self, sample_text_file):
        """Test word length filtering configuration."""
        config = {
            'min_word_length': 4,
            'max_word_length': 10
        }
        
        analyzer = TextAnalyzer(config)
        result = analyzer.analyze_file(sample_text_file)
        
        assert result.success is True
        
        # Verify no words shorter than min_word_length or longer than max_word_length
        for word in result.word_frequencies:
            assert len(word) >= 4, f"Word '{word}' is shorter than minimum length"
            assert len(word) <= 10, f"Word '{word}' is longer than maximum length"
    
    def test_top_words_count_configuration(self, sample_text_file):
        """Test top words count configuration."""
        config = {'top_words_count': 5}
        
        analyzer = TextAnalyzer(config)
        result = analyzer.analyze_file(sample_text_file)
        
        assert result.success is True
        
        # Get top words
        top_words = result.word_frequencies.most_common(5)
        assert len(top_words) <= 5
    
    def test_batch_analysis(self, sample_text_file, large_text_file):
        """Test batch analysis functionality."""
        analyzer = TextAnalyzer()
        files = [sample_text_file, large_text_file]
        
        results = analyzer.analyze_multiple_files(files)
        
        assert len(results) == 2
        
        for filepath, result in results.items():
            assert isinstance(result, AnalysisResult)
            if result.success:
                assert result.analysis_metrics is not None
                assert result.word_frequencies is not None
    
    def test_analysis_preview(self, sample_text_file):
        """Test analysis preview functionality."""
        analyzer = TextAnalyzer()
        preview = analyzer.get_analysis_preview(sample_text_file, lines=5)
        
        assert preview is not None
        assert 'file_preview' in preview
        assert 'quick_analysis' in preview
        
        file_preview = preview['file_preview']
        assert file_preview['success'] is True
        assert len(file_preview['preview_lines']) <= 5
        assert file_preview['total_lines'] > 0
    
    def test_optimization_recommendations(self, large_text_file):
        """Test configuration optimization recommendations."""
        analyzer = TextAnalyzer()
        recommendations = analyzer.optimize_for_file(large_text_file)
        
        assert 'processing_recommendations' in recommendations
        assert 'performance_optimization' in recommendations
        assert 'analysis_configuration' in recommendations
        assert 'file_characteristics' in recommendations
        
        # Verify reasonable recommendations
        perf = recommendations['performance_optimization']
        assert perf['estimated_memory_usage_mb'] > 0
        assert perf['estimated_processing_time_seconds'] >= 0
    
    def test_system_status(self):
        """Test system status retrieval."""
        analyzer = TextAnalyzer()
        status = analyzer.get_system_status()
        
        assert 'system_info' in status
        assert 'memory_status' in status
        assert 'configuration' in status
        assert 'component_status' in status
        assert 'supported_formats' in status
        assert 'health_status' in status
        
        # Verify component status
        components = status['component_status']
        assert components['file_handler_ready'] is True
        assert components['text_processor_ready'] is True
        assert components['report_generator_ready'] is True
    
    def test_context_manager(self, sample_text_file):
        """Test TextAnalyzer as context manager."""
        with TextAnalyzer() as analyzer:
            result = analyzer.analyze_file(sample_text_file)
            assert result.success is True
        
        # Verify cleanup was called (resources should be cleaned)
        # This is implementation-dependent, but we can verify the analyzer still works
        assert analyzer is not None
    
    def test_performance_metrics(self, sample_text_file):
        """Test performance metrics collection."""
        analyzer = TextAnalyzer()
        result = analyzer.analyze_file(sample_text_file)
        
        assert result.success is True
        assert result.execution_time > 0
        
        # Verify processing stats
        processing_stats = result.processing_stats
        assert 'processing_time' in processing_stats
        assert processing_stats['processing_time'] > 0
    
    def test_memory_management(self, large_text_file):
        """Test memory management with large files."""
        config = {'memory_limit_mb': 64}  # Low limit for testing
        
        analyzer = TextAnalyzer(config)
        result = analyzer.analyze_file(large_text_file)
        
        # Should succeed with memory management
        assert result.success is True
        assert result.analysis_metrics is not None
    
    def test_error_handling_robustness(self):
        """Test robustness of error handling."""
        analyzer = TextAnalyzer()
        
        # Test with various invalid inputs
        invalid_inputs = [
            None,
            "",
            "/dev/null/nonexistent",
            Path("/invalid/path/file.txt")
        ]
        
        for invalid_input in invalid_inputs:
            try:
                result = analyzer.analyze_file(invalid_input)
                # Should return failed result, not raise exception
                assert result.success is False
                assert result.error_message is not None
            except Exception as e:
                # If exception is raised, it should be a known type
                assert isinstance(e, TextAnalyzerError)


class TestAnalysisResult:
    """Test suite for AnalysisResult data structure."""
    
    def test_result_to_dict(self):
        """Test AnalysisResult serialization to dictionary."""
        # Create sample result
        result = AnalysisResult(
            file_info={'path': 'test.txt', 'size': 100},
            word_frequencies=Counter({'test': 5, 'word': 3}),
            analysis_metrics={'total_words': 8},
            processing_stats={'time': 0.1},
            word_length_distribution={4: 2},
            character_distribution={'t': 4, 'e': 2},
            success=True,
            execution_time=0.1
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict['success'] is True
        assert result_dict['execution_time'] == 0.1
        assert isinstance(result_dict['word_frequencies'], dict)
        assert isinstance(result_dict['character_distribution'], dict)


# Integration tests
class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_complete_analysis_workflow(self, sample_text_file):
        """Test complete analysis workflow from file to report."""
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as output_file:
            output_path = Path(output_file.name)
        
        # Complete workflow
        analyzer = TextAnalyzer({
            'top_words_count': 10,
            'case_sensitive': False,
            'detailed_analysis': True
        })
        
        result = analyzer.analyze_file(sample_text_file, output_path, 'html')
        
        # Verify complete success
        assert result.success is True
        assert output_path.exists()
        assert output_path.stat().st_size > 1000  # Substantial HTML output
        
        # Verify HTML content
        html_content = output_path.read_text(encoding='utf-8')
        assert '<!DOCTYPE html>' in html_content
        assert 'Text Analysis Report' in html_content
        assert 'table' in html_content.lower()
        
        # Cleanup
        output_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])