"""
Strategic test suite for ReportGenerator component.
Comprehensive testing of report generation in multiple formats.
"""

import pytest
import json
import tempfile
from pathlib import Path
from collections import Counter
from unittest.mock import patch, mock_open

# Import components under test
import sys
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.text_analyzer.core.report_generator import (
    ReportGenerator,
    TextReportFormatter,
    JSONReportFormatter,
    CSVReportFormatter,
    HTMLReportFormatter
)
from src.text_analyzer.utils import (
    ReportFormattingError,
    ReportOutputError
)


class TestReportFormatters:
    """Test suite for individual report formatters."""
    
    @pytest.fixture
    def sample_data(self):
        """Sample data for report generation testing."""
        return {
            'version': '1.0.0',
            'file_info': {
                'path': '/test/sample.txt',
                'size': 1024,
                'encoding': 'utf-8',
                'lines': 10,
                'checksum': 'abc123'
            },
            'analysis_summary': {
                'total_words': 150,
                'unique_words': 75,
                'vocabulary_richness': 0.5,
                'avg_word_length': 4.5,
                'processing_time': 0.123
            },
            'top_words': [
                ('the', 15),
                ('and', 10),
                ('fox', 8),
                ('dog', 6),
                ('quick', 5)
            ],
            'total_words': 150,
            'processing_stats': {
                'original_length': 800,
                'processed_length': 700,
                'words_per_second': 1200,
                'operations_applied': ['clean', 'normalize'],
                'memory_mb': 2.5
            },
            'word_length_distribution': {
                3: 25,
                4: 30,
                5: 20,
                6: 15,
                7: 10
            },
            'character_distribution': {
                'e': 45,
                't': 30,
                'a': 25,
                'o': 20,
                'i': 15
            }
        }
    
    def test_text_formatter_creation(self):
        """Test TextReportFormatter creation."""
        formatter = TextReportFormatter()
        assert formatter is not None
        assert formatter.width == 80
        assert hasattr(formatter, 'separator_char')
        assert hasattr(formatter, 'sub_separator_char')
    
    def test_text_formatter_custom_width(self):
        """Test TextReportFormatter with custom width."""
        formatter = TextReportFormatter(width=100)
        assert formatter.width == 100
    
    def test_text_formatter_format_report(self, sample_data):
        """Test text report formatting."""
        formatter = TextReportFormatter()
        report = formatter.format_report(sample_data)
        
        assert isinstance(report, str)
        assert len(report) > 0
        
        # Check for expected sections
        assert 'TEXT ANALYSIS REPORT' in report
        assert 'FILE INFORMATION' in report
        assert 'ANALYSIS SUMMARY' in report
        assert 'MOST FREQUENT WORDS' in report
        assert 'PROCESSING STATISTICS' in report
        
        # Check for specific data
        assert '/test/sample.txt' in report
        assert '150' in report  # total words
        assert '75' in report   # unique words
        assert 'the' in report  # top word
    
    def test_text_formatter_file_extension(self):
        """Test text formatter file extension."""
        formatter = TextReportFormatter()
        assert formatter.get_file_extension() == '.txt'
    
    def test_json_formatter_creation(self):
        """Test JSONReportFormatter creation."""
        formatter = JSONReportFormatter()
        assert formatter is not None
    
    def test_json_formatter_format_report(self, sample_data):
        """Test JSON report formatting."""
        formatter = JSONReportFormatter()
        report = formatter.format_report(sample_data)
        
        assert isinstance(report, str)
        
        # Should be valid JSON
        parsed_json = json.loads(report)
        assert isinstance(parsed_json, dict)
        
        # Check structure
        assert 'metadata' in parsed_json
        assert 'file_information' in parsed_json
        assert 'analysis_summary' in parsed_json
        assert 'top_words' in parsed_json
        assert 'processing_statistics' in parsed_json
        
        # Check metadata
        metadata = parsed_json['metadata']
        assert metadata['analyzer_version'] == '1.0.0'
        assert 'generated_at' in metadata
        
        # Check top words structure
        top_words = parsed_json['top_words']
        assert isinstance(top_words, list)
        assert len(top_words) > 0
        
        first_word = top_words[0]
        assert 'rank' in first_word
        assert 'word' in first_word
        assert 'count' in first_word
        assert 'percentage' in first_word
    
    def test_json_formatter_file_extension(self):
        """Test JSON formatter file extension."""
        formatter = JSONReportFormatter()
        assert formatter.get_file_extension() == '.json'
    
    def test_csv_formatter_creation(self):
        """Test CSVReportFormatter creation."""
        formatter = CSVReportFormatter()
        assert formatter is not None
    
    def test_csv_formatter_format_report(self, sample_data):
        """Test CSV report formatting."""
        formatter = CSVReportFormatter()
        report = formatter.format_report(sample_data)
        
        assert isinstance(report, str)
        assert len(report) > 0
        
        # Check for CSV sections
        assert '# Top Words' in report
        assert 'Rank,Word,Count,Percentage' in report
        assert '# Word Length Distribution' in report
        assert 'Length,Count' in report
        assert '# Summary Statistics' in report
        assert 'Metric,Value' in report
        
        # Check for data
        assert 'the' in report
        assert '150' in report  # total words
    
    def test_csv_formatter_file_extension(self):
        """Test CSV formatter file extension."""
        formatter = CSVReportFormatter()
        assert formatter.get_file_extension() == '.csv'
    
    def test_html_formatter_creation(self):
        """Test HTMLReportFormatter creation."""
        formatter = HTMLReportFormatter()
        assert formatter is not None
    
    def test_html_formatter_format_report(self, sample_data):
        """Test HTML report formatting."""
        formatter = HTMLReportFormatter()
        report = formatter.format_report(sample_data)
        
        assert isinstance(report, str)
        assert len(report) > 0
        
        # Check for HTML structure
        assert '<!DOCTYPE html>' in report
        assert '<html>' in report
        assert '<head>' in report
        assert '<body>' in report
        assert '</html>' in report
        
        # Check for content sections
        assert 'Text Analysis Report' in report
        assert 'File Information' in report
        assert 'Analysis Summary' in report
        assert 'Most Frequent Words' in report
        
        # Check for CSS styling
        assert '<style>' in report
        assert 'font-family' in report
        
        # Check for data
        assert '/test/sample.txt' in report
        assert '150' in report  # total words
    
    def test_html_formatter_file_extension(self):
        """Test HTML formatter file extension."""
        formatter = HTMLReportFormatter()
        assert formatter.get_file_extension() == '.html'


class TestReportGenerator:
    """Comprehensive test suite for ReportGenerator class."""
    
    @pytest.fixture
    def report_generator(self):
        """Create ReportGenerator instance."""
        return ReportGenerator()
    
    @pytest.fixture
    def sample_analysis_data(self):
        """Sample analysis data for report generation."""
        return {
            'word_frequencies': Counter({
                'the': 15,
                'and': 10,
                'fox': 8,
                'dog': 6,
                'quick': 5,
                'brown': 4,
                'lazy': 3,
                'jumps': 2,
                'over': 2,
                'cat': 1
            }),
            'file_info': {
                'path': '/test/document.txt',
                'size': 2048,
                'encoding': 'utf-8',
                'lines': 25,
                'checksum': 'def456'
            },
            'processing_stats': {
                'processing_time': 0.256,
                'original_length': 1500,
                'processed_length': 1200,
                'operations_applied': ['unicode_normalization', 'case_normalization']
            },
            'word_length_distribution': {
                2: 5,
                3: 15,
                4: 20,
                5: 25,
                6: 15,
                7: 10,
                8: 5
            },
            'character_distribution': {
                'e': 50,
                't': 35,
                'a': 30,
                'o': 25,
                'i': 20,
                'n': 18,
                's': 15,
                'h': 12,
                'r': 10,
                'd': 8
            }
        }
    
    def test_report_generator_initialization(self, report_generator):
        """Test ReportGenerator initialization."""
        assert report_generator is not None
        assert hasattr(report_generator, 'logger')
        assert hasattr(report_generator, 'formatters')
        assert hasattr(report_generator, 'version')
        
        # Check available formatters
        formatters = report_generator.formatters
        assert 'txt' in formatters
        assert 'json' in formatters
        assert 'csv' in formatters
        assert 'html' in formatters
    
    def test_prepare_report_data(self, report_generator, sample_analysis_data):
        """Test report data preparation."""
        report_data = report_generator._prepare_report_data(sample_analysis_data)
        
        assert isinstance(report_data, dict)
        
        # Check required sections
        assert 'version' in report_data
        assert 'file_info' in report_data
        assert 'analysis_summary' in report_data
        assert 'top_words' in report_data
        assert 'total_words' in report_data
        assert 'processing_stats' in report_data
        
        # Check analysis summary calculations
        summary = report_data['analysis_summary']
        assert summary['total_words'] == 56  # Sum of frequencies
        assert summary['unique_words'] == 10  # Number of unique words
        assert 0 <= summary['vocabulary_richness'] <= 1
        assert summary['avg_word_length'] > 0
        
        # Check top words
        top_words = report_data['top_words']
        assert isinstance(top_words, list)
        assert len(top_words) <= 10  # Default top count
        assert top_words[0][0] == 'the'  # Most frequent word
        assert top_words[0][1] == 15    # Its count
    
    def test_generate_report_txt_format(self, report_generator, sample_analysis_data):
        """Test report generation in TXT format."""
        result = report_generator.generate_report(sample_analysis_data, 'txt')
        
        assert result['success'] is True
        assert result['format'] == 'txt'
        assert 'content' in result
        assert 'generated_at' in result
        
        content = result['content']
        assert isinstance(content, str)
        assert len(content) > 0
        assert 'TEXT ANALYSIS REPORT' in content
    
    def test_generate_report_json_format(self, report_generator, sample_analysis_data):
        """Test report generation in JSON format."""
        result = report_generator.generate_report(sample_analysis_data, 'json')
        
        assert result['success'] is True
        assert result['format'] == 'json'
        
        content = result['content']
        # Should be valid JSON
        parsed = json.loads(content)
        assert isinstance(parsed, dict)
    
    def test_generate_report_csv_format(self, report_generator, sample_analysis_data):
        """Test report generation in CSV format."""
        result = report_generator.generate_report(sample_analysis_data, 'csv')
        
        assert result['success'] is True
        assert result['format'] == 'csv'
        
        content = result['content']
        assert 'Rank,Word,Count,Percentage' in content
    
    def test_generate_report_html_format(self, report_generator, sample_analysis_data):
        """Test report generation in HTML format."""
        result = report_generator.generate_report(sample_analysis_data, 'html')