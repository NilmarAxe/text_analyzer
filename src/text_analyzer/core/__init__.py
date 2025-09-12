"""
Core module initialization for Text Analyzer.
"""

from .analyzer import TextAnalyzer, AnalysisResult
from .file_handler import FileHandler, FileInfo
from .text_processor import TextProcessor, WordAnalysis, ProcessingStats
from .report_generator import (
    ReportGenerator,
    ReportFormatter,
    TextReportFormatter,
    JSONReportFormatter,
    CSVReportFormatter,
    HTMLReportFormatter
)

# Export all core classes and functions
__all__ = [
    # Main analyzer
    'TextAnalyzer',
    'AnalysisResult',
    
    # File handling
    'FileHandler',
    'FileInfo',
    
    # Text processing
    'TextProcessor',
    'WordAnalysis',
    'ProcessingStats',
    
    # Report generation
    'ReportGenerator',
    'ReportFormatter',
    'TextReportFormatter',
    'JSONReportFormatter',
    'CSVReportFormatter',
    'HTMLReportFormatter'
]