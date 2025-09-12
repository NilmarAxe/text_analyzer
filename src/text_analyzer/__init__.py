"""
Text Analyzer - Strategic Text Analysis System
"""

from .core import TextAnalyzer, AnalysisResult, FileHandler, TextProcessor, ReportGenerator
from .cli import CLIInterface, main, ArgumentParser
from .utils import (
    TextAnalyzerError,
    FileHandlerError,
    TextProcessorError,
    ReportGeneratorError,
    CLIError,
    PerformanceMonitor,
    FormattingHelper
)

# Package metadata
__version__ = "1.0.0"
__author__ = "Strategic Analysis Systems"
__description__ = "Comprehensive text analysis system with systematic approach to frequency analysis and statistical reporting"

# Export main classes and functions
__all__ = [
    # Core components
    'TextAnalyzer',
    'AnalysisResult',
    'FileHandler', 
    'TextProcessor',
    'ReportGenerator',
    
    # CLI components
    'CLIInterface',
    'main',
    'ArgumentParser',
    
    # Utilities
    'TextAnalyzerError',
    'FileHandlerError',
    'TextProcessorError', 
    'ReportGeneratorError',
    'CLIError',
    'PerformanceMonitor',
    'FormattingHelper',
    
    # Package info
    '__version__',
    '__author__',
    '__description__'
]