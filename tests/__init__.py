"""
Test suite initialization for Text Analyzer.
"""

# Test suite metadata
__version__ = "1.0.0"
__author__ = "Strategic Analysis Systems - Test Suite"
__description__ = "Comprehensive test suite for Text Analyzer system"

# Test configuration
TEST_CONFIG = {
    'coverage_threshold': 95,  # Minimum coverage percentage
    'performance_threshold': {
        'small_file_processing': 1.0,    # seconds
        'medium_file_processing': 5.0,   # seconds
        'large_file_processing': 30.0,   # seconds
        'memory_usage_limit': 512        # MB
    },
    'supported_formats': ['txt', 'json', 'csv', 'html'],
    'supported_encodings': ['utf-8', 'utf-16', 'latin-1', 'cp1252'],
    'test_data_sizes': {
        'small': 1000,      # words
        'medium': 10000,    # words  
        'large': 100000,    # words
        'xlarge': 1000000   # words
    }
}

# Test categories
TEST_CATEGORIES = {
    'unit': 'Individual component testing',
    'integration': 'Component interaction testing',
    'performance': 'Performance and benchmarking tests',
    'regression': 'Regression prevention tests',
    'stress': 'System stress and load testing',
    'edge_cases': 'Edge case and error condition testing'
}

# Export test utilities
from .conftest import (
    # Main fixtures
    sample_texts,
    create_test_files, 
    complete_analysis_data,
    
    # Utility fixtures
    temp_file,
    temp_dir,
    performance_timer,
    memory_monitor,
    
    # Configuration fixtures
    default_config,
    custom_config,
    
    # Test data fixtures
    sample_word_frequencies,
    sample_file_info,
    sample_processing_stats,
    
    # Validation fixtures
    validation_test_cases,
    error_conditions,
    regression_test_data,
    
    # Environment fixtures
    integration_environment,
    test_results_collector
)

__all__ = [
    'TEST_CONFIG',
    'TEST_CATEGORIES', 
    'sample_texts',
    'create_test_files',
    'complete_analysis_data',
    'temp_file',
    'temp_dir',
    'performance_timer',
    'memory_monitor',
    'default_config',
    'custom_config',
    'sample_word_frequencies',
    'sample_file_info',
    'sample_processing_stats',
    'validation_test_cases',
    'error_conditions',
    'regression_test_data',
    'integration_environment',
    'test_results_collector'
]