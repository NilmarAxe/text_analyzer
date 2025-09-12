"""
Configuration settings for the Text Analyzer system.
"""

import os
from pathlib import Path
from typing import List, Dict, Any

# Base configuration
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent

class AnalysisConfig:
    """Core analysis configuration parameters."""
    
    # Text processing settings
    MIN_WORD_LENGTH = 2
    MAX_WORD_LENGTH = 50
    DEFAULT_TOP_WORDS_COUNT = 10
    CASE_SENSITIVE = False
    
    # Supported file encodings (in order of preference)
    SUPPORTED_ENCODINGS = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
    
    # File size limits (in bytes)
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    MIN_FILE_SIZE = 1  # 1 byte
    
    # Supported file extensions
    SUPPORTED_EXTENSIONS = ['.txt', '.text', '.log', '.md']
    
    # Text cleaning patterns
    WORD_PATTERN = r'\b[a-zA-ZÀ-ÿ]+\b'  # Includes accented characters
    PUNCTUATION_PATTERN = r'[^\w\s]'
    
    # Stop words (common words to optionally filter)
    DEFAULT_STOP_WORDS = {
        'english': {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 
            'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 
            'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 
            'him', 'her', 'us', 'them', 'my', 'your', 'his', 'our', 'their',
            'a', 'an'
        }
    }

class OutputConfig:
    """Output and reporting configuration."""
    
    # Default output directory
    OUTPUT_DIR = PROJECT_ROOT / 'output'
    
    # Report formatting
    REPORT_WIDTH = 80
    SEPARATOR_CHAR = '='
    SUB_SEPARATOR_CHAR = '-'
    
    # File naming patterns
    REPORT_FILENAME_PATTERN = "analysis_report_{timestamp}.txt"
    TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
    
    # Export formats
    SUPPORTED_EXPORT_FORMATS = ['txt', 'json', 'csv', 'html']
    
    # Console output colors (if supported)
    COLORS = {
        'header': '\033[95m',
        'blue': '\033[94m',
        'green': '\033[92m',
        'warning': '\033[93m',
        'error': '\033[91m',
        'reset': '\033[0m',
        'bold': '\033[1m',
        'underline': '\033[4m'
    }

class SystemConfig:
    """System-level configuration settings."""
    
    # Performance settings
    CHUNK_SIZE = 8192  # For file reading
    MAX_MEMORY_USAGE = 512 * 1024 * 1024  # 512MB
    
    # Logging configuration
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = PROJECT_ROOT / 'logs' / 'text_analyzer.log'
    
    # Error handling
    ENABLE_DEBUG_MODE = False
    ENABLE_STACK_TRACES = False
    
    # Validation settings
    STRICT_VALIDATION = True
    ENABLE_WARNINGS = True

class EnvironmentConfig:
    """Environment-specific configuration."""
    
    @staticmethod
    def get_config() -> Dict[str, Any]:
        """Get configuration based on environment variables."""
        env = os.getenv('TEXT_ANALYZER_ENV', 'development')
        
        config = {
            'development': {
                'debug': True,
                'log_level': 'DEBUG',
                'strict_validation': False,
                'enable_warnings': True
            },
            'testing': {
                'debug': False,
                'log_level': 'WARNING',
                'strict_validation': True,
                'enable_warnings': False
            },
            'production': {
                'debug': False,
                'log_level': 'ERROR',
                'strict_validation': True,
                'enable_warnings': False
            }
        }
        
        return config.get(env, config['development'])

# Global configuration instance
def get_settings() -> Dict[str, Any]:
    """Get complete configuration dictionary."""
    env_config = EnvironmentConfig.get_config()
    
    return {
        'analysis': AnalysisConfig,
        'output': OutputConfig,
        'system': SystemConfig,
        'environment': env_config
    }

# Configuration validation
def validate_configuration() -> bool:
    """Validate configuration settings integrity."""
    try:
        # Ensure required directories exist
        OutputConfig.OUTPUT_DIR.mkdir(exist_ok=True)
        SystemConfig.LOG_FILE.parent.mkdir(exist_ok=True)
        
        # Validate file size limits
        assert AnalysisConfig.MIN_FILE_SIZE < AnalysisConfig.MAX_FILE_SIZE
        
        # Validate word length constraints
        assert 0 < AnalysisConfig.MIN_WORD_LENGTH <= AnalysisConfig.MAX_WORD_LENGTH
        
        return True
        
    except (AssertionError, OSError) as e:
        print(f"Configuration validation failed: {e}")
        return False