"""
Comprehensive logging configuration system.
Strategic logging setup for debugging, monitoring, and system analysis.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
from .settings import SystemConfig, EnvironmentConfig

class LoggerSetup:
    """
    Centralized logger configuration following INTJ systematic approach.
    Provides structured logging with multiple output channels and severity levels.
    """
    
    _loggers = {}
    _configured = False
    
    @classmethod
    def setup_logging(cls, 
                     log_level: Optional[str] = None,
                     log_file: Optional[Path] = None,
                     enable_console: bool = True,
                     enable_file: bool = True) -> None:
        """
        Configure comprehensive logging system.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Path to log file
            enable_console: Enable console output
            enable_file: Enable file output
        """
        if cls._configured:
            return
            
        # Environment-based configuration
        env_config = EnvironmentConfig.get_config()
        log_level = log_level or env_config.get('log_level', 'INFO')
        log_file = log_file or SystemConfig.LOG_FILE
        
        # Ensure log directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Root logger configuration
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Formatter configuration
        detailed_formatter = logging.Formatter(
            '%(asctime)s | %(name)-20s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        
        # Console handler setup
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(simple_formatter)
            root_logger.addHandler(console_handler)
        
        # File handler setup
        if enable_file:
            # Rotating file handler to manage log size
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(detailed_formatter)
            root_logger.addHandler(file_handler)
        
        # Error handler for critical issues
        error_handler = logging.StreamHandler(sys.stderr)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(error_handler)
        
        cls._configured = True
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get configured logger instance.
        
        Args:
            name: Logger name (typically __name__)
            
        Returns:
            Configured logger instance
        """
        if not cls._configured:
            cls.setup_logging()
        
        if name not in cls._loggers:
            logger = logging.getLogger(name)
            cls._loggers[name] = logger
            
        return cls._loggers[name]
    
    @classmethod
    def set_level(cls, name: str, level: str) -> None:
        """
        Dynamically adjust logger level.
        
        Args:
            name: Logger name
            level: New logging level
        """
        logger = cls.get_logger(name)
        logger.setLevel(getattr(logging, level.upper()))

# Specialized loggers for different system components
def get_analyzer_logger() -> logging.Logger:
    """Get logger for core analysis operations."""
    return LoggerSetup.get_logger('text_analyzer.core')

def get_file_handler_logger() -> logging.Logger:
    """Get logger for file operations."""
    return LoggerSetup.get_logger('text_analyzer.file_handler')

def get_processor_logger() -> logging.Logger:
    """Get logger for text processing operations."""
    return LoggerSetup.get_logger('text_analyzer.processor')

def get_reporter_logger() -> logging.Logger:
    """Get logger for report generation."""
    return LoggerSetup.get_logger('text_analyzer.reporter')

def get_cli_logger() -> logging.Logger:
    """Get logger for command-line interface."""
    return LoggerSetup.get_logger('text_analyzer.cli')

def get_system_logger() -> logging.Logger:
    """Get logger for system-level operations."""
    return LoggerSetup.get_logger('text_analyzer.system')

# Performance monitoring logger
class PerformanceLogger:
    """Specialized logger for performance monitoring and optimization analysis."""
    
    def __init__(self):
        self.logger = LoggerSetup.get_logger('text_analyzer.performance')
    
    def log_file_processing_time(self, filepath: str, processing_time: float, file_size: int):
        """Log file processing performance metrics."""
        rate = file_size / processing_time if processing_time > 0 else 0
        self.logger.info(
            f"File processed: {filepath} | "
            f"Size: {file_size:,} bytes | "
            f"Time: {processing_time:.3f}s | "
            f"Rate: {rate:,.0f} bytes/s"
        )
    
    def log_analysis_metrics(self, total_words: int, unique_words: int, processing_time: float):
        """Log text analysis performance metrics."""
        word_rate = total_words / processing_time if processing_time > 0 else 0
        self.logger.info(
            f"Analysis completed | "
            f"Total words: {total_words:,} | "
            f"Unique words: {unique_words:,} | "
            f"Time: {processing_time:.3f}s | "
            f"Rate: {word_rate:,.0f} words/s"
        )
    
    def log_memory_usage(self, operation: str, memory_mb: float):
        """Log memory usage for specific operations."""
        self.logger.debug(f"Memory usage - {operation}: {memory_mb:.2f} MB")

# Context manager for operation logging
class LoggedOperation:
    """Context manager for automatic operation logging with performance tracking."""
    
    def __init__(self, operation_name: str, logger: logging.Logger):
        self.operation_name = operation_name
        self.logger = logger
        self.start_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        self.logger.info(f"Starting operation: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        duration = time.time() - self.start_time
        
        if exc_type is None:
            self.logger.info(f"Operation completed: {self.operation_name} ({duration:.3f}s)")
        else:
            self.logger.error(f"Operation failed: {self.operation_name} - {exc_val} ({duration:.3f}s)")
        
        return False

# Initialize logging system
def initialize_logging():
    """Initialize the complete logging system."""
    LoggerSetup.setup_logging()
    
    system_logger = get_system_logger()
    system_logger.info("Text Analyzer logging system initialized")
    system_logger.debug(f"Log file: {SystemConfig.LOG_FILE}")
    
    return True