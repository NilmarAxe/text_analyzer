"""
Utilities module initialization.
"""

from .exceptions import (
    TextAnalyzerError,
    FileHandlerError,
    FileNotFoundError,
    FileAccessError,
    FileEncodingError,
    FileSizeError,
    EmptyFileError,
    TextProcessorError,
    InvalidTextError,
    TextProcessingError,
    WordExtractionError,
    AnalysisError,
    InsufficientDataError,
    AnalysisConfigurationError,
    ReportGeneratorError,
    ReportFormattingError,
    ReportOutputError,
    CLIError,
    InvalidArgumentError,
    CommandExecutionError,
    SystemError,
    MemoryError,
    ConfigurationError,
    ValidationError,
    ParameterValidationError,
    DataValidationError,
    handle_exception,
    raise_with_context
)

from .validators import (
    FileValidator,
    TextValidator,
    ParameterValidator,
    ConfigurationValidator,
    CLIValidator,
    validate_all_parameters,
    create_validation_report
)

from .helpers import (
    PerformanceMonitor,
    timing_decorator,
    FileSystemHelper,
    TextHelper,
    DataStructureHelper,
    FormattingHelper,
    ConfigurationHelper,
    SystemHelper,
    safe_divide,
    ensure_list,
    batch_process,
    log_execution_context
)

# Export all utility classes and functions
__all__ = [
    # Exceptions
    'TextAnalyzerError',
    'FileHandlerError',
    'FileNotFoundError', 
    'FileAccessError',
    'FileEncodingError',
    'FileSizeError',
    'EmptyFileError',
    'TextProcessorError',
    'InvalidTextError',
    'TextProcessingError',
    'WordExtractionError',
    'AnalysisError',
    'InsufficientDataError',
    'AnalysisConfigurationError',
    'ReportGeneratorError',
    'ReportFormattingError',
    'ReportOutputError',
    'CLIError',
    'InvalidArgumentError',
    'CommandExecutionError',
    'SystemError',
    'MemoryError',
    'ConfigurationError',
    'ValidationError',
    'ParameterValidationError',
    'DataValidationError',
    'handle_exception',
    'raise_with_context',
    
    # Validators
    'FileValidator',
    'TextValidator',
    'ParameterValidator',
    'ConfigurationValidator',
    'CLIValidator',
    'validate_all_parameters',
    'create_validation_report',
    
    # Helpers
    'PerformanceMonitor',
    'timing_decorator',
    'FileSystemHelper',
    'TextHelper',
    'DataStructureHelper',
    'FormattingHelper',
    'ConfigurationHelper',
    'SystemHelper',
    'safe_divide',
    'ensure_list',
    'batch_process',
    'log_execution_context'
]