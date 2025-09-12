"""
Comprehensive exception hierarchy for the Text Analyzer system.
"""

from typing import Optional, Any


class TextAnalyzerError(Exception):
    """
    Base exception class for all Text Analyzer errors.
    Provides structured error handling with context information.
    """
    
    def __init__(self, message: str, error_code: Optional[str] = None, context: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
    
    def __str__(self) -> str:
        error_str = self.message
        if self.error_code:
            error_str = f"[{self.error_code}] {error_str}"
        return error_str
    
    def to_dict(self) -> dict:
        """Convert exception to dictionary for structured logging."""
        return {
            'type': self.__class__.__name__,
            'message': self.message,
            'error_code': self.error_code,
            'context': self.context
        }


# File handling exceptions
class FileHandlerError(TextAnalyzerError):
    """Base class for file handling errors."""
    pass


class FileNotFoundError(FileHandlerError):
    """Raised when a specified file cannot be found."""
    
    def __init__(self, filepath: str, context: Optional[dict] = None):
        message = f"File not found: {filepath}"
        super().__init__(message, "FILE_NOT_FOUND", context or {'filepath': filepath})


class FileAccessError(FileHandlerError):
    """Raised when file cannot be accessed due to permissions or locks."""
    
    def __init__(self, filepath: str, reason: str, context: Optional[dict] = None):
        message = f"Cannot access file '{filepath}': {reason}"
        super().__init__(message, "FILE_ACCESS_DENIED", context or {'filepath': filepath, 'reason': reason})


class FileEncodingError(FileHandlerError):
    """Raised when file encoding cannot be determined or decoded."""
    
    def __init__(self, filepath: str, attempted_encodings: list, context: Optional[dict] = None):
        message = f"Unable to decode file '{filepath}' with encodings: {attempted_encodings}"
        super().__init__(message, "FILE_ENCODING_ERROR", 
                        context or {'filepath': filepath, 'attempted_encodings': attempted_encodings})


class FileSizeError(FileHandlerError):
    """Raised when file size exceeds system limitations."""
    
    def __init__(self, filepath: str, file_size: int, max_size: int, context: Optional[dict] = None):
        message = f"File '{filepath}' ({file_size:,} bytes) exceeds maximum size limit ({max_size:,} bytes)"
        super().__init__(message, "FILE_SIZE_EXCEEDED", 
                        context or {'filepath': filepath, 'file_size': file_size, 'max_size': max_size})


class EmptyFileError(FileHandlerError):
    """Raised when file is empty or contains no analyzable content."""
    
    def __init__(self, filepath: str, context: Optional[dict] = None):
        message = f"File '{filepath}' is empty or contains no text content"
        super().__init__(message, "FILE_EMPTY", context or {'filepath': filepath})


# Text processing exceptions
class TextProcessorError(TextAnalyzerError):
    """Base class for text processing errors."""
    pass


class InvalidTextError(TextProcessorError):
    """Raised when text content is invalid or cannot be processed."""
    
    def __init__(self, reason: str, sample_text: Optional[str] = None, context: Optional[dict] = None):
        message = f"Invalid text content: {reason}"
        ctx = context or {'reason': reason}
        if sample_text:
            ctx['sample_text'] = sample_text[:100] + "..." if len(sample_text) > 100 else sample_text
        super().__init__(message, "INVALID_TEXT", ctx)


class TextProcessingError(TextProcessorError):
    """Raised when text processing operations fail."""
    
    def __init__(self, operation: str, reason: str, context: Optional[dict] = None):
        message = f"Text processing failed during {operation}: {reason}"
        super().__init__(message, "PROCESSING_FAILED", 
                        context or {'operation': operation, 'reason': reason})


class WordExtractionError(TextProcessorError):
    """Raised when word extraction fails."""
    
    def __init__(self, pattern: str, text_length: int, context: Optional[dict] = None):
        message = f"Failed to extract words using pattern '{pattern}' from text ({text_length} characters)"
        super().__init__(message, "WORD_EXTRACTION_FAILED",
                        context or {'pattern': pattern, 'text_length': text_length})


# Analysis exceptions
class AnalysisError(TextAnalyzerError):
    """Base class for analysis operation errors."""
    pass


class InsufficientDataError(AnalysisError):
    """Raised when there is insufficient data for meaningful analysis."""
    
    def __init__(self, data_type: str, required_minimum: int, actual_count: int, context: Optional[dict] = None):
        message = f"Insufficient {data_type} for analysis: {actual_count} found, minimum {required_minimum} required"
        super().__init__(message, "INSUFFICIENT_DATA",
                        context or {'data_type': data_type, 'required': required_minimum, 'actual': actual_count})


class AnalysisConfigurationError(AnalysisError):
    """Raised when analysis configuration is invalid."""
    
    def __init__(self, parameter: str, value: Any, reason: str, context: Optional[dict] = None):
        message = f"Invalid configuration parameter '{parameter}' = {value}: {reason}"
        super().__init__(message, "INVALID_CONFIGURATION",
                        context or {'parameter': parameter, 'value': value, 'reason': reason})


# Report generation exceptions
class ReportGeneratorError(TextAnalyzerError):
    """Base class for report generation errors."""
    pass


class ReportFormattingError(ReportGeneratorError):
    """Raised when report formatting fails."""
    
    def __init__(self, format_type: str, reason: str, context: Optional[dict] = None):
        message = f"Report formatting failed for format '{format_type}': {reason}"
        super().__init__(message, "REPORT_FORMATTING_FAILED",
                        context or {'format_type': format_type, 'reason': reason})


class ReportOutputError(ReportGeneratorError):
    """Raised when report output operations fail."""
    
    def __init__(self, output_path: str, reason: str, context: Optional[dict] = None):
        message = f"Failed to save report to '{output_path}': {reason}"
        super().__init__(message, "REPORT_OUTPUT_FAILED",
                        context or {'output_path': output_path, 'reason': reason})


# CLI exceptions
class CLIError(TextAnalyzerError):
    """Base class for command-line interface errors."""
    pass


class InvalidArgumentError(CLIError):
    """Raised when command-line arguments are invalid."""
    
    def __init__(self, argument: str, value: Optional[str], reason: str, context: Optional[dict] = None):
        if value:
            message = f"Invalid argument '{argument}' = '{value}': {reason}"
        else:
            message = f"Missing required argument '{argument}': {reason}"
        super().__init__(message, "INVALID_ARGUMENT",
                        context or {'argument': argument, 'value': value, 'reason': reason})


class CommandExecutionError(CLIError):
    """Raised when CLI command execution fails."""
    
    def __init__(self, command: str, reason: str, context: Optional[dict] = None):
        message = f"Command execution failed '{command}': {reason}"
        super().__init__(message, "COMMAND_EXECUTION_FAILED",
                        context or {'command': command, 'reason': reason})


# System-level exceptions
class SystemError(TextAnalyzerError):
    """Base class for system-level errors."""
    pass


class MemoryError(SystemError):
    """Raised when system memory limitations are exceeded."""
    
    def __init__(self, operation: str, memory_used: int, memory_limit: int, context: Optional[dict] = None):
        message = f"Memory limit exceeded during {operation}: {memory_used:,} bytes used, limit {memory_limit:,} bytes"
        super().__init__(message, "MEMORY_LIMIT_EXCEEDED",
                        context or {'operation': operation, 'memory_used': memory_used, 'memory_limit': memory_limit})


class ConfigurationError(SystemError):
    """Raised when system configuration is invalid or incomplete."""
    
    def __init__(self, component: str, issue: str, context: Optional[dict] = None):
        message = f"Configuration error in {component}: {issue}"
        super().__init__(message, "CONFIGURATION_ERROR",
                        context or {'component': component, 'issue': issue})


# Validation exceptions
class ValidationError(TextAnalyzerError):
    """Base class for validation errors."""
    pass


class ParameterValidationError(ValidationError):
    """Raised when parameter validation fails."""
    
    def __init__(self, parameter: str, value: Any, constraint: str, context: Optional[dict] = None):
        message = f"Parameter '{parameter}' value {value} violates constraint: {constraint}"
        super().__init__(message, "PARAMETER_VALIDATION_FAILED",
                        context or {'parameter': parameter, 'value': value, 'constraint': constraint})


class DataValidationError(ValidationError):
    """Raised when data validation fails."""
    
    def __init__(self, data_type: str, validation_rule: str, context: Optional[dict] = None):
        message = f"{data_type} validation failed: {validation_rule}"
        super().__init__(message, "DATA_VALIDATION_FAILED",
                        context or {'data_type': data_type, 'validation_rule': validation_rule})


# Exception handler utilities
def handle_exception(exception: Exception, logger, operation: str = "operation") -> dict:
    """
    Standardized exception handling with logging.
    
    Args:
        exception: The caught exception
        logger: Logger instance for recording the error
        operation: Description of the operation that failed
        
    Returns:
        Dictionary with error information
    """
    error_info = {
        'success': False,
        'error_type': exception.__class__.__name__,
        'error_message': str(exception),
        'operation': operation
    }
    
    if isinstance(exception, TextAnalyzerError):
        error_info.update(exception.to_dict())
        logger.error(f"TextAnalyzer error during {operation}: {exception}")
    else:
        logger.error(f"Unexpected error during {operation}: {exception}", exc_info=True)
    
    return error_info


def raise_with_context(exception_class, message: str, **context):
    """
    Utility function to raise exceptions with structured context.
    
    Args:
        exception_class: Exception class to raise
        message: Error message
        **context: Additional context information
    """
    raise exception_class(message, context=context)