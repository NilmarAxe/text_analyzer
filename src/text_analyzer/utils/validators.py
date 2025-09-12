"""
Comprehensive validation system for Text Analyzer.
"""

import os
import re
from pathlib import Path
from typing import Any, List, Dict, Optional, Union, Tuple
from .exceptions import (
    ParameterValidationError,
    DataValidationError,
    FileNotFoundError,
    FileSizeError,
    InvalidArgumentError
)


class FileValidator:
    """Strategic file validation with comprehensive checks."""
    
    @staticmethod
    def validate_file_path(filepath: Union[str, Path]) -> Path:
        """
        Validate file path existence and accessibility.
        
        Args:
            filepath: Path to validate
            
        Returns:
            Validated Path object
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ParameterValidationError: If path is invalid
        """
        if not filepath:
            raise ParameterValidationError(
                'filepath', filepath, 'Path cannot be empty or None'
            )
        
        path_obj = Path(filepath) if isinstance(filepath, str) else filepath
        
        if not path_obj.exists():
            raise FileNotFoundError(str(path_obj))
        
        if not path_obj.is_file():
            raise ParameterValidationError(
                'filepath', str(path_obj), 'Path must point to a file, not a directory'
            )
        
        return path_obj
    
    @staticmethod
    def validate_file_size(filepath: Path, max_size: int, min_size: int = 1) -> int:
        """
        Validate file size constraints.
        
        Args:
            filepath: File to check
            max_size: Maximum allowed size in bytes
            min_size: Minimum required size in bytes
            
        Returns:
            File size in bytes
            
        Raises:
            FileSizeError: If file size is outside limits
        """
        try:
            file_size = filepath.stat().st_size
        except OSError as e:
            raise ParameterValidationError(
                'filepath', str(filepath), f'Cannot access file: {e}'
            )
        
        if file_size < min_size:
            raise FileSizeError(str(filepath), file_size, min_size)
        
        if file_size > max_size:
            raise FileSizeError(str(filepath), file_size, max_size)
        
        return file_size
    
    @staticmethod
    def validate_file_extension(filepath: Path, allowed_extensions: List[str]) -> bool:
        """
        Validate file extension against allowed types.
        
        Args:
            filepath: File to check
            allowed_extensions: List of allowed extensions (e.g., ['.txt', '.md'])
            
        Returns:
            True if extension is valid
            
        Raises:
            ParameterValidationError: If extension is not allowed
        """
        extension = filepath.suffix.lower()
        
        if extension not in [ext.lower() for ext in allowed_extensions]:
            raise ParameterValidationError(
                'file_extension', extension, 
                f'Extension must be one of: {allowed_extensions}'
            )
        
        return True
    
    @staticmethod
    def validate_file_permissions(filepath: Path, read: bool = True, write: bool = False) -> bool:
        """
        Validate file access permissions.
        
        Args:
            filepath: File to check
            read: Whether read access is required
            write: Whether write access is required
            
        Returns:
            True if permissions are valid
            
        Raises:
            ParameterValidationError: If required permissions are missing
        """
        if read and not os.access(filepath, os.R_OK):
            raise ParameterValidationError(
                'file_permissions', str(filepath), 'Read permission denied'
            )
        
        if write and not os.access(filepath, os.W_OK):
            raise ParameterValidationError(
                'file_permissions', str(filepath), 'Write permission denied'
            )
        
        return True


class TextValidator:
    """Comprehensive text content validation."""
    
    @staticmethod
    def validate_text_content(text: str, min_length: int = 1, max_length: Optional[int] = None) -> bool:
        """
        Validate basic text content requirements.
        
        Args:
            text: Text to validate
            min_length: Minimum required length
            max_length: Maximum allowed length
            
        Returns:
            True if text is valid
            
        Raises:
            DataValidationError: If text fails validation
        """
        if not isinstance(text, str):
            raise DataValidationError(
                'text_content', f'Must be string, got {type(text).__name__}'
            )
        
        if len(text) < min_length:
            raise DataValidationError(
                'text_content', f'Minimum length {min_length} characters, got {len(text)}'
            )
        
        if max_length and len(text) > max_length:
            raise DataValidationError(
                'text_content', f'Maximum length {max_length} characters, got {len(text)}'
            )
        
        return True
    
    @staticmethod
    def validate_encoding(text: bytes, encodings: List[str]) -> Tuple[str, str]:
        """
        Validate and detect text encoding.
        
        Args:
            text: Byte content to decode
            encodings: List of encodings to try
            
        Returns:
            Tuple of (decoded_text, successful_encoding)
            
        Raises:
            DataValidationError: If no encoding works
        """
        for encoding in encodings:
            try:
                decoded = text.decode(encoding)
                return decoded, encoding
            except UnicodeDecodeError:
                continue
        
        raise DataValidationError(
            'text_encoding', f'Cannot decode with any of: {encodings}'
        )
    
    @staticmethod
    def validate_word_pattern(pattern: str) -> bool:
        """
        Validate regex pattern for word extraction.
        
        Args:
            pattern: Regex pattern to validate
            
        Returns:
            True if pattern is valid
            
        Raises:
            ParameterValidationError: If pattern is invalid
        """
        try:
            re.compile(pattern)
            return True
        except re.error as e:
            raise ParameterValidationError(
                'word_pattern', pattern, f'Invalid regex pattern: {e}'
            )


class ParameterValidator:
    """Strategic parameter validation for all system components."""
    
    @staticmethod
    def validate_positive_integer(value: Any, parameter_name: str, minimum: int = 1) -> int:
        """
        Validate positive integer parameters.
        
        Args:
            value: Value to validate
            parameter_name: Name of parameter for error reporting
            minimum: Minimum allowed value
            
        Returns:
            Validated integer value
            
        Raises:
            ParameterValidationError: If value is invalid
        """
        if not isinstance(value, int):
            try:
                value = int(value)
            except (ValueError, TypeError):
                raise ParameterValidationError(
                    parameter_name, value, 'Must be a valid integer'
                )
        
        if value < minimum:
            raise ParameterValidationError(
                parameter_name, value, f'Must be >= {minimum}'
            )
        
        return value
    
    @staticmethod
    def validate_string_parameter(value: Any, parameter_name: str, 
                                 min_length: int = 1, max_length: Optional[int] = None,
                                 allowed_values: Optional[List[str]] = None) -> str:
        """
        Validate string parameters with constraints.
        
        Args:
            value: Value to validate
            parameter_name: Parameter name for error reporting
            min_length: Minimum string length
            max_length: Maximum string length
            allowed_values: List of allowed values
            
        Returns:
            Validated string value
            
        Raises:
            ParameterValidationError: If value is invalid
        """
        if not isinstance(value, str):
            raise ParameterValidationError(
                parameter_name, value, 'Must be a string'
            )
        
        if len(value) < min_length:
            raise ParameterValidationError(
                parameter_name, value, f'Minimum length {min_length} characters'
            )
        
        if max_length and len(value) > max_length:
            raise ParameterValidationError(
                parameter_name, value, f'Maximum length {max_length} characters'
            )
        
        if allowed_values and value not in allowed_values:
            raise ParameterValidationError(
                parameter_name, value, f'Must be one of: {allowed_values}'
            )
        
        return value
    
    @staticmethod
    def validate_boolean_parameter(value: Any, parameter_name: str) -> bool:
        """
        Validate boolean parameters with type coercion.
        
        Args:
            value: Value to validate
            parameter_name: Parameter name for error reporting
            
        Returns:
            Validated boolean value
            
        Raises:
            ParameterValidationError: If value cannot be converted to boolean
        """
        if isinstance(value, bool):
            return value
        
        if isinstance(value, str):
            lower_value = value.lower()
            if lower_value in ('true', '1', 'yes', 'on'):
                return True
            elif lower_value in ('false', '0', 'no', 'off'):
                return False
        
        if isinstance(value, (int, float)):
            return bool(value)
        
        raise ParameterValidationError(
            parameter_name, value, 'Must be a valid boolean value'
        )
    
    @staticmethod
    def validate_list_parameter(value: Any, parameter_name: str,
                               element_type: type, min_length: int = 0,
                               max_length: Optional[int] = None) -> List[Any]:
        """
        Validate list parameters with element type checking.
        
        Args:
            value: Value to validate
            parameter_name: Parameter name for error reporting
            element_type: Required type for list elements
            min_length: Minimum list length
            max_length: Maximum list length
            
        Returns:
            Validated list
            
        Raises:
            ParameterValidationError: If value is invalid
        """
        if not isinstance(value, list):
            raise ParameterValidationError(
                parameter_name, value, 'Must be a list'
            )
        
        if len(value) < min_length:
            raise ParameterValidationError(
                parameter_name, value, f'Minimum {min_length} elements required'
            )
        
        if max_length and len(value) > max_length:
            raise ParameterValidationError(
                parameter_name, value, f'Maximum {max_length} elements allowed'
            )
        
        for i, element in enumerate(value):
            if not isinstance(element, element_type):
                raise ParameterValidationError(
                    f'{parameter_name}[{i}]', element, 
                    f'Must be of type {element_type.__name__}'
                )
        
        return value


class ConfigurationValidator:
    """Systematic configuration validation."""
    
    @staticmethod
    def validate_analysis_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate analysis configuration parameters.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Validated configuration
            
        Raises:
            ParameterValidationError: If configuration is invalid
        """
        validated = {}
        
        # Validate word length constraints
        if 'min_word_length' in config:
            validated['min_word_length'] = ParameterValidator.validate_positive_integer(
                config['min_word_length'], 'min_word_length', 1
            )
        
        if 'max_word_length' in config:
            validated['max_word_length'] = ParameterValidator.validate_positive_integer(
                config['max_word_length'], 'max_word_length', 1
            )
        
        # Ensure min <= max
        if ('min_word_length' in validated and 'max_word_length' in validated and
            validated['min_word_length'] > validated['max_word_length']):
            raise ParameterValidationError(
                'word_length_constraints', 
                f"min={validated['min_word_length']}, max={validated['max_word_length']}", 
                'Minimum word length cannot exceed maximum'
            )
        
        # Validate top words count
        if 'top_words_count' in config:
            validated['top_words_count'] = ParameterValidator.validate_positive_integer(
                config['top_words_count'], 'top_words_count', 1
            )
        
        # Validate case sensitivity
        if 'case_sensitive' in config:
            validated['case_sensitive'] = ParameterValidator.validate_boolean_parameter(
                config['case_sensitive'], 'case_sensitive'
            )
        
        # Validate word pattern
        if 'word_pattern' in config:
            pattern = ParameterValidator.validate_string_parameter(
                config['word_pattern'], 'word_pattern', 1
            )
            TextValidator.validate_word_pattern(pattern)
            validated['word_pattern'] = pattern
        
        return validated
    
    @staticmethod
    def validate_output_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate output configuration parameters.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Validated configuration
            
        Raises:
            ParameterValidationError: If configuration is invalid
        """
        validated = {}
        
        # Validate output directory
        if 'output_dir' in config:
            output_dir = Path(config['output_dir'])
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                validated['output_dir'] = output_dir
            except OSError as e:
                raise ParameterValidationError(
                    'output_dir', str(output_dir), f'Cannot create directory: {e}'
                )
        
        # Validate export formats
        if 'export_formats' in config:
            allowed_formats = ['txt', 'json', 'csv', 'html']
            formats = ParameterValidator.validate_list_parameter(
                config['export_formats'], 'export_formats', str
            )
            for fmt in formats:
                if fmt not in allowed_formats:
                    raise ParameterValidationError(
                        'export_formats', fmt, f'Must be one of: {allowed_formats}'
                    )
            validated['export_formats'] = formats
        
        # Validate report width
        if 'report_width' in config:
            validated['report_width'] = ParameterValidator.validate_positive_integer(
                config['report_width'], 'report_width', 40
            )
        
        return validated


class CLIValidator:
    """Command-line interface argument validation."""
    
    @staticmethod
    def validate_cli_arguments(args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate complete CLI argument set.
        
        Args:
            args: Parsed CLI arguments
            
        Returns:
            Validated arguments
            
        Raises:
            InvalidArgumentError: If arguments are invalid
        """
        validated = {}
        
        try:
            # Validate input file
            if 'input_file' in args and args['input_file']:
                file_path = FileValidator.validate_file_path(args['input_file'])
                validated['input_file'] = file_path
            
            # Validate output file
            if 'output_file' in args and args['output_file']:
                output_path = Path(args['output_file'])
                # Create parent directory if needed
                output_path.parent.mkdir(parents=True, exist_ok=True)
                validated['output_file'] = output_path
            
            # Validate top words count
            if 'top_count' in args and args['top_count'] is not None:
                validated['top_count'] = ParameterValidator.validate_positive_integer(
                    args['top_count'], 'top_count', 1
                )
            
            # Validate format
            if 'format' in args and args['format']:
                validated['format'] = ParameterValidator.validate_string_parameter(
                    args['format'], 'format', 
                    allowed_values=['txt', 'json', 'csv', 'html']
                )
            
            # Validate verbose flag
            if 'verbose' in args:
                validated['verbose'] = ParameterValidator.validate_boolean_parameter(
                    args['verbose'], 'verbose'
                )
            
            return validated
            
        except (ParameterValidationError, DataValidationError, FileNotFoundError) as e:
            raise InvalidArgumentError('cli_arguments', str(args), str(e))


# Validation utilities
def validate_all_parameters(**kwargs) -> Dict[str, Any]:
    """
    Comprehensive parameter validation utility.
    
    Args:
        **kwargs: Parameters to validate
        
    Returns:
        Dictionary of validated parameters
        
    Raises:
        ParameterValidationError: If any parameter is invalid
    """
    validated = {}
    
    for param_name, param_value in kwargs.items():
        if param_value is not None:
            validated[param_name] = param_value
    
    return validated


def create_validation_report(validation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate comprehensive validation report.
    
    Args:
        validation_results: List of validation result dictionaries
        
    Returns:
        Summary report of all validations
    """
    total_validations = len(validation_results)
    successful_validations = sum(1 for result in validation_results if result.get('success', False))
    failed_validations = total_validations - successful_validations
    
    return {
        'total_validations': total_validations,
        'successful_validations': successful_validations,
        'failed_validations': failed_validations,
        'success_rate': successful_validations / total_validations if total_validations > 0 else 0.0,
        'validation_details': validation_results
    }