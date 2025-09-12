"""
Strategic file handling system for Text Analyzer.
Robust file I/O operations with comprehensive error handling and encoding detection.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple
from collections import namedtuple

from config import get_file_handler_logger, AnalysisConfig, SystemConfig, LoggedOperation
from ..utils import (
    FileValidator, 
    FileNotFoundError, 
    FileAccessError, 
    FileEncodingError, 
    FileSizeError, 
    EmptyFileError,
    FileSystemHelper,
    PerformanceMonitor,
    timing_decorator
)

# File information structure
FileInfo = namedtuple('FileInfo', ['path', 'size', 'encoding', 'lines', 'checksum'])

class FileHandler:
    """
    Strategic file handling with systematic approach to I/O operations.
    Implements INTJ principles: thorough planning, error anticipation, and optimization.
    """
    
    def __init__(self):
        self.logger = get_file_handler_logger()
        self.performance_monitor = PerformanceMonitor()
        self.supported_encodings = AnalysisConfig.SUPPORTED_ENCODINGS
        self.max_file_size = AnalysisConfig.MAX_FILE_SIZE
        self.min_file_size = AnalysisConfig.MIN_FILE_SIZE
        self.supported_extensions = AnalysisConfig.SUPPORTED_EXTENSIONS
    
    @timing_decorator("file_validation")
    def validate_file(self, filepath: Union[str, Path]) -> Path:
        """
        Comprehensive file validation before processing.
        
        Args:
            filepath: Path to file for validation
            
        Returns:
            Validated Path object
            
        Raises:
            FileNotFoundError: If file doesn't exist
            FileAccessError: If file cannot be accessed
            FileSizeError: If file size is outside limits
        """
        with LoggedOperation("File Validation", self.logger):
            # Convert to Path object and validate existence
            validated_path = FileValidator.validate_file_path(filepath)
            
            # Validate file size constraints
            FileValidator.validate_file_size(validated_path, self.max_file_size, self.min_file_size)
            
            # Validate file extension if restrictions apply
            if self.supported_extensions:
                FileValidator.validate_file_extension(validated_path, self.supported_extensions)
            
            # Validate file permissions
            FileValidator.validate_file_permissions(validated_path, read=True)
            
            self.logger.info(f"File validation successful: {validated_path}")
            return validated_path
    
    @timing_decorator("encoding_detection")
    def detect_encoding(self, filepath: Path) -> Tuple[str, float]:
        """
        Strategic encoding detection with systematic approach.
        
        Args:
            filepath: File to analyze
            
        Returns:
            Tuple of (encoding, confidence)
            
        Raises:
            FileEncodingError: If encoding cannot be determined
        """
        with LoggedOperation("Encoding Detection", self.logger):
            # Read sample for encoding detection
            sample_size = min(8192, filepath.stat().st_size)
            
            try:
                with open(filepath, 'rb') as f:
                    sample = f.read(sample_size)
            except IOError as e:
                raise FileAccessError(str(filepath), f"Cannot read file sample: {e}")
            
            # Try encodings in order of preference
            for encoding in self.supported_encodings:
                try:
                    decoded = sample.decode(encoding)
                    # Basic validation - check for null bytes or excessive control characters
                    if self._is_text_content(decoded):
                        confidence = self._calculate_confidence(sample, encoding)
                        self.logger.debug(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")
                        return encoding, confidence
                except UnicodeDecodeError:
                    continue
            
            # If all encodings fail, raise error
            raise FileEncodingError(str(filepath), self.supported_encodings)
    
    def _is_text_content(self, content: str) -> bool:
        """
        Validate that content appears to be text.
        
        Args:
            content: Content to validate
            
        Returns:
            True if content appears to be text
        """
        # Check for excessive null bytes or control characters
        null_bytes = content.count('\0')
        control_chars = sum(1 for c in content if ord(c) < 32 and c not in '\t\n\r')
        
        total_chars = len(content)
        if total_chars == 0:
            return False
        
        # Allow up to 1% null bytes and 5% control characters
        null_ratio = null_bytes / total_chars
        control_ratio = control_chars / total_chars
        
        return null_ratio <= 0.01 and control_ratio <= 0.05
    
    def _calculate_confidence(self, content: bytes, encoding: str) -> float:
        """
        Calculate confidence score for encoding detection.
        
        Args:
            content: Raw content bytes
            encoding: Encoding to test
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        try:
            decoded = content.decode(encoding)
            
            # Base confidence starts high
            confidence = 0.9
            
            # Reduce confidence for presence of replacement characters
            replacement_chars = decoded.count('\ufffd')
            if replacement_chars > 0:
                confidence -= (replacement_chars / len(decoded)) * 0.5
            
            # Boost confidence for UTF-8
            if encoding.lower() == 'utf-8':
                confidence += 0.05
            
            # Boost confidence for common ASCII characters
            ascii_chars = sum(1 for c in decoded if ord(c) < 128)
            ascii_ratio = ascii_chars / len(decoded) if decoded else 0
            confidence += ascii_ratio * 0.05
            
            return min(1.0, max(0.0, confidence))
            
        except UnicodeDecodeError:
            return 0.0
    
    @timing_decorator("file_reading")
    def read_file(self, filepath: Union[str, Path], 
                  encoding: Optional[str] = None) -> Tuple[str, FileInfo]:
        """
        Strategic file reading with automatic encoding detection and validation.
        
        Args:
            filepath: Path to file
            encoding: Specific encoding to use (auto-detect if None)
            
        Returns:
            Tuple of (file_content, file_info)
            
        Raises:
            FileNotFoundError: If file doesn't exist
            FileAccessError: If file cannot be read
            EmptyFileError: If file is empty
            FileEncodingError: If encoding fails
        """
        self.performance_monitor.start_monitoring()
        
        with LoggedOperation("File Reading", self.logger):
            # Validate file first
            validated_path = self.validate_file(filepath)
            self.performance_monitor.checkpoint("validation_complete")
            
            # Detect encoding if not specified
            if encoding is None:
                detected_encoding, confidence = self.detect_encoding(validated_path)
                encoding = detected_encoding
                self.logger.info(f"Auto-detected encoding: {encoding} (confidence: {confidence:.2f})")
            else:
                # Validate specified encoding
                if encoding not in self.supported_encodings:
                    self.logger.warning(f"Encoding {encoding} not in supported list")
            
            self.performance_monitor.checkpoint("encoding_detected")
            
            # Read file content
            try:
                with open(validated_path, 'r', encoding=encoding, errors='replace') as f:
                    content = f.read()
                
                self.performance_monitor.checkpoint("content_read")
                
                # Validate content is not empty
                if not content.strip():
                    raise EmptyFileError(str(validated_path))
                
                # Generate file information
                file_info = self._generate_file_info(validated_path, encoding, content)
                self.performance_monitor.checkpoint("info_generated")
                
                # Log performance metrics
                perf_report = self.performance_monitor.get_performance_report()
                self.logger.info(f"File read successfully: {len(content)} characters in {perf_report['total_execution_time']:.3f}s")
                
                return content, file_info
                
            except UnicodeDecodeError as e:
                raise FileEncodingError(str(validated_path), [encoding])
            except IOError as e:
                raise FileAccessError(str(validated_path), f"Read operation failed: {e}")
    
    def _generate_file_info(self, filepath: Path, encoding: str, content: str) -> FileInfo:
        """
        Generate comprehensive file information.
        
        Args:
            filepath: File path
            encoding: Used encoding
            content: File content
            
        Returns:
            FileInfo namedtuple with metadata
        """
        # Calculate file statistics
        file_size = filepath.stat().st_size
        line_count = content.count('\n') + 1 if content else 0
        
        # Generate checksum for integrity verification
        checksum = FileSystemHelper.calculate_file_hash(filepath, 'sha256')
        
        return FileInfo(
            path=str(filepath.absolute()),
            size=file_size,
            encoding=encoding,
            lines=line_count,
            checksum=checksum[:16]  # Shortened for display
        )
    
    @timing_decorator("batch_file_reading")
    def read_multiple_files(self, filepaths: List[Union[str, Path]]) -> Dict[str, Dict[str, Any]]:
        """
        Efficiently read multiple files with batch processing.
        
        Args:
            filepaths: List of file paths to read
            
        Returns:
            Dictionary mapping filepath to {content, file_info, success, error}
        """
        results = {}
        
        with LoggedOperation(f"Batch File Reading ({len(filepaths)} files)", self.logger):
            for filepath in filepaths:
                try:
                    content, file_info = self.read_file(filepath)
                    results[str(filepath)] = {
                        'content': content,
                        'file_info': file_info,
                        'success': True,
                        'error': None
                    }
                    self.logger.info(f"Successfully read: {filepath}")
                    
                except Exception as e:
                    results[str(filepath)] = {
                        'content': None,
                        'file_info': None,
                        'success': False,
                        'error': str(e)
                    }
                    self.logger.error(f"Failed to read {filepath}: {e}")
        
        success_count = sum(1 for result in results.values() if result['success'])
        self.logger.info(f"Batch read complete: {success_count}/{len(filepaths)} files successful")
        
        return results
    
    def get_file_preview(self, filepath: Union[str, Path], 
                        lines: int = 10, encoding: Optional[str] = None) -> Dict[str, Any]:
        """
        Get preview of file content for inspection.
        
        Args:
            filepath: Path to file
            lines: Number of lines to preview
            encoding: Specific encoding to use
            
        Returns:
            Dictionary with preview information
        """
        try:
            validated_path = self.validate_file(filepath)
            
            # Detect encoding if needed
            if encoding is None:
                encoding, _ = self.detect_encoding(validated_path)
            
            preview_lines = []
            total_lines = 0
            
            with open(validated_path, 'r', encoding=encoding, errors='replace') as f:
                for i, line in enumerate(f):
                    total_lines += 1
                    if i < lines:
                        preview_lines.append(line.rstrip('\n\r'))
                    elif i >= lines:
                        # Continue counting lines but don't store content
                        pass
            
            return {
                'success': True,
                'filepath': str(validated_path),
                'encoding': encoding,
                'preview_lines': preview_lines,
                'total_lines': total_lines,
                'is_truncated': total_lines > lines,
                'file_size': validated_path.stat().st_size
            }
            
        except Exception as e:
            return {
                'success': False,
                'filepath': str(filepath),
                'error': str(e),
                'preview_lines': [],
                'total_lines': 0
            }
    
    def verify_file_integrity(self, filepath: Path, expected_checksum: Optional[str] = None) -> bool:
        """
        Verify file integrity using checksum.
        
        Args:
            filepath: File to verify
            expected_checksum: Expected checksum (generate new if None)
            
        Returns:
            True if file is valid
        """
        try:
            current_checksum = FileSystemHelper.calculate_file_hash(filepath, 'sha256')
            
            if expected_checksum is None:
                self.logger.info(f"Generated checksum for {filepath}: {current_checksum[:16]}...")
                return True
            
            is_valid = current_checksum == expected_checksum
            if is_valid:
                self.logger.info(f"File integrity verified: {filepath}")
            else:
                self.logger.warning(f"File integrity check failed: {filepath}")
            
            return is_valid
            
        except Exception as e:
            self.logger.error(f"Integrity verification failed for {filepath}: {e}")
            return False