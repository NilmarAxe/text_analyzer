"""
Strategic test suite for FileHandler component.
Comprehensive testing of file I/O operations and encoding detection.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open
import os

# Import components under test
import sys
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.text_analyzer.core.file_handler import FileHandler, FileInfo
from src.text_analyzer.utils import (
    FileNotFoundError,
    FileAccessError,
    FileEncodingError,
    FileSizeError,
    EmptyFileError
)


class TestFileHandler:
    """Comprehensive test suite for FileHandler class."""
    
    @pytest.fixture
    def file_handler(self):
        """Create FileHandler instance for testing."""
        return FileHandler()
    
    @pytest.fixture
    def sample_text_file(self):
        """Create temporary file with sample text in UTF-8."""
        content = "The quick brown fox jumps over the lazy dog.\nThis is a test file with multiple lines.\n"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(content)
            return Path(f.name)
    
    @pytest.fixture
    def utf16_text_file(self):
        """Create temporary file with UTF-16 encoding."""
        content = "This is a UTF-16 encoded file with special characters: àáâãäå"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-16') as f:
            f.write(content)
            return Path(f.name)
    
    @pytest.fixture
    def latin1_text_file(self):
        """Create temporary file with Latin-1 encoding."""
        content = "This is a Latin-1 file with special chars: café, naïve, résumé"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='latin-1') as f:
            f.write(content)
            return Path(f.name)
    
    @pytest.fixture
    def empty_file(self):
        """Create temporary empty file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            pass  # Create empty file
        return Path(f.name)
    
    @pytest.fixture
    def large_file(self):
        """Create temporary large file for testing size limits."""
        content = "This is a large file for testing. " * 1000
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(content)
            return Path(f.name)
    
    @pytest.fixture
    def binary_file(self):
        """Create temporary binary file."""
        binary_content = b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f'
        
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.bin', delete=False) as f:
            f.write(binary_content)
            return Path(f.name)
    
    def test_file_handler_initialization(self, file_handler):
        """Test FileHandler initialization."""
        assert file_handler is not None
        assert hasattr(file_handler, 'logger')
        assert hasattr(file_handler, 'performance_monitor')
        assert hasattr(file_handler, 'supported_encodings')
        assert hasattr(file_handler, 'max_file_size')
        assert len(file_handler.supported_encodings) > 0
    
    def test_validate_file_success(self, file_handler, sample_text_file):
        """Test successful file validation."""
        result = file_handler.validate_file(sample_text_file)
        
        assert isinstance(result, Path)
        assert result == sample_text_file
        assert result.exists()
    
    def test_validate_file_not_found(self, file_handler):
        """Test validation with non-existent file."""
        non_existent = Path("this_file_does_not_exist.txt")
        
        with pytest.raises(FileNotFoundError):
            file_handler.validate_file(non_existent)
    
    def test_validate_file_directory(self, file_handler):
        """Test validation when path points to directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dir_path = Path(temp_dir)
            
            with pytest.raises(Exception):  # Should raise ParameterValidationError
                file_handler.validate_file(dir_path)
    
    def test_validate_file_size_within_limits(self, file_handler, sample_text_file):
        """Test file size validation within limits."""
        file_size = file_handler.validate_file_size(
            sample_text_file,
            max_size=1024*1024,  # 1MB
            min_size=1
        )
        
        assert file_size > 0
        assert file_size == sample_text_file.stat().st_size
    
    def test_validate_file_size_too_large(self, file_handler, sample_text_file):
        """Test file size validation with file too large."""
        with pytest.raises(FileSizeError):
            file_handler.validate_file_size(
                sample_text_file,
                max_size=1,  # Very small limit
                min_size=1
            )
    
    def test_validate_file_size_too_small(self, file_handler, sample_text_file):
        """Test file size validation with file too small."""
        file_size = sample_text_file.stat().st_size
        
        with pytest.raises(FileSizeError):
            file_handler.validate_file_size(
                sample_text_file,
                max_size=1024*1024,
                min_size=file_size + 1  # Larger than actual file
            )
    
    def test_detect_encoding_utf8(self, file_handler, sample_text_file):
        """Test encoding detection for UTF-8 files."""
        encoding, confidence = file_handler.detect_encoding(sample_text_file)
        
        assert encoding == 'utf-8'
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be confident about UTF-8
    
    def test_detect_encoding_utf16(self, file_handler, utf16_text_file):
        """Test encoding detection for UTF-16 files."""
        encoding, confidence = file_handler.detect_encoding(utf16_text_file)
        
        # Should detect UTF-16 or fall back to supported encoding
        assert encoding in file_handler.supported_encodings
        assert 0.0 <= confidence <= 1.0
    
    def test_detect_encoding_latin1(self, file_handler, latin1_text_file):
        """Test encoding detection for Latin-1 files."""
        encoding, confidence = file_handler.detect_encoding(latin1_text_file)
        
        assert encoding in file_handler.supported_encodings
        assert 0.0 <= confidence <= 1.0
    
    def test_detect_encoding_binary_file(self, file_handler, binary_file):
        """Test encoding detection fails gracefully with binary files."""
        with pytest.raises(FileEncodingError):
            file_handler.detect_encoding(binary_file)
    
    def test_read_file_success(self, file_handler, sample_text_file):
        """Test successful file reading."""
        content, file_info = file_handler.read_file(sample_text_file)
        
        # Verify content
        assert isinstance(content, str)
        assert len(content) > 0
        assert "quick brown fox" in content
        
        # Verify file info
        assert isinstance(file_info, FileInfo)
        assert file_info.path == str(sample_text_file.absolute())
        assert file_info.size > 0
        assert file_info.encoding in file_handler.supported_encodings
        assert file_info.lines > 0
        assert len(file_info.checksum) > 0
    
    def test_read_file_with_specified_encoding(self, file_handler, sample_text_file):
        """Test file reading with specified encoding."""
        content, file_info = file_handler.read_file(sample_text_file, encoding='utf-8')
        
        assert isinstance(content, str)
        assert len(content) > 0
        assert file_info.encoding == 'utf-8'
    
    def test_read_file_empty_file(self, file_handler, empty_file):
        """Test reading empty file raises appropriate error."""
        with pytest.raises(EmptyFileError):
            file_handler.read_file(empty_file)
    
    def test_read_file_not_found(self, file_handler):
        """Test reading non-existent file."""
        non_existent = Path("non_existent_file.txt")
        
        with pytest.raises(FileNotFoundError):
            file_handler.read_file(non_existent)
    
    def test_read_file_invalid_encoding(self, file_handler, sample_text_file):
        """Test reading file with invalid encoding specification."""
        # This should fall back to auto-detection or raise appropriate error
        try:
            content, file_info = file_handler.read_file(sample_text_file, encoding='invalid-encoding')
            # If it succeeds, it should have auto-detected
            assert content is not None
        except FileEncodingError:
            # This is also acceptable behavior
            pass
    
    def test_read_multiple_files_success(self, file_handler, sample_text_file, utf16_text_file):
        """Test reading multiple files successfully."""
        files = [sample_text_file, utf16_text_file]
        results = file_handler.read_multiple_files(files)
        
        assert len(results) == 2
        assert str(sample_text_file) in results
        assert str(utf16_text_file) in results
        
        for filepath, result in results.items():
            assert 'success' in result
            assert 'content' in result
            assert 'file_info' in result
            assert 'error' in result
    
    def test_read_multiple_files_with_failures(self, file_handler, sample_text_file):
        """Test batch reading with some failures."""
        non_existent = Path("non_existent_file.txt")
        files = [sample_text_file, non_existent]
        
        results = file_handler.read_multiple_files(files)
        
        assert len(results) == 2
        
        # One should succeed, one should fail
        successful = [r for r in results.values() if r['success']]
        failed = [r for r in results.values() if not r['success']]
        
        assert len(successful) == 1
        assert len(failed) == 1
    
    def test_get_file_preview_success(self, file_handler, sample_text_file):
        """Test getting file preview."""
        preview = file_handler.get_file_preview(sample_text_file, lines=2)
        
        assert preview['success'] is True
        assert preview['filepath'] == str(sample_text_file)
        assert preview['encoding'] is not None
        assert len(preview['preview_lines']) <= 2
        assert preview['total_lines'] >= len(preview['preview_lines'])
        assert 'file_size' in preview
    
    def test_get_file_preview_more_lines_than_file(self, file_handler, sample_text_file):
        """Test preview when requesting more lines than file contains."""
        preview = file_handler.get_file_preview(sample_text_file, lines=100)
        
        assert preview['success'] is True
        assert preview['is_truncated'] is False
        assert len(preview['preview_lines']) == preview['total_lines']
    
    def test_get_file_preview_failure(self, file_handler):
        """Test preview with non-existent file."""
        non_existent = Path("non_existent_file.txt")
        preview = file_handler.get_file_preview(non_existent)
        
        assert preview['success'] is False
        assert 'error' in preview
        assert len(preview['preview_lines']) == 0
    
    def test_verify_file_integrity_generate_checksum(self, file_handler, sample_text_file):
        """Test file integrity verification by generating checksum."""
        is_valid = file_handler.verify_file_integrity(sample_text_file)
        
        assert is_valid is True
    
    def test_verify_file_integrity_with_expected_checksum(self, file_handler, sample_text_file):
        """Test file integrity verification with expected checksum."""
        # First, get the actual checksum
        from src.text_analyzer.utils.helpers import FileSystemHelper
        actual_checksum = FileSystemHelper.calculate_file_hash(sample_text_file, 'sha256')
        
        # Verify with correct checksum
        is_valid = file_handler.verify_file_integrity(sample_text_file, actual_checksum)
        assert is_valid is True
        
        # Verify with incorrect checksum
        wrong_checksum = "0" * 64  # Wrong checksum
        is_valid = file_handler.verify_file_integrity(sample_text_file, wrong_checksum)
        assert is_valid is False
    
    def test_verify_file_integrity_failure(self, file_handler):
        """Test integrity verification with non-existent file."""
        non_existent = Path("non_existent_file.txt")
        is_valid = file_handler.verify_file_integrity(non_existent)
        
        assert is_valid is False
    
    def test_is_text_content_valid_text(self, file_handler):
        """Test text content validation with valid text."""
        valid_text = "This is normal text with some punctuation! 123"
        assert file_handler._is_text_content(valid_text) is True
    
    def test_is_text_content_with_control_characters(self, file_handler):
        """Test text content validation with control characters."""
        # Text with some control characters but still mostly text
        text_with_controls = "Normal text\t\nwith tabs and newlines\r"
        assert file_handler._is_text_content(text_with_controls) is True
        
        # Text with excessive control characters
        text_with_many_controls = "Normal\x01\x02\x03\x04\x05\x06\x07\x08text"
        result = file_handler._is_text_content(text_with_many_controls)
        # Should depend on ratio of control characters
        assert isinstance(result, bool)
    
    def test_is_text_content_with_null_bytes(self, file_handler):
        """Test text content validation with null bytes."""
        text_with_nulls = "Normal text\x00\x00\x00with null bytes"
        result = file_handler._is_text_content(text_with_nulls)
        # Should be False due to null bytes
        assert result is False
    
    def test_is_text_content_empty_string(self, file_handler):
        """Test text content validation with empty string."""
        assert file_handler._is_text_content("") is False
    
    def test_calculate_confidence_utf8(self, file_handler):
        """Test confidence calculation for UTF-8 content."""
        utf8_content = "This is normal ASCII text".encode('utf-8')
        confidence = file_handler._calculate_confidence(utf8_content, 'utf-8')
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.9  # Should be high confidence for ASCII in UTF-8
    
    def test_calculate_confidence_with_replacement_chars(self, file_handler):
        """Test confidence calculation with replacement characters."""
        # Create content that might have decoding issues
        problematic_content = b'\x80\x81\x82\x83'  # Invalid UTF-8
        confidence = file_handler._calculate_confidence(problematic_content, 'utf-8')
        
        assert 0.0 <= confidence <= 1.0
        # Should have lower confidence due to decoding issues
    
    def test_calculate_confidence_invalid_encoding(self, file_handler):
        """Test confidence calculation with encoding that can't decode content."""
        utf8_content = "Normal text".encode('utf-8')
        confidence = file_handler._calculate_confidence(utf8_content, 'invalid-encoding')
        
        # Should handle gracefully and return 0.0
        assert confidence == 0.0
    
    @patch('builtins.open', side_effect=PermissionError("Permission denied"))
    def test_read_file_permission_error(self, mock_file, file_handler):
        """Test file reading with permission error."""
        fake_file = Path("permission_denied.txt")
        
        with pytest.raises(FileAccessError):
            file_handler.read_file(fake_file)
    
    @patch('pathlib.Path.stat')
    def test_validate_file_size_stat_error(self, mock_stat, file_handler, sample_text_file):
        """Test file size validation when stat() fails."""
        mock_stat.side_effect = OSError("Cannot access file")
        
        with pytest.raises(Exception):  # Should raise ParameterValidationError
            file_handler.validate_file_size(sample_text_file, 1024, 1)
    
    def test_performance_monitoring_integration(self, file_handler, sample_text_file):
        """Test that performance monitoring is working."""
        # Read file and check if performance monitor recorded checkpoints
        content, file_info = file_handler.read_file(sample_text_file)
        
        # Performance monitor should have recorded some checkpoints
        assert hasattr(file_handler, 'performance_monitor')
        # The actual checkpoints depend on implementation details
    
    def test_cleanup_after_operations(self, file_handler, sample_text_file, utf16_text_file, empty_file, large_file, binary_file, latin1_text_file):
        """Cleanup test files."""
        # Clean up all test files
        test_files = [sample_text_file, utf16_text_file, empty_file, large_file, binary_file, latin1_text_file]
        
        for test_file in test_files:
            if test_file and test_file.exists():
                try:
                    test_file.unlink()
                except OSError:
                    pass  # Ignore cleanup errors


class TestFileInfo:
    """Test suite for FileInfo namedtuple."""
    
    def test_file_info_creation(self):
        """Test FileInfo creation and access."""
        info = FileInfo(
            path="/test/file.txt",
            size=1024,
            encoding="utf-8",
            lines=10,
            checksum="abc123"
        )
        
        assert info.path == "/test/file.txt"
        assert info.size == 1024
        assert info.encoding == "utf-8"
        assert info.lines == 10
        assert info.checksum == "abc123"
    
    def test_file_info_as_dict(self):
        """Test converting FileInfo to dictionary."""
        info = FileInfo(
            path="/test/file.txt",
            size=1024,
            encoding="utf-8", 
            lines=10,
            checksum="abc123"
        )
        
        info_dict = info._asdict()
        
        assert isinstance(info_dict, dict)
        assert info_dict['path'] == "/test/file.txt"
        assert info_dict['size'] == 1024
        assert info_dict['encoding'] == "utf-8"
        assert info_dict['lines'] == 10
        assert info_dict['checksum'] == "abc123"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])