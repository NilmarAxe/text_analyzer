"""
Strategic test suite for TextProcessor component.
Comprehensive testing of text processing, cleaning, and analysis operations.
"""

import pytest
from collections import Counter
from unittest.mock import patch

# Import components under test
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.text_analyzer.core.text_processor import TextProcessor, WordAnalysis, ProcessingStats
from src.text_analyzer.utils import (
    InvalidTextError,
    TextProcessingError,
    WordExtractionError
)


class TestTextProcessor:
    """Comprehensive test suite for TextProcessor class."""
    
    @pytest.fixture
    def default_processor(self):
        """Create TextProcessor with default configuration."""
        return TextProcessor()
    
    @pytest.fixture
    def custom_processor(self):
        """Create TextProcessor with custom configuration."""
        config = {
            'min_word_length': 3,
            'max_word_length': 15,
            'case_sensitive': True,
            'remove_stop_words': True,
            'stop_words_language': 'english'
        }
        return TextProcessor(config)
    
    @pytest.fixture
    def sample_text(self):
        """Sample text for testing."""
        return """The quick brown fox jumps over the lazy dog.
        The dog was not amused by the fox's clever antics.
        Brown foxes are known for their quick movements and intelligence.
        This is a test document with repeated words for analysis."""
    
    @pytest.fixture
    def mixed_case_text(self):
        """Text with mixed case for case sensitivity testing."""
        return "The QUICK Brown fox JUMPS over The lazy DOG."
    
    @pytest.fixture
    def special_chars_text(self):
        """Text with special characters and numbers."""
        return "Hello, world! This is a test... with numbers 123 and symbols @#$%."
    
    @pytest.fixture
    def unicode_text(self):
        """Text with Unicode characters."""
        return "Café naïve résumé Москва 北京 العربية हिन्दी"
    
    @pytest.fixture
    def empty_text(self):
        """Empty text for edge case testing.""" 
        return ""
    
    @pytest.fixture
    def whitespace_text(self):
        """Text with only whitespace."""
        return "   \t\n   \r\n   "
    
    def test_processor_initialization_default(self, default_processor):
        """Test TextProcessor initialization with default config."""
        assert default_processor is not None
        assert hasattr(default_processor, 'config')
        assert hasattr(default_processor, 'logger')
        assert hasattr(default_processor, 'performance_monitor')
        
        # Check default configuration
        config = default_processor.config
        assert config['case_sensitive'] is False
        assert config['remove_stop_words'] is False
        assert config['min_word_length'] >= 1
        assert config['max_word_length'] >= config['min_word_length']
    
    def test_processor_initialization_custom(self, custom_processor):
        """Test TextProcessor initialization with custom config."""
        config = custom_processor.config
        
        assert config['min_word_length'] == 3
        assert config['max_word_length'] == 15
        assert config['case_sensitive'] is True
        assert config['remove_stop_words'] is True
        assert config['stop_words_language'] == 'english'
    
    def test_load_stop_words(self):
        """Test stop words loading."""
        processor = TextProcessor({'remove_stop_words': True, 'stop_words_language': 'english'})
        
        assert 'stop_words' in processor.config
        stop_words = processor.config['stop_words']
        assert isinstance(stop_words, set)
        assert len(stop_words) > 0
        assert 'the' in stop_words
        assert 'and' in stop_words
        assert 'or' in stop_words
    
    def test_compile_patterns(self, default_processor):
        """Test regex pattern compilation."""
        assert hasattr(default_processor, 'word_pattern')
        assert hasattr(default_processor, 'whitespace_pattern')
        assert hasattr(default_processor, 'unicode_normalization_pattern')
        
        # Test that patterns work
        test_text = "Hello, world! 123"
        words = default_processor.word_pattern.findall(test_text)
        assert 'Hello' in words
        assert 'world' in words
    
    def test_clean_text_basic(self, default_processor, sample_text):
        """Test basic text cleaning."""
        cleaned_text, operations = default_processor.clean_text(sample_text)
        
        assert isinstance(cleaned_text, str)
        assert isinstance(operations, list)
        assert len(cleaned_text) > 0
        assert len(operations) > 0
        
        # Should be lowercase (case_sensitive=False by default)
        assert cleaned_text.islower() or not any(c.isupper() for c in cleaned_text)
        
        # Check operations applied
        assert 'case_normalization' in operations
        assert 'whitespace_normalization' in operations
    
    def test_clean_text_case_sensitive(self):
        """Test text cleaning with case sensitivity enabled."""
        processor = TextProcessor({'case_sensitive': True})
        text = "The QUICK Brown Fox"
        
        cleaned_text, operations = processor.clean_text(text)
        
        # Should preserve case
        assert 'QUICK' in cleaned_text or 'quick' not in cleaned_text.lower()
        assert 'case_normalization' not in operations
    
    def test_clean_text_unicode_normalization(self, default_processor, unicode_text):
        """Test Unicode normalization during cleaning."""
        cleaned_text, operations = default_processor.clean_text(unicode_text)
        
        assert isinstance(cleaned_text, str)
        assert 'unicode_normalization' in operations
        # Should handle Unicode characters gracefully
        assert len(cleaned_text) > 0
    
    def test_clean_text_empty_input(self, default_processor):
        """Test cleaning empty text."""
        with pytest.raises(InvalidTextError):
            default_processor.clean_text("")
    
    def test_clean_text_whitespace_only(self, default_processor, whitespace_text):
        """Test cleaning text with only whitespace."""
        with pytest.raises(InvalidTextError):
            default_processor.clean_text(whitespace_text)
    
    def test_extract_words_basic(self, default_processor, sample_text):
        """Test basic word extraction."""
        words = default_processor.extract_words(sample_text.lower())
        
        assert isinstance(words, list)
        assert len(words) > 0
        assert 'the' in words
        assert 'fox' in words
        assert 'dog' in words
        
        # Check word length filtering (default min=2, max=50)
        for word in words:
            assert len(word) >= default_processor.config['min_word_length']
            assert len(word) <= default_processor.config['max_word_length']
    
    def test_extract_words_length_filtering(self):
        """Test word extraction with length filtering."""
        processor = TextProcessor({
            'min_word_length': 4,
            'max_word_length': 8
        })
        
        text = "a big elephant runs very quickly through the forest"
        words = processor.extract_words(text)
        
        # Should only include words of length 4-8
        for word in words:
            assert 4 <= len(word) <= 8
        
        assert 'elephant' in words  # 8 chars
        assert 'quickly' in words   # 7 chars
        assert 'through' in words   # 7 chars
        assert 'forest' in words    # 6 chars
        
        # Should exclude short and long words
        assert 'a' not in words     # too short
        assert 'very' in words      # 4 chars - should be included
    
    def test_extract_words_stop_words_removal(self):
        """Test word extraction with stop words removal."""
        processor = TextProcessor({
            'remove_stop_words': True,
            'stop_words_language': 'english'
        })
        
        text = "the quick brown fox jumps over the lazy dog"
        words = processor.extract_words(text)
        
        # Stop words should be removed
        assert 'the' not in words
        assert 'over' not in words
        
        # Content words should remain
        assert 'quick' in words
        assert 'brown' in words
        assert 'fox' in words
        assert 'jumps' in words
        assert 'lazy' in words
        assert 'dog' in words
    
    def test_extract_words_special_characters(self, default_processor, special_chars_text):
        """Test word extraction with special characters."""
        words = default_processor.extract_words(special_chars_text.lower())
        
        # Should extract only alphabetic words
        assert 'hello' in words
        assert 'world' in words
        assert 'test' in words
        assert 'numbers' in words
        assert 'symbols' in words
        
        # Should not include numbers or symbols
        assert '123' not in words
        assert '@#$%' not in words
    
    def test_extract_words_empty_result(self):
        """Test word extraction resulting in no words."""
        processor = TextProcessor({'min_word_length': 10})
        text = "a big cat"  # All words < 10 chars
        
        words = processor.extract_words(text)
        assert words == []
    
    def test_analyze_word_frequencies_basic(self, default_processor):
        """Test basic word frequency analysis."""
        words = ['the', 'cat', 'sat', 'on', 'the', 'mat', 'the', 'cat']
        frequencies = default_processor.analyze_word_frequencies(words)
        
        assert isinstance(frequencies, Counter)
        assert frequencies['the'] == 3
        assert frequencies['cat'] == 2
        assert frequencies['sat'] == 1
        assert frequencies['on'] == 1
        assert frequencies['mat'] == 1
    
    def test_analyze_word_frequencies_empty_list(self, default_processor):
        """Test frequency analysis with empty word list."""
        frequencies = default_processor.analyze_word_frequencies([])
        
        assert isinstance(frequencies, Counter)
        assert len(frequencies) == 0
    
    def test_analyze_text_complete_pipeline(self, default_processor, sample_text):
        """Test complete text analysis pipeline."""
        analysis = default_processor.analyze_text(sample_text)
        
        # Check analysis structure
        assert isinstance(analysis, WordAnalysis)