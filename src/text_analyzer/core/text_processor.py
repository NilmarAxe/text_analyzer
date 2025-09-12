"""
Strategic text processing system for Text Analyzer.
Systematic approach to text cleaning, tokenization, and preprocessing operations.
"""

import re
from typing import List, Dict, Set, Any, Optional, Tuple, Union
from collections import Counter, defaultdict
from dataclasses import dataclass

from config import get_processor_logger, AnalysisConfig, LoggedOperation
from ..utils import (
    TextValidator,
    InvalidTextError,
    TextProcessingError,
    WordExtractionError,
    TextHelper,
    PerformanceMonitor,
    timing_decorator,
    DataStructureHelper
)

@dataclass
class ProcessingStats:
    """Statistics from text processing operations."""
    original_length: int
    processed_length: int
    total_words: int
    unique_words: int
    filtered_words: int
    processing_time: float
    operations_applied: List[str]

@dataclass
class WordAnalysis:
    """Comprehensive word analysis results."""
    word_frequencies: Counter
    word_lengths: Dict[int, int]
    character_frequencies: Counter
    processing_stats: ProcessingStats

class TextProcessor:
    """
    Strategic text processing with systematic optimization.
    Implements INTJ principles: efficiency, thoroughness, and systematic analysis.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = get_processor_logger()
        self.performance_monitor = PerformanceMonitor()
        
        # Load configuration with defaults
        self.config = self._load_config(config or {})
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
        
        self.logger.info("TextProcessor initialized with configuration")
    
    def _load_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Load and validate processor configuration."""
        default_config = {
            'min_word_length': AnalysisConfig.MIN_WORD_LENGTH,
            'max_word_length': AnalysisConfig.MAX_WORD_LENGTH,
            'case_sensitive': AnalysisConfig.CASE_SENSITIVE,
            'word_pattern': AnalysisConfig.WORD_PATTERN,
            'remove_stop_words': False,
            'stop_words_language': 'english',
            'normalize_unicode': True,
            'preserve_contractions': True,
            'remove_digits': False,
            'custom_stop_words': set()
        }
        
        # Merge with provided config
        merged_config = {**default_config, **config}
        
        # Load stop words if required
        if merged_config['remove_stop_words']:
            language = merged_config['stop_words_language']
            merged_config['stop_words'] = self._load_stop_words(language)
        else:
            merged_config['stop_words'] = set()
        
        return merged_config
    
    def _load_stop_words(self, language: str) -> Set[str]:
        """Load stop words for specified language."""
        stop_words = AnalysisConfig.DEFAULT_STOP_WORDS.get(language, set())
        stop_words.update(self.config.get('custom_stop_words', set()))
        
        self.logger.debug(f"Loaded {len(stop_words)} stop words for {language}")
        return stop_words
    
    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for efficient processing."""
        self.word_pattern = re.compile(self.config['word_pattern'], re.UNICODE)
        self.whitespace_pattern = re.compile(r'\s+')
        self.unicode_normalization_pattern = re.compile(r'[^\w\s]', re.UNICODE)
        
        # Digit removal pattern
        if self.config['remove_digits']:
            self.digit_pattern = re.compile(r'\d+')
        
        self.logger.debug("Regex patterns compiled for text processing")
    
    @timing_decorator("text_cleaning")
    def clean_text(self, text: str) -> Tuple[str, List[str]]:
        """
        Comprehensive text cleaning with systematic preprocessing.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Tuple of (cleaned_text, operations_applied)
            
        Raises:
            InvalidTextError: If text is invalid
            TextProcessingError: If cleaning fails
        """
        with LoggedOperation("Text Cleaning", self.logger):
            # Validate input text
            TextValidator.validate_text_content(text, min_length=1)
            
            operations_applied = []
            cleaned_text = text
            
            try:
                # Step 1: Unicode normalization
                if self.config['normalize_unicode']:
                    import unicodedata
                    cleaned_text = unicodedata.normalize('NFKD', cleaned_text)
                    operations_applied.append('unicode_normalization')
                
                # Step 2: Case normalization
                if not self.config['case_sensitive']:
                    cleaned_text = cleaned_text.lower()
                    operations_applied.append('case_normalization')
                
                # Step 3: Whitespace normalization
                cleaned_text = TextHelper.clean_whitespace(cleaned_text, normalize_spaces=True)
                operations_applied.append('whitespace_normalization')
                
                # Step 4: Remove digits if configured
                if self.config['remove_digits']:
                    cleaned_text = self.digit_pattern.sub('', cleaned_text)
                    operations_applied.append('digit_removal')
                
                self.logger.debug(f"Text cleaning complete. Applied: {operations_applied}")
                return cleaned_text, operations_applied
                
            except Exception as e:
                raise TextProcessingError('text_cleaning', str(e))
    
    @timing_decorator("word_extraction")
    def extract_words(self, text: str) -> List[str]:
        """
        Strategic word extraction using optimized regex patterns.
        
        Args:
            text: Text to process
            
        Returns:
            List of extracted words
            
        Raises:
            WordExtractionError: If word extraction fails
        """
        with LoggedOperation("Word Extraction", self.logger):
            try:
                # Extract words using compiled pattern
                words = self.word_pattern.findall(text)
                
                # Apply word length filtering
                filtered_words = [
                    word for word in words 
                    if self.config['min_word_length'] <= len(word) <= self.config['max_word_length']
                ]
                
                # Remove stop words if configured
                if self.config['remove_stop_words'] and self.config['stop_words']:
                    filtered_words = [
                        word for word in filtered_words 
                        if word.lower() not in self.config['stop_words']
                    ]
                
                self.logger.debug(f"Extracted {len(filtered_words)} words from {len(words)} raw matches")
                return filtered_words
                
            except Exception as e:
                raise WordExtractionError(self.config['word_pattern'], len(text))
    
    @timing_decorator("frequency_analysis")
    def analyze_word_frequencies(self, words: List[str]) -> Counter:
        """
        Efficient frequency analysis with optimization.
        
        Args:
            words: List of words to analyze
            
        Returns:
            Counter with word frequencies
        """
        with LoggedOperation("Frequency Analysis", self.logger):
            # Use Counter for O(n) frequency counting
            frequencies = Counter(words)
            
            self.logger.debug(f"Analyzed {len(words)} words, found {len(frequencies)} unique")
            return frequencies
    
    @timing_decorator("comprehensive_analysis")
    def analyze_text(self, text: str) -> WordAnalysis:
        """
        Comprehensive text analysis with full processing pipeline.
        
        Args:
            text: Text to analyze
            
        Returns:
            WordAnalysis with complete results
            
        Raises:
            TextProcessingError: If analysis fails
        """
        self.performance_monitor.start_monitoring()
        
        with LoggedOperation("Comprehensive Text Analysis", self.logger):
            try:
                original_length = len(text)
                
                # Step 1: Clean text
                cleaned_text, operations = self.clean_text(text)
                self.performance_monitor.checkpoint("text_cleaned")
                processed_length = len(cleaned_text)
                
                # Step 2: Extract words
                words = self.extract_words(cleaned_text)
                self.performance_monitor.checkpoint("words_extracted")
                
                # Step 3: Frequency analysis
                word_frequencies = self.analyze_word_frequencies(words)
                self.performance_monitor.checkpoint("frequencies_calculated")
                
                # Step 4: Additional analysis
                word_lengths = self._analyze_word_lengths(words)
                character_frequencies = self._analyze_character_frequencies(cleaned_text)
                self.performance_monitor.checkpoint("additional_analysis")
                
                # Generate processing statistics
                perf_report = self.performance_monitor.get_performance_report()
                processing_stats = ProcessingStats(
                    original_length=original_length,
                    processed_length=processed_length,
                    total_words=len(words),
                    unique_words=len(word_frequencies),
                    filtered_words=original_length - len(words),  # Approximation
                    processing_time=perf_report['total_execution_time'],
                    operations_applied=operations
                )
                
                # Create comprehensive analysis
                analysis = WordAnalysis(
                    word_frequencies=word_frequencies,
                    word_lengths=word_lengths,
                    character_frequencies=character_frequencies,
                    processing_stats=processing_stats
                )
                
                self.logger.info(f"Text analysis complete: {len(words)} words, {len(word_frequencies)} unique")
                return analysis
                
            except Exception as e:
                raise TextProcessingError('comprehensive_analysis', str(e))
    
    def _analyze_word_lengths(self, words: List[str]) -> Dict[int, int]:
        """
        Analyze distribution of word lengths.
        
        Args:
            words: List of words
            
        Returns:
            Dictionary mapping length to count
        """
        length_distribution = defaultdict(int)
        for word in words:
            length_distribution[len(word)] += 1
        return dict(length_distribution)
    
    def _analyze_character_frequencies(self, text: str) -> Counter:
        """
        Analyze character frequency distribution.
        
        Args:
            text: Text to analyze
            
        Returns:
            Counter with character frequencies
        """
        # Only count alphabetic characters
        chars = [c for c in text.lower() if c.isalpha()]
        return Counter(chars)
    
    @timing_decorator("batch_processing")
    def process_multiple_texts(self, texts: List[str]) -> List[WordAnalysis]:
        """
        Efficiently process multiple texts with batch optimization.
        
        Args:
            texts: List of texts to process
            
        Returns:
            List of WordAnalysis results
        """
        results = []
        
        with LoggedOperation(f"Batch Text Processing ({len(texts)} texts)", self.logger):
            for i, text in enumerate(texts):
                try:
                    analysis = self.analyze_text(text)
                    results.append(analysis)
                    self.logger.debug(f"Processed text {i+1}/{len(texts)}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to process text {i+1}: {e}")
                    # Create empty analysis for failed text
                    empty_analysis = WordAnalysis(
                        word_frequencies=Counter(),
                        word_lengths={},
                        character_frequencies=Counter(),
                        processing_stats=ProcessingStats(0, 0, 0, 0, 0, 0.0, ['error'])
                    )
                    results.append(empty_analysis)
        
        success_count = sum(1 for r in results if r.processing_stats.total_words > 0)
        self.logger.info(f"Batch processing complete: {success_count}/{len(texts)} successful")
        
        return results
    
    def get_processing_statistics(self, analysis: WordAnalysis) -> Dict[str, Any]:
        """
        Generate detailed processing statistics.
        
        Args:
            analysis: WordAnalysis to analyze
            
        Returns:
            Dictionary with comprehensive statistics
        """
        stats = analysis.processing_stats
        frequencies = analysis.word_frequencies
        
        # Calculate additional metrics
        avg_word_length = sum(len(word) * count for word, count in frequencies.items()) / stats.total_words if stats.total_words > 0 else 0
        vocabulary_richness = stats.unique_words / stats.total_words if stats.total_words > 0 else 0
        
        # Most/least frequent words
        most_common = frequencies.most_common(5) if frequencies else []
        least_common = frequencies.most_common()[-5:] if len(frequencies) >= 5 else []
        
        # Word length statistics
        if analysis.word_lengths:
            avg_length_weighted = sum(length * count for length, count in analysis.word_lengths.items()) / sum(analysis.word_lengths.values())
            max_word_length = max(analysis.word_lengths.keys())
            min_word_length = min(analysis.word_lengths.keys())
        else:
            avg_length_weighted = 0
            max_word_length = 0
            min_word_length = 0
        
        return {
            'processing_statistics': {
                'original_length': stats.original_length,
                'processed_length': stats.processed_length,
                'compression_ratio': stats.processed_length / stats.original_length if stats.original_length > 0 else 0,
                'processing_time': stats.processing_time,
                'words_per_second': stats.total_words / stats.processing_time if stats.processing_time > 0 else 0,
                'operations_applied': stats.operations_applied
            },
            'word_statistics': {
                'total_words': stats.total_words,
                'unique_words': stats.unique_words,
                'vocabulary_richness': vocabulary_richness,
                'average_word_length': avg_word_length,
                'average_word_length_weighted': avg_length_weighted,
                'max_word_length': max_word_length,
                'min_word_length': min_word_length
            },
            'frequency_statistics': {
                'most_common_words': most_common,
                'least_common_words': least_common,
                'hapax_legomena': len([word for word, count in frequencies.items() if count == 1]),  # Words appearing once
                'dis_legomena': len([word for word, count in frequencies.items() if count == 2])  # Words appearing twice
            },
            'distribution_statistics': {
                'word_length_distribution': dict(analysis.word_lengths),
                'character_distribution': dict(analysis.character_frequencies.most_common(10))
            }
        }
    
    def optimize_processing_config(self, sample_text: str) -> Dict[str, Any]:
        """
        Optimize processing configuration based on sample text.
        
        Args:
            sample_text: Sample text for optimization
            
        Returns:
            Optimized configuration recommendations
        """
        with LoggedOperation("Configuration Optimization", self.logger):
            # Analyze sample text characteristics
            sample_words = self.word_pattern.findall(sample_text.lower())
            
            if not sample_words:
                return self.config  # Return current config if no words found
            
            # Calculate optimal parameters
            word_lengths = [len(word) for word in sample_words]
            avg_length = sum(word_lengths) / len(word_lengths)
            
            # Determine optimal min/max word lengths
            optimal_min_length = max(1, int(avg_length * 0.3))
            optimal_max_length = min(50, int(avg_length * 2.5))
            
            # Analyze character distribution for language detection
            char_freq = Counter(char.lower() for char in sample_text if char.isalpha())
            
            # Generate recommendations
            recommendations = {
                'recommended_min_word_length': optimal_min_length,
                'recommended_max_word_length': optimal_max_length,
                'estimated_language': self._detect_language(char_freq),
                'should_remove_stop_words': len(sample_words) > 1000,  # Recommend for longer texts
                'optimal_case_sensitive': self._should_be_case_sensitive(sample_text),
                'processing_complexity': self._estimate_complexity(sample_text)
            }
            
            self.logger.info("Configuration optimization complete")
            return recommendations
    
    def _detect_language(self, char_freq: Counter) -> str:
        """Simple language detection based on character frequencies."""
        # Very basic language detection - can be enhanced
        common_english_chars = set('etaoinshrdlu')
        english_score = sum(char_freq[c] for c in common_english_chars if c in char_freq)
        total_chars = sum(char_freq.values())
        
        english_ratio = english_score / total_chars if total_chars > 0 else 0
        
        return 'english' if english_ratio > 0.6 else 'unknown'
    
    def _should_be_case_sensitive(self, text: str) -> bool:
        """Determine if text should be processed case-sensitively."""
        upper_count = sum(1 for c in text if c.isupper())
        total_alpha = sum(1 for c in text if c.isalpha())
        
        if total_alpha == 0:
            return False
        
        upper_ratio = upper_count / total_alpha
        # Recommend case sensitivity if more than 20% uppercase (excluding first letters)
        return upper_ratio > 0.2
    
    def _estimate_complexity(self, text: str) -> str:
        """Estimate text processing complexity."""
        text_length = len(text)
        word_count = len(text.split())
        
        if text_length < 1000:
            return 'low'
        elif text_length < 10000:
            return 'medium'
        elif text_length < 100000:
            return 'high'
        else:
            return 'very_high'