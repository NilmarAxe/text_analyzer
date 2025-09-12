"""
Strategic pytest configuration for Text Analyzer test suite.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from collections import Counter
import os
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


# Test configuration
def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )


# Session-scoped fixtures
@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary directory for test data files."""
    temp_dir = tempfile.mkdtemp(prefix="text_analyzer_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def sample_texts():
    """Collection of sample texts for testing."""
    return {
        'simple': "The quick brown fox jumps over the lazy dog.",
        'multiline': """The quick brown fox jumps over the lazy dog.
        The dog was not amused by the fox's antics.
        Brown foxes are known for their quick movements.""",
        'repeated_words': """The the the fox fox dog dog dog dog.
        Quick quick brown brown brown.""",
        'mixed_case': "The QUICK Brown FOX jumps Over the LAZY dog.",
        'with_numbers': "The 123 quick 456 brown fox jumps over 789 lazy dogs.",
        'with_punctuation': "Hello, world! How are you? Fine, thanks... Great!",
        'unicode': "Café naïve résumé Москва 北京 العربية हिन्दी",
        'empty': "",
        'whitespace_only': "   \t\n   \r\n   ",
        'long_text': " ".join(["word"] * 1000) + ".",
        'single_word': "hello",
        'contractions': "don't can't won't shouldn't I'll you're we're they're",
        'special_chars': "@#$%^&*()[]{}|\\:;\"'<>,.?/~`",
        'minimal': "a b c d e f g h i j k l m n o p q r s t u v w x y z"
    }


@pytest.fixture(scope="session")
def create_test_files(test_data_dir, sample_texts):
    """Create test files with various content types."""
    test_files = {}
    
    for name, content in sample_texts.items():
        if content:  # Skip empty content for file creation
            file_path = test_data_dir / f"{name}.txt"
            file_path.write_text(content, encoding='utf-8')
            test_files[name] = file_path
    
    # Create files with different encodings
    encodings_dir = test_data_dir / "encodings"
    encodings_dir.mkdir(exist_ok=True)
    
    utf8_file = encodings_dir / "utf8_text.txt"
    utf8_file.write_text("UTF-8 encoded text with special chars: àáâãäå", encoding='utf-8')
    test_files['utf8'] = utf8_file
    
    latin1_file = encodings_dir / "latin1_text.txt"
    latin1_file.write_text("Latin-1 text with chars: café résumé naïve", encoding='latin-1')
    test_files['latin1'] = latin1_file
    
    # Create binary file (non-text)
    binary_file = test_data_dir / "binary_file.bin"
    binary_file.write_bytes(b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09')
    test_files['binary'] = binary_file
    
    # Create empty file
    empty_file = test_data_dir / "empty_file.txt"
    empty_file.touch()
    test_files['empty_file'] = empty_file
    
    # Create large file
    large_content = "This is a large file for testing. " * 2000
    large_file = test_data_dir / "large_file.txt"
    large_file.write_text(large_content, encoding='utf-8')
    test_files['large'] = large_file
    
    return test_files


# Function-scoped fixtures
@pytest.fixture
def temp_file():
    """Create temporary file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    temp_path = Path(tempfile.mkdtemp(prefix="test_"))
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_word_frequencies():
    """Sample word frequency data for testing."""
    return Counter({
        'the': 150,
        'and': 89,
        'of': 76,
        'to': 65,
        'a': 58,
        'in': 45,
        'for': 42,
        'is': 38,
        'on': 35,
        'that': 32,
        'by': 28,
        'this': 25,
        'with': 22,
        'i': 20,
        'you': 18,
        'it': 15,
        'not': 13,
        'or': 11,
        'be': 10,
        'are': 9,
        'from': 8,
        'at': 7,
        'as': 6,
        'your': 5,
        'all': 4,
        'any': 3,
        'can': 2,
        'had': 1
    })


@pytest.fixture
def sample_file_info():
    """Sample file information for testing."""
    return {
        'path': '/test/sample_document.txt',
        'size': 2048,
        'encoding': 'utf-8',
        'lines': 25,
        'checksum': 'abc123def456'
    }


@pytest.fixture
def sample_processing_stats():
    """Sample processing statistics for testing."""
    return {
        'original_length': 1500,
        'processed_length': 1200,
        'total_words': 250,
        'unique_words': 125,
        'filtered_words': 25,
        'processing_time': 0.345,
        'operations_applied': ['unicode_normalization', 'case_normalization', 'whitespace_normalization']
    }


@pytest.fixture
def complete_analysis_data(sample_word_frequencies, sample_file_info, sample_processing_stats):
    """Complete analysis data combining all components."""
    return {
        'word_frequencies': sample_word_frequencies,
        'file_info': sample_file_info,
        'processing_stats': sample_processing_stats,
        'word_length_distribution': {
            2: 45,
            3: 78,
            4: 92,
            5: 67,
            6: 43,
            7: 28,
            8: 15,
            9: 8,
            10: 4
        },
        'character_distribution': {
            'e': 120,
            't': 95,
            'a': 87,
            'o': 78,
            'i': 72,
            'n': 65,
            's': 58,
            'h': 52,
            'r': 47,
            'd': 42,
            'l': 38,
            'c': 34,
            'u': 30,
            'm': 26,
            'w': 22,
            'f': 18,
            'g': 15,
            'y': 12,
            'p': 10,
            'b': 8,
            'v': 6,
            'k': 4,
            'j': 2,
            'x': 1,
            'q': 1,
            'z': 1
        },
        'top_count': 10
    }


# Utility fixtures
@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    from unittest.mock import Mock
    logger = Mock()
    logger.info = Mock()
    logger.debug = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    return logger


@pytest.fixture
def performance_timer():
    """Simple performance timer for testing."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
            return self.elapsed()
        
        def elapsed(self):
            if self.start_time is None:
                return 0
            end = self.end_time if self.end_time else time.time()
            return end - self.start_time
    
    return Timer()


# Configuration fixtures
@pytest.fixture
def default_config():
    """Default configuration for testing."""
    return {
        'top_words_count': 10,
        'min_word_length': 2,
        'max_word_length': 50,
        'case_sensitive': False,
        'remove_stop_words': False,
        'word_pattern': r'\b[a-zA-Z]+\b',
        'normalize_unicode': True,
        'preserve_contractions': True,
        'remove_digits': False
    }


@pytest.fixture
def custom_config():
    """Custom configuration for advanced testing."""
    return {
        'top_words_count': 15,
        'min_word_length': 3,
        'max_word_length': 20,
        'case_sensitive': True,
        'remove_stop_words': True,
        'stop_words_language': 'english',
        'word_pattern': r'\b[a-zA-Z]{3,20}\b',
        'normalize_unicode': True,
        'preserve_contractions': False,
        'remove_digits': True,
        'detailed_analysis': True,
        'memory_limit_mb': 256
    }


# Parameterized fixtures
@pytest.fixture(params=['txt', 'json', 'csv', 'html'])
def report_format(request):
    """Parametrized fixture for testing all report formats."""
    return request.param


@pytest.fixture(params=['utf-8', 'latin-1', 'cp1252'])
def encoding(request):
    """Parametrized fixture for testing different encodings."""
    return request.param


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Automatically cleanup temporary files after each test."""
    temp_files = []
    
    def register_temp_file(filepath):
        temp_files.append(Path(filepath))
    
    yield register_temp_file
    
    # Cleanup
    for temp_file in temp_files:
        if temp_file.exists():
            try:
                if temp_file.is_file():
                    temp_file.unlink()
                elif temp_file.is_dir():
                    shutil.rmtree(temp_file)
            except (OSError, PermissionError):
                pass  # Ignore cleanup errors


# Performance testing fixtures
@pytest.fixture
def performance_benchmark():
    """Benchmark fixture for performance testing."""
    benchmarks = {}
    
    def add_benchmark(name, duration, iterations=1):
        benchmarks[name] = {
            'duration': duration,
            'iterations': iterations,
            'avg_duration': duration / iterations
        }
    
    def get_benchmark(name):
        return benchmarks.get(name)
    
    def get_all_benchmarks():
        return benchmarks.copy()
    
    class Benchmark:
        add = staticmethod(add_benchmark)
        get = staticmethod(get_benchmark)
        all = staticmethod(get_all_benchmarks)
    
    return Benchmark


# Integration testing fixtures
@pytest.fixture
def integration_environment(test_data_dir, create_test_files):
    """Complete environment setup for integration testing."""
    env = {
        'data_dir': test_data_dir,
        'test_files': create_test_files,
        'output_dir': test_data_dir / 'output',
        'config_dir': test_data_dir / 'config'
    }
    
    # Create additional directories
    env['output_dir'].mkdir(exist_ok=True)
    env['config_dir'].mkdir(exist_ok=True)
    
    # Create sample config file
    import json
    config_file = env['config_dir'] / 'test_config.json'
    config_data = {
        'analysis': {
            'top_words_count': 15,
            'case_sensitive': False,
            'remove_stop_words': True
        },
        'output': {
            'default_format': 'txt',
            'detailed_analysis': True
        },
        'performance': {
            'memory_limit_mb': 512
        }
    }
    config_file.write_text(json.dumps(config_data, indent=2))
    env['config_file'] = config_file
    
    return env


# Error testing fixtures
@pytest.fixture
def error_conditions():
    """Various error conditions for testing error handling."""
    return {
        'file_not_found': '/nonexistent/path/file.txt',
        'permission_denied': '/root/restricted_file.txt',
        'invalid_encoding': 'invalid-encoding-name',
        'empty_content': '',
        'whitespace_only': '   \t\n   ',
        'binary_content': b'\x00\x01\x02\x03\x04\x05',
        'very_large_file': 'x' * (100 * 1024 * 1024),  # 100MB string
        'invalid_json': '{"invalid": json content}',
        'malformed_csv': 'invalid,csv\ncontent"with"quotes',
        'invalid_regex': '[invalid regex(',
        'negative_numbers': -1,
        'zero_values': 0,
        'none_values': None
    }


# Mock fixtures for external dependencies
@pytest.fixture
def mock_system_info():
    """Mock system information for testing."""
    return {
        'cpu_count': 8,
        'memory_total': 16 * 1024 * 1024 * 1024,  # 16GB
        'memory_available': 8 * 1024 * 1024 * 1024,  # 8GB
        'disk_usage': {
            'total': 1000 * 1024 * 1024 * 1024,  # 1TB
            'free': 500 * 1024 * 1024 * 1024     # 500GB
        },
        'python_version': '3.9.0',
        'platform': 'Linux'
    }


@pytest.fixture
def mock_file_operations():
    """Mock file operations for testing without actual file I/O."""
    from unittest.mock import Mock, patch
    
    class MockFileOps:
        def __init__(self):
            self.read_calls = []
            self.write_calls = []
            self.exists_calls = []
        
        def mock_read_text(self, content='test content'):
            def _read_text(encoding='utf-8'):
                self.read_calls.append({'encoding': encoding})
                return content
            return _read_text
        
        def mock_write_text(self):
            def _write_text(content, encoding='utf-8'):
                self.write_calls.append({'content': content, 'encoding': encoding})
            return _write_text
        
        def mock_exists(self, exists=True):
            def _exists():
                self.exists_calls.append(True)
                return exists
            return _exists
    
    return MockFileOps()


# Data validation fixtures
@pytest.fixture
def validation_test_cases():
    """Test cases for validation testing."""
    return {
        'valid_paths': [
            '/valid/path/file.txt',
            'relative/path/file.txt',
            './current/dir/file.txt',
            '../parent/dir/file.txt'
        ],
        'invalid_paths': [
            '',
            None,
            123,
            '/dev/null/invalid',
            '\x00invalid'
        ],
        'valid_encodings': [
            'utf-8',
            'utf-16',
            'latin-1',
            'cp1252',
            'ascii'
        ],
        'invalid_encodings': [
            'invalid-encoding',
            '',
            None,
            123,
            'utf-99'
        ],
        'valid_formats': [
            'txt',
            'json',
            'csv',
            'html'
        ],
        'invalid_formats': [
            'pdf',
            'docx',
            'xml',
            '',
            None,
            123
        ],
        'valid_word_lengths': [
            (1, 50),
            (2, 20),
            (3, 15),
            (1, 100)
        ],
        'invalid_word_lengths': [
            (0, 10),
            (-1, 10),
            (10, 5),  # min > max
            (50, 50)  # min == max
        ],
        'valid_top_counts': [1, 5, 10, 25, 50, 100],
        'invalid_top_counts': [-1, 0, -10, None, 'ten', 3.14]
    }


# Regression testing fixtures
@pytest.fixture
def regression_test_data():
    """Data for regression testing - known good results."""
    return {
        'simple_text_analysis': {
            'input': "the quick brown fox jumps over the lazy dog",
            'expected_word_count': 9,
            'expected_unique_count': 8,
            'expected_top_word': 'the',
            'expected_top_count': 2
        },
        'case_sensitive_analysis': {
            'input': "The THE the Quick QUICK quick",
            'case_sensitive_unique': 6,
            'case_insensitive_unique': 2
        },
        'stop_words_analysis': {
            'input': "the quick brown fox jumps over the lazy dog",
            'with_stop_words': 9,
            'without_stop_words': 6  # removes 'the', 'over', 'the'
        },
        'word_length_filtering': {
            'input': "a big elephant runs very quickly",
            'min_length_4': 4,  # 'elephant', 'runs', 'very', 'quickly'
            'max_length_5': 5   # 'big', 'runs', 'very' (excluding 'elephant', 'quickly')
        }
    }


# Stress testing fixtures
@pytest.fixture
def stress_test_data():
    """Data for stress testing the system."""
    def generate_large_text(word_count):
        words = ['word', 'test', 'analysis', 'frequency', 'text', 'data']
        import random
        return ' '.join(random.choices(words, k=word_count))
    
    def generate_many_unique_words(count):
        return ' '.join([f'word_{i}' for i in range(count)])
    
    def generate_repeated_word(word, count):
        return ' '.join([word] * count)
    
    return {
        'large_text_10k': generate_large_text(10000),
        'large_text_100k': generate_large_text(100000),
        'many_unique_1k': generate_many_unique_words(1000),
        'many_unique_10k': generate_many_unique_words(10000),
        'repeated_word_1k': generate_repeated_word('test', 1000),
        'repeated_word_10k': generate_repeated_word('analysis', 10000)
    }


# Skip conditions for different environments
def pytest_runtest_setup(item):
    """Setup function to handle test skipping based on conditions."""
    # Skip slow tests unless specifically requested
    if "slow" in item.keywords and not item.config.getoption("--runslow", default=False):
        pytest.skip("need --runslow option to run slow tests")
    
    # Skip integration tests unless specifically requested  
    if "integration" in item.keywords and not item.config.getoption("--integration", default=False):
        pytest.skip("need --integration option to run integration tests")
    
    # Skip performance tests unless specifically requested
    if "performance" in item.keywords and not item.config.getoption("--performance", default=False):
        pytest.skip("need --performance option to run performance tests")


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--runslow", 
        action="store_true", 
        default=False, 
        help="run slow tests"
    )
    parser.addoption(
        "--integration", 
        action="store_true", 
        default=False, 
        help="run integration tests"
    )
    parser.addoption(
        "--performance", 
        action="store_true", 
        default=False, 
        help="run performance tests"
    )
    parser.addoption(
        "--benchmark", 
        action="store_true", 
        default=False, 
        help="run benchmark tests"
    )


# Test result collection
@pytest.fixture
def test_results_collector():
    """Collector for test results and metrics."""
    results = {
        'passed': 0,
        'failed': 0,
        'skipped': 0,
        'errors': [],
        'performance_data': {},
        'coverage_data': {}
    }
    
    def record_result(test_name, status, duration=None, error=None):
        results[status] += 1
        if error:
            results['errors'].append({'test': test_name, 'error': str(error)})
        if duration:
            results['performance_data'][test_name] = duration
    
    def get_summary():
        total = results['passed'] + results['failed'] + results['skipped']
        return {
            'total_tests': total,
            'success_rate': results['passed'] / total if total > 0 else 0,
            'failure_rate': results['failed'] / total if total > 0 else 0,
            **results
        }
    
    class ResultsCollector:
        record = staticmethod(record_result)
        summary = staticmethod(get_summary)
        data = results
    
    return ResultsCollector


# Memory management fixtures
@pytest.fixture
def memory_monitor():
    """Monitor memory usage during tests."""
    import psutil
    import os
    
    class MemoryMonitor:
        def __init__(self):
            self.process = psutil.Process(os.getpid())
            self.initial_memory = self.get_memory_usage()
            self.peak_memory = self.initial_memory
        
        def get_memory_usage(self):
            return self.process.memory_info().rss / 1024 / 1024  # MB
        
        def update_peak(self):
            current = self.get_memory_usage()
            if current > self.peak_memory:
                self.peak_memory = current
            return current
        
        def get_memory_increase(self):
            return self.get_memory_usage() - self.initial_memory
        
        def get_peak_increase(self):
            return self.peak_memory - self.initial_memory
        
        def reset(self):
            self.initial_memory = self.get_memory_usage()
            self.peak_memory = self.initial_memory
    
    return MemoryMonitor()


# Final cleanup
@pytest.fixture(scope="session", autouse=True)
def global_cleanup():
    """Global cleanup after all tests."""
    yield
    
    # Cleanup any remaining temporary files
    import tempfile
    import glob
    
    temp_pattern = os.path.join(tempfile.gettempdir(), "text_analyzer_*")
    for temp_item in glob.glob(temp_pattern):
        temp_path = Path(temp_item)
        try:
            if temp_path.is_file():
                temp_path.unlink()
            elif temp_path.is_dir():
                shutil.rmtree(temp_path)
        except (OSError, PermissionError):
            pass  # Ignore cleanup errors