"""
Utility helper functions for the Text Analyzer system.
"""

import time
import psutil
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Union, Callable
from datetime import datetime
from functools import wraps
import hashlib
import json


class PerformanceMonitor:
    """Strategic performance monitoring and optimization utilities."""
    
    def __init__(self):
        self.start_time = None
        self.checkpoints = {}
        self.memory_snapshots = {}
    
    def start_monitoring(self) -> None:
        """Initialize performance monitoring session."""
        self.start_time = time.time()
        self.checkpoints.clear()
        self.memory_snapshots.clear()
        self._record_memory_snapshot('start')
    
    def checkpoint(self, name: str) -> float:
        """
        Record performance checkpoint.
        
        Args:
            name: Checkpoint identifier
            
        Returns:
            Elapsed time since monitoring start
        """
        if self.start_time is None:
            self.start_monitoring()
        
        elapsed = time.time() - self.start_time
        self.checkpoints[name] = elapsed
        self._record_memory_snapshot(name)
        
        return elapsed
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Returns:
            Performance metrics dictionary
        """
        if not self.checkpoints:
            return {'error': 'No checkpoints recorded'}
        
        total_time = max(self.checkpoints.values()) if self.checkpoints else 0
        
        return {
            'total_execution_time': total_time,
            'checkpoints': self.checkpoints,
            'memory_usage': self.memory_snapshots,
            'performance_summary': self._analyze_performance()
        }
    
    def _record_memory_snapshot(self, checkpoint: str) -> None:
        """Record memory usage at checkpoint."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            self.memory_snapshots[checkpoint] = {
                'rss': memory_info.rss,  # Resident Set Size
                'vms': memory_info.vms,  # Virtual Memory Size
                'percent': process.memory_percent()
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            self.memory_snapshots[checkpoint] = {'error': 'Unable to access memory info'}
    
    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze performance patterns and identify bottlenecks."""
        if len(self.checkpoints) < 2:
            return {'analysis': 'Insufficient data for analysis'}
        
        checkpoint_times = list(self.checkpoints.items())
        durations = []
        
        for i in range(1, len(checkpoint_times)):
            duration = checkpoint_times[i][1] - checkpoint_times[i-1][1]
            durations.append((f"{checkpoint_times[i-1][0]} -> {checkpoint_times[i][0]}", duration))
        
        # Find slowest operation
        slowest_operation = max(durations, key=lambda x: x[1]) if durations else None
        
        return {
            'operation_durations': durations,
            'slowest_operation': slowest_operation,
            'average_operation_time': sum(d[1] for d in durations) / len(durations) if durations else 0
        }


def timing_decorator(operation_name: str):
    """
    Decorator for automatic function timing.
    
    Args:
        operation_name: Name of the operation for reporting
        
    Returns:
        Decorated function with timing capabilities
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Add timing info to result if it's a dict
                if isinstance(result, dict):
                    result['_timing'] = {
                        'operation': operation_name,
                        'duration': duration,
                        'timestamp': datetime.now().isoformat()
                    }
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                # Log timing even for failed operations
                print(f"Operation '{operation_name}' failed after {duration:.3f}s: {e}")
                raise
                
        return wrapper
    return decorator


class FileSystemHelper:
    """Strategic file system operations and utilities."""
    
    @staticmethod
    def ensure_directory(path: Union[str, Path]) -> Path:
        """
        Ensure directory exists, creating if necessary.
        
        Args:
            path: Directory path
            
        Returns:
            Path object for the directory
        """
        dir_path = Path(path)
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path
    
    @staticmethod
    def get_file_info(filepath: Path) -> Dict[str, Any]:
        """
        Get comprehensive file information.
        
        Args:
            filepath: File to analyze
            
        Returns:
            Dictionary with file metadata
        """
        try:
            stat = filepath.stat()
            return {
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime),
                'created': datetime.fromtimestamp(stat.st_ctime),
                'extension': filepath.suffix.lower(),
                'name': filepath.name,
                'parent': str(filepath.parent),
                'absolute_path': str(filepath.absolute())
            }
        except OSError as e:
            return {'error': str(e)}
    
    @staticmethod
    def generate_unique_filename(base_path: Path, extension: str = '') -> Path:
        """
        Generate unique filename to avoid conflicts.
        
        Args:
            base_path: Base path for the file
            extension: File extension to add
            
        Returns:
            Unique file path
        """
        counter = 1
        original_path = base_path.with_suffix(extension) if extension else base_path
        unique_path = original_path
        
        while unique_path.exists():
            stem = base_path.stem
            suffix = base_path.suffix if not extension else extension
            unique_path = base_path.parent / f"{stem}_{counter}{suffix}"
            counter += 1
        
        return unique_path
    
    @staticmethod
    def calculate_file_hash(filepath: Path, algorithm: str = 'sha256') -> str:
        """
        Calculate file hash for integrity verification.
        
        Args:
            filepath: File to hash
            algorithm: Hash algorithm ('md5', 'sha1', 'sha256', 'sha512')
            
        Returns:
            Hexadecimal hash string
        """
        hash_algo = hashlib.new(algorithm)
        
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hash_algo.update(chunk)
        
        return hash_algo.hexdigest()


class TextHelper:
    """Strategic text processing utilities."""
    
    @staticmethod
    def clean_whitespace(text: str, normalize_spaces: bool = True) -> str:
        """
        Clean and normalize whitespace in text.
        
        Args:
            text: Text to clean
            normalize_spaces: Whether to normalize multiple spaces to single space
            
        Returns:
            Cleaned text
        """
        # Remove leading/trailing whitespace
        cleaned = text.strip()
        
        if normalize_spaces:
            # Replace multiple whitespace characters with single space
            import re
            cleaned = re.sub(r'\s+', ' ', cleaned)
        
        return cleaned
    
    @staticmethod
    def truncate_text(text: str, max_length: int, suffix: str = '...') -> str:
        """
        Truncate text to specified length with suffix.
        
        Args:
            text: Text to truncate
            max_length: Maximum length including suffix
            suffix: Suffix to add to truncated text
            
        Returns:
            Truncated text
        """
        if len(text) <= max_length:
            return text
        
        truncate_length = max_length - len(suffix)
        return text[:truncate_length] + suffix
    
    @staticmethod
    def extract_sample(text: str, sample_length: int = 100, from_middle: bool = False) -> str:
        """
        Extract representative sample from text.
        
        Args:
            text: Source text
            sample_length: Length of sample to extract
            from_middle: Whether to sample from middle instead of beginning
            
        Returns:
            Text sample
        """
        if len(text) <= sample_length:
            return text
        
        if from_middle:
            start = (len(text) - sample_length) // 2
            return text[start:start + sample_length]
        else:
            return text[:sample_length]
    
    @staticmethod
    def count_lines(text: str) -> int:
        """
        Count lines in text efficiently.
        
        Args:
            text: Text to count
            
        Returns:
            Number of lines
        """
        return text.count('\n') + 1 if text else 0
    
    @staticmethod
    def estimate_reading_time(text: str, words_per_minute: int = 200) -> float:
        """
        Estimate reading time for text.
        
        Args:
            text: Text to analyze
            words_per_minute: Average reading speed
            
        Returns:
            Estimated reading time in minutes
        """
        word_count = len(text.split())
        return word_count / words_per_minute


class DataStructureHelper:
    """Efficient data structure operations and utilities."""
    
    @staticmethod
    def merge_counters(*counters) -> Dict[Any, int]:
        """
        Efficiently merge multiple counter dictionaries.
        
        Args:
            *counters: Counter dictionaries to merge
            
        Returns:
            Merged counter dictionary
        """
        merged = {}
        for counter in counters:
            for key, value in counter.items():
                merged[key] = merged.get(key, 0) + value
        return merged
    
    @staticmethod
    def sort_dict_by_value(dictionary: Dict[Any, Any], reverse: bool = True) -> List[Tuple[Any, Any]]:
        """
        Sort dictionary by values efficiently.
        
        Args:
            dictionary: Dictionary to sort
            reverse: Whether to sort in descending order
            
        Returns:
            List of (key, value) tuples sorted by value
        """
        return sorted(dictionary.items(), key=lambda item: item[1], reverse=reverse)
    
    @staticmethod
    def filter_dict_by_value(dictionary: Dict[Any, Any], 
                            min_value: Optional[Any] = None,
                            max_value: Optional[Any] = None) -> Dict[Any, Any]:
        """
        Filter dictionary entries by value range.
        
        Args:
            dictionary: Dictionary to filter
            min_value: Minimum value threshold
            max_value: Maximum value threshold
            
        Returns:
            Filtered dictionary
        """
        filtered = {}
        
        for key, value in dictionary.items():
            if min_value is not None and value < min_value:
                continue
            if max_value is not None and value > max_value:
                continue
            filtered[key] = value
        
        return filtered
    
    @staticmethod
    def get_top_n_items(dictionary: Dict[Any, Any], n: int = 10) -> List[Tuple[Any, Any]]:
        """
        Get top N items from dictionary by value.
        
        Args:
            dictionary: Dictionary to process
            n: Number of top items to return
            
        Returns:
            List of top N (key, value) tuples
        """
        return DataStructureHelper.sort_dict_by_value(dictionary)[:n]


class FormattingHelper:
    """Strategic text formatting and presentation utilities."""
    
    @staticmethod
    def format_number(number: Union[int, float], thousands_separator: str = ',') -> str:
        """
        Format numbers with thousands separators.
        
        Args:
            number: Number to format
            thousands_separator: Separator character
            
        Returns:
            Formatted number string
        """
        if isinstance(number, int):
            return f"{number:,}".replace(',', thousands_separator)
        elif isinstance(number, float):
            return f"{number:,.2f}".replace(',', thousands_separator)
        else:
            return str(number)
    
    @staticmethod
    def format_percentage(value: float, total: float, decimal_places: int = 2) -> str:
        """
        Format percentage values consistently.
        
        Args:
            value: Numerator value
            total: Denominator value
            decimal_places: Number of decimal places
            
        Returns:
            Formatted percentage string
        """
        if total == 0:
            return "0.00%"
        
        percentage = (value / total) * 100
        return f"{percentage:.{decimal_places}f}%"
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """
        Format duration in human-readable format.
        
        Args:
            seconds: Duration in seconds
            
        Returns:
            Formatted duration string
        """
        if seconds < 1:
            return f"{seconds*1000:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            remaining_seconds = seconds % 60
            return f"{minutes}m {remaining_seconds:.1f}s"
        else:
            hours = int(seconds // 3600)
            remaining_minutes = int((seconds % 3600) // 60)
            return f"{hours}h {remaining_minutes}m"
    
    @staticmethod
    def format_bytes(bytes_value: int) -> str:
        """
        Format byte values in human-readable format.
        
        Args:
            bytes_value: Size in bytes
            
        Returns:
            Formatted size string
        """
        units = ['B', 'KB', 'MB', 'GB', 'TB']
        size = float(bytes_value)
        unit_index = 0
        
        while size >= 1024 and unit_index < len(units) - 1:
            size /= 1024
            unit_index += 1
        
        if unit_index == 0:
            return f"{int(size)} {units[unit_index]}"
        else:
            return f"{size:.2f} {units[unit_index]}"
    
    @staticmethod
    def create_separator_line(width: int = 80, char: str = '=') -> str:
        """
        Create separator line for text formatting.
        
        Args:
            width: Line width
            char: Character to use
            
        Returns:
            Separator line string
        """
        return char * width
    
    @staticmethod
    def center_text(text: str, width: int = 80, fill_char: str = ' ') -> str:
        """
        Center text within specified width.
        
        Args:
            text: Text to center
            width: Total width
            fill_char: Fill character
            
        Returns:
            Centered text string
        """
        return text.center(width, fill_char)
    
    @staticmethod
    def create_table_row(columns: List[str], widths: List[int], 
                        separator: str = ' | ', alignment: str = 'left') -> str:
        """
        Create formatted table row.
        
        Args:
            columns: Column values
            widths: Column widths
            separator: Column separator
            alignment: Text alignment ('left', 'right', 'center')
            
        Returns:
            Formatted table row
        """
        formatted_columns = []
        
        for i, (column, width) in enumerate(zip(columns, widths)):
            column_str = str(column)
            
            if alignment == 'left':
                formatted_columns.append(column_str.ljust(width))
            elif alignment == 'right':
                formatted_columns.append(column_str.rjust(width))
            else:  # center
                formatted_columns.append(column_str.center(width))
        
        return separator.join(formatted_columns)


class ConfigurationHelper:
    """Configuration management and serialization utilities."""
    
    @staticmethod
    def save_config_to_json(config: Dict[str, Any], filepath: Path) -> bool:
        """
        Save configuration to JSON file.
        
        Args:
            config: Configuration dictionary
            filepath: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, default=str)
            return True
        except (IOError, TypeError) as e:
            print(f"Error saving configuration: {e}")
            return False
    
    @staticmethod
    def load_config_from_json(filepath: Path) -> Optional[Dict[str, Any]]:
        """
        Load configuration from JSON file.
        
        Args:
            filepath: Configuration file path
            
        Returns:
            Configuration dictionary or None if failed
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error loading configuration: {e}")
            return None
    
    @staticmethod
    def merge_configurations(base_config: Dict[str, Any], 
                           override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge configuration dictionaries with override priority.
        
        Args:
            base_config: Base configuration
            override_config: Override configuration
            
        Returns:
            Merged configuration
        """
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                merged[key] = ConfigurationHelper.merge_configurations(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    @staticmethod
    def validate_config_structure(config: Dict[str, Any], 
                                 required_keys: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate configuration structure.
        
        Args:
            config: Configuration to validate
            required_keys: List of required configuration keys
            
        Returns:
            Tuple of (is_valid, missing_keys)
        """
        missing_keys = []
        
        for key in required_keys:
            if key not in config:
                missing_keys.append(key)
        
        return len(missing_keys) == 0, missing_keys


class SystemHelper:
    """System-level utility functions."""
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """
        Get comprehensive system information.
        
        Returns:
            System information dictionary
        """
        try:
            return {
                'cpu_count': psutil.cpu_count(),
                'memory_total': psutil.virtual_memory().total,
                'memory_available': psutil.virtual_memory().available,
                'disk_usage': {
                    'total': psutil.disk_usage('/').total,
                    'free': psutil.disk_usage('/').free
                },
                'python_version': f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}.{__import__('sys').version_info.micro}",
                'platform': __import__('platform').system()
            }
        except Exception as e:
            return {'error': f'Unable to retrieve system info: {e}'}
    
    @staticmethod
    def check_memory_usage() -> Dict[str, Any]:
        """
        Check current memory usage.
        
        Returns:
            Memory usage information
        """
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            system_memory = psutil.virtual_memory()
            
            return {
                'process_memory_mb': memory_info.rss / (1024 * 1024),
                'process_memory_percent': process.memory_percent(),
                'system_memory_percent': system_memory.percent,
                'system_available_mb': system_memory.available / (1024 * 1024)
            }
        except Exception as e:
            return {'error': f'Unable to check memory usage: {e}'}
    
    @staticmethod
    def is_memory_available(required_mb: float) -> bool:
        """
        Check if required memory is available.
        
        Args:
            required_mb: Required memory in megabytes
            
        Returns:
            True if memory is available
        """
        try:
            available_mb = psutil.virtual_memory().available / (1024 * 1024)
            return available_mb >= required_mb
        except Exception:
            return False
    
    @staticmethod
    def get_temp_directory() -> Path:
        """
        Get system temporary directory.
        
        Returns:
            Path to temporary directory
        """
        import tempfile
        return Path(tempfile.gettempdir())
    
    @staticmethod
    def cleanup_temp_files(pattern: str = "text_analyzer_*") -> int:
        """
        Clean up temporary files matching pattern.
        
        Args:
            pattern: File pattern to match
            
        Returns:
            Number of files cleaned up
        """
        import glob
        temp_dir = SystemHelper.get_temp_directory()
        files_cleaned = 0
        
        try:
            for file_path in glob.glob(str(temp_dir / pattern)):
                Path(file_path).unlink(missing_ok=True)
                files_cleaned += 1
        except Exception as e:
            print(f"Error during cleanup: {e}")
        
        return files_cleaned


# Utility functions for common operations
def safe_divide(numerator: Union[int, float], denominator: Union[int, float], 
               default: Union[int, float] = 0) -> Union[int, float]:
    """
    Safely divide two numbers, returning default for division by zero.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value for division by zero
        
    Returns:
        Division result or default value
    """
    return numerator / denominator if denominator != 0 else default


def ensure_list(value: Any) -> List[Any]:
    """
    Ensure value is a list, wrapping single values.
    
    Args:
        value: Value to ensure as list
        
    Returns:
        List containing the value(s)
    """
    if isinstance(value, list):
        return value
    elif value is None:
        return []
    else:
        return [value]


def batch_process(items: List[Any], batch_size: int = 1000) -> List[List[Any]]:
    """
    Process items in batches for memory efficiency.
    
    Args:
        items: Items to batch
        batch_size: Size of each batch
        
    Returns:
        List of batches
    """
    batches = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i:i + batch_size])
    return batches


def log_execution_context(func_name: str, args: Tuple, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Log execution context for debugging.
    
    Args:
        func_name: Function name
        args: Function arguments
        kwargs: Function keyword arguments
        
    Returns:
        Execution context dictionary
    """
    return {
        'function': func_name,
        'args_count': len(args),
        'kwargs_keys': list(kwargs.keys()),
        'timestamp': datetime.now().isoformat(),
        'memory_usage': SystemHelper.check_memory_usage()
    }