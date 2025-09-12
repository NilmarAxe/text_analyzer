"""
Strategic integration tests for Text Analyzer system.
End-to-end testing of complete workflows and component interactions.
"""

import pytest
import tempfile
import json
from pathlib import Path
from collections import Counter

# Import components under test
import sys
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.text_analyzer import TextAnalyzer, AnalysisResult
from src.text_analyzer.core import FileHandler, TextProcessor, ReportGenerator
from src.text_analyzer.cli.main import CLIInterface


@pytest.mark.integration
class TestCompleteWorkflows:
    """Integration tests for complete analysis workflows."""
    
    def test_end_to_end_file_analysis(self, integration_environment):
        """Test complete end-to-end file analysis workflow."""
        # Use sample file from integration environment
        input_file = integration_environment['test_files']['multiline']
        output_dir = integration_environment['output_dir']
        
        # Create analyzer and run complete analysis
        analyzer = TextAnalyzer({
            'top_words_count': 15,
            'detailed_analysis': True,
            'case_sensitive': False
        })
        
        # Test each output format
        formats = ['txt', 'json', 'csv', 'html']
        results = {}
        
        for fmt in formats:
            output_file = output_dir / f"analysis_result.{fmt}"
            result = analyzer.analyze_file(input_file, output_file, fmt)
            
            assert result.success is True
            assert result.execution_time > 0
            assert output_file.exists()
            assert output_file.stat().st_size > 0
            
            results[fmt] = result
        
        # Verify consistency across formats
        base_result = results['txt']
        for fmt, result in results.items():
            assert result.analysis_metrics['word_statistics']['total_words'] == \
                   base_result.analysis_metrics['word_statistics']['total_words']
            assert result.analysis_metrics['word_statistics']['unique_words'] == \
                   base_result.analysis_metrics['word_statistics']['unique_words']
    
    def test_batch_processing_workflow(self, integration_environment):
        """Test complete batch processing workflow."""
        # Use multiple test files
        test_files = [
            integration_environment['test_files']['simple'],
            integration_environment['test_files']['multiline'],
            integration_environment['test_files']['mixed_case']
        ]
        
        output_dir = integration_environment['output_dir'] / 'batch'
        output_dir.mkdir(exist_ok=True)
        
        analyzer = TextAnalyzer({'top_words_count': 10})
        
        # Run batch analysis
        results = analyzer.analyze_multiple_files(test_files, output_dir, 'json')
        
        assert len(results) == 3
        
        successful_results = [r for r in results.values() if r.success]
        assert len(successful_results) >= 2  # At least most should succeed
        
        # Generate comparison report
        comparison = analyzer.generate_comparison_report(
            successful_results,
            output_dir / 'comparison.txt'
        )
        
        assert 'error' not in comparison
        assert comparison['summary_statistics']['total_files_analyzed'] >= 2
    
    def test_cli_integration(self, integration_environment):
        """Test CLI integration with file system operations."""
        input_file = integration_environment['test_files']['simple']
        output_file = integration_environment['output_dir'] / 'cli_output.txt'
        
        # Test CLI interface
        cli = CLIInterface()
        
        # Simulate command line arguments
        args = [
            str(input_file),
            '--output', str(output_file),
            '--format', 'txt',
            '--top-count', '10'
        ]
        
        exit_code = cli.run(args)
        
        assert exit_code == 0
        assert output_file.exists()
        
        # Verify output content
        content = output_file.read_text(encoding='utf-8')
        assert 'TEXT ANALYSIS REPORT' in content
        assert 'MOST FREQUENT WORDS' in content
    
    def test_configuration_integration(self, integration_environment):
        """Test integration with configuration files."""
        config_file = integration_environment['config_file']
        input_file = integration_environment['test_files']['multiline']
        
        # Load configuration and create analyzer
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        analyzer_config = {
            **config_data['analysis'],
            **config_data['performance']
        }
        
        analyzer = TextAnalyzer(analyzer_config)
        result = analyzer.analyze_file(input_file)
        
        assert result.success is True
        # Should respect configuration settings
        assert result.analysis_metrics['word_statistics']['total_words'] > 0
    
    def test_error_recovery_integration(self, integration_environment):
        """Test error recovery and graceful failure handling."""
        output_dir = integration_environment['output_dir'] / 'error_test'
        output_dir.mkdir(exist_ok=True)
        
        analyzer = TextAnalyzer()
        
        # Test with mix of valid and invalid files
        test_files = [
            integration_environment['test_files']['simple'],  # Valid
            Path('/nonexistent/file.txt'),                    # Invalid
            integration_environment['test_files']['empty_file'],  # Empty
            integration_environment['test_files']['multiline']   # Valid
        ]
        
        results = analyzer.analyze_multiple_files(test_files, output_dir, 'txt')
        
        assert len(results) == 4
        
        # Should have both successful and failed results
        successful = [r for r in results.values() if r.success]
        failed = [r for r in results.values() if not r.success]
        
        assert len(successful) >= 2  # Valid files should succeed
        assert len(failed) >= 2      # Invalid files should fail gracefully
        
        # Failed results should have meaningful error messages
        for result in failed:
            assert result.error_message is not None
            assert len(result.error_message) > 0
    
    def test_memory_management_integration(self, integration_environment):
        """Test memory management with large files."""
        # Create a large text file for testing
        large_content = "This is a test sentence for memory testing. " * 10000
        large_file = integration_environment['data_dir'] / 'large_memory_test.txt'
        large_file.write_text(large_content, encoding='utf-8')
        
        try:
            # Configure analyzer with memory limits
            analyzer = TextAnalyzer({
                'memory_limit_mb': 128,
                'detailed_analysis': False  # Reduce memory usage
            })
            
            result = analyzer.analyze_file(large_file)
            
            assert result.success is True
            assert result.analysis_metrics['word_statistics']['total_words'] > 50000
            
        finally:
            if large_file.exists():
                large_file.unlink()
    
    def test_unicode_handling_integration(self, integration_environment):
        """Test Unicode handling across the entire system."""
        # Create Unicode test file
        unicode_content = """
        English: The quick brown fox jumps over the lazy dog.
        French: Le renard brun et rapide saute par-dessus le chien paresseux.  
        Spanish: El zorro marrón rápido salta sobre el perro perezoso.
        German: Der schnelle braune Fuchs springt über den faulen Hund.
        Russian: Быстрая коричневая лиса прыгает через ленивую собаку.
        Chinese: 敏捷的棕色狐狸跳过懒惰的狗。
        Arabic: الثعلب البني السريع يقفز فوق الكلب الكسول.
        Hindi: तेज़ भूरी लोमड़ी आलसी कुत्ते के ऊपर से कूदती है।
        """
        
        unicode_file = integration_environment['data_dir'] / 'unicode_test.txt'
        unicode_file.write_text(unicode_content, encoding='utf-8')
        output_dir = integration_environment['output_dir'] / 'unicode'
        output_dir.mkdir(exist_ok=True)
        
        try:
            analyzer = TextAnalyzer()
            
            # Test all output formats with Unicode
            for fmt in ['txt', 'json', 'csv', 'html']:
                output_file = output_dir / f'unicode_result.{fmt}'
                result = analyzer.analyze_file(unicode_file, output_file, fmt)
                
                assert result.success is True
                assert output_file.exists()
                
                # Verify Unicode characters are preserved in output
                content = output_file.read_text(encoding='utf-8')
                assert len(content) > 0
                
                # For formats that should contain the original text
                if fmt in ['txt', 'html']:
                    # Should contain some Unicode characters
                    unicode_chars_found = any(ord(c) > 127 for c in content)
                    assert unicode_chars_found
                
        finally:
            if unicode_file.exists():
                unicode_file.unlink()
    
    def test_concurrent_analysis_safety(self, integration_environment):
        """Test safety of concurrent analysis operations."""
        import threading
        import time
        
        test_files = list(integration_environment['test_files'].values())[:3]
        results = {}
        errors = []
        
        def analyze_file(file_path, thread_id):
            try:
                analyzer = TextAnalyzer()
                result = analyzer.analyze_file(file_path)
                results[thread_id] = result
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Start multiple analysis threads
        threads = []
        for i, file_path in enumerate(test_files):
            thread = threading.Thread(
                target=analyze_file, 
                args=(file_path, i)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout
        
        # Verify results
        assert len(errors) == 0, f"Concurrent analysis errors: {errors}"
        assert len(results) == len(test_files)
        
        # All results should be successful
        for thread_id, result in results.items():
            assert result.success is True


@pytest.mark.integration
class TestComponentInteractions:
    """Integration tests for component interactions."""
    
    def test_file_handler_processor_integration(self, integration_environment):
        """Test FileHandler and TextProcessor integration."""
        input_file = integration_environment['test_files']['mixed_case']
        
        # Create components
        file_handler = FileHandler()
        text_processor = TextProcessor({'case_sensitive': False})
        
        # Read file
        content, file_info = file_handler.read_file(input_file)
        
        # Process text
        analysis = text_processor.analyze_text(content)
        
        # Verify integration
        assert isinstance(content, str)
        assert len(content) > 0
        assert analysis.processing_stats.total_words > 0
        assert len(analysis.word_frequencies) > 0
        
        # File info should be consistent
        assert file_info.size > 0
        assert file_info.encoding in file_handler.supported_encodings
    
    def test_processor_reporter_integration(self, integration_environment):
        """Test TextProcessor and ReportGenerator integration."""
        input_file = integration_environment['test_files']['multiline']
        
        # Create components
        file_handler = FileHandler()
        text_processor = TextProcessor()
        report_generator = ReportGenerator()
        
        # Process file
        content, file_info = file_handler.read_file(input_file)
        analysis = text_processor.analyze_text(content)
        
        # Generate report
        report_data = {
            'word_frequencies': analysis.word_frequencies,
            'file_info': file_info,
            'processing_stats': analysis.processing_stats,
            'word_length_distribution': analysis.word_lengths,
            'character_distribution': analysis.character_frequencies
        }
        
        result = report_generator.generate_report(report_data, 'json')
        
        assert result['success'] is True
        
        # Parse and verify JSON report
        report_json = json.loads(result['content'])
        assert 'analysis_summary' in report_json
        assert 'top_words' in report_json
        assert len(report_json['top_words']) > 0
    
    def test_full_pipeline_integration(self, integration_environment):
        """Test complete analysis pipeline integration."""
        input_file = integration_environment['test_files']['simple']
        output_file = integration_environment['output_dir'] / 'pipeline_test.html'
        
        # Create all components
        analyzer = TextAnalyzer({
            'top_words_count': 8,
            'case_sensitive': False,
            'detailed_analysis': True
        })
        
        # Run complete pipeline
        result = analyzer.analyze_file(input_file, output_file, 'html')
        
        # Verify end-to-end success
        assert result.success is True
        assert result.execution_time > 0
        assert output_file.exists()
        
        # Verify HTML output structure
        html_content = output_file.read_text(encoding='utf-8')
        assert '<!DOCTYPE html>' in html_content
        assert 'Text Analysis Report' in html_content
        assert '<table>' in html_content.lower()
        
        # Verify data consistency
        assert result.analysis_metrics['word_statistics']['total_words'] > 0
        assert len(result.word_frequencies) > 0


@pytest.mark.integration
@pytest.mark.slow
class TestSystemLimits:
    """Integration tests for system limits and boundaries."""
    
    def test_large_file_processing(self, integration_environment):
        """Test processing of large files."""
        # Create large test file
        large_content = "word " * 50000  # 50k words
        large_file = integration_environment['data_dir'] / 'large_test.txt'
        large_file.write_text(large_content, encoding='utf-8')
        
        try:
            analyzer = TextAnalyzer({
                'memory_limit_mb': 256,
                'detailed_analysis': False
            })
            
            result = analyzer.analyze_file(large_file)
            
            assert result.success is True
            assert result.analysis_metrics['word_statistics']['total_words'] >= 45000
            assert result.execution_time > 0
            
        finally:
            if large_file.exists():
                large_file.unlink()
    
    def test_many_unique_words(self, integration_environment):
        """Test handling of text with many unique words."""
        # Create text with many unique words
        unique_words = [f'unique_word_{i}' for i in range(5000)]
        unique_content = ' '.join(unique_words)
        unique_file = integration_environment['data_dir'] / 'unique_words_test.txt'
        unique_file.write_text(unique_content, encoding='utf-8')
        
        try:
            analyzer = TextAnalyzer()
            result = analyzer.analyze_file(unique_file)
            
            assert result.success is True
            stats = result.analysis_metrics['word_statistics']
            assert stats['total_words'] == 5000
            assert stats['unique_words'] == 5000
            assert stats['vocabulary_richness'] == 1.0  # All words unique
            
        finally:
            if unique_file.exists():
                unique_file.unlink()
    
    def test_high_frequency_words(self, integration_environment):
        """Test handling of text with very high frequency words."""
        # Create text with one word repeated many times
        high_freq_content = 'test ' * 10000
        high_freq_file = integration_environment['data_dir'] / 'high_freq_test.txt'
        high_freq_file.write_text(high_freq_content, encoding='utf-8')
        
        try:
            analyzer = TextAnalyzer()
            result = analyzer.analyze_file(high_freq_file)
            
            assert result.success is True
            
            # Should have one very frequent word
            most_common = result.word_frequencies.most_common(1)[0]
            assert most_common[0] == 'test'
            assert most_common[1] == 10000
            
            stats = result.analysis_metrics['word_statistics']
            assert stats['vocabulary_richness'] < 0.1  # Very low diversity
            
        finally:
            if high_freq_file.exists():
                high_freq_file.unlink()


@pytest.mark.integration
class TestErrorIntegration:
    """Integration tests for error handling across components."""
    
    def test_cascading_error_handling(self, integration_environment):
        """Test error handling propagation through the system."""
        # Test with various error conditions
        error_conditions = [
            ('nonexistent_file.txt', 'file not found'),
            (integration_environment['test_files']['empty_file'], 'empty'),
            (integration_environment['test_files']['binary'], 'encoding')
        ]
        
        analyzer = TextAnalyzer()
        
        for file_path, expected_error_type in error_conditions:
            result = analyzer.analyze_file(Path(file_path))
            
            assert result.success is False
            assert result.error_message is not None
            assert expected_error_type.replace('_', ' ') in result.error_message.lower()
    
    def test_partial_failure_recovery(self, integration_environment):
        """Test system recovery from partial failures."""
        # Create analyzer with restrictive settings that might cause issues
        analyzer = TextAnalyzer({
            'min_word_length': 20,  # Very high minimum
            'max_word_length': 25,  # Very restrictive range
            'remove_stop_words': True
        })
        
        input_file = integration_environment['test_files']['simple']
        result = analyzer.analyze_file(input_file)
        
        # Should succeed but with very few or no words
        assert result.success is True
        # Might have 0 words due to restrictive filtering
        assert result.analysis_metrics['word_statistics']['total_words'] >= 0
    
    def test_output_error_handling(self, integration_environment):
        """Test handling of output-related errors."""
        input_file = integration_environment['test_files']['simple']
        
        # Test with invalid output path
        invalid_output = Path('/root/restricted_output.txt')  # Typically no permission
        
        analyzer = TextAnalyzer()
        
        # Should handle output errors gracefully
        try:
            result = analyzer.analyze_file(input_file, invalid_output, 'txt')
            # If it succeeds, that's also OK (might have different permissions)
            if not result.success:
                assert 'permission' in result.error_message.lower() or \
                       'access' in result.error_message.lower()
        except Exception:
            # Should not raise unhandled exceptions
            pytest.fail("Unhandled exception during output error handling")


@pytest.mark.integration 
class TestDataIntegrity:
    """Integration tests for data integrity across the system."""
    
    def test_data_consistency_across_formats(self, integration_environment):
        """Test data consistency across different output formats."""
        input_file = integration_environment['test_files']['multiline']
        output_dir = integration_environment['output_dir'] / 'consistency'
        output_dir.mkdir(exist_ok=True)
        
        analyzer = TextAnalyzer({'top_words_count': 10})
        
        # Generate reports in all formats
        formats = ['txt', 'json', 'csv', 'html']
        results = {}
        
        for fmt in formats:
            output_file = output_dir / f'consistency_test.{fmt}'
            result = analyzer.analyze_file(input_file, output_file, fmt)
            results[fmt] = result
        
        # All should succeed
        for result in results.values():
            assert result.success is True
        
        # Core metrics should be consistent
        base_metrics = results['txt'].analysis_metrics['word_statistics']
        
        for fmt, result in results.items():
            metrics = result.analysis_metrics['word_statistics']
            assert metrics['total_words'] == base_metrics['total_words']
            assert metrics['unique_words'] == base_metrics['unique_words']
            assert abs(metrics['vocabulary_richness'] - base_metrics['vocabulary_richness']) < 0.001
    
    def test_encoding_preservation(self, integration_environment):
        """Test that text encoding is preserved throughout the system."""
        # Test with files in different encodings
        test_files = [
            integration_environment['test_files']['utf8'],
            integration_environment['test_files']['latin1']
        ]
        
        analyzer = TextAnalyzer()
        
        for test_file in test_files:
            result = analyzer.analyze_file(test_file)
            
            assert result.success is True
            
            # File info should record the detected encoding
            assert 'encoding' in result.file_info
            assert result.file_info['encoding'] in ['utf-8', 'latin-1', 'utf-16']
            
            # Should have successfully processed the content
            assert result.analysis_metrics['word_statistics']['total_words'] > 0
    
    def test_numerical_precision(self, integration_environment):
        """Test numerical precision in calculations."""
        input_file = integration_environment['test_files']['simple']
        
        analyzer = TextAnalyzer()
        
        # Run analysis multiple times
        results = []
        for _ in range(3):
            result = analyzer.analyze_file(input_file)
            results.append(result)
        
        # Results should be identical
        base_metrics = results[0].analysis_metrics['word_statistics']
        
        for result in results[1:]:
            metrics = result.analysis_metrics['word_statistics']
            assert metrics['total_words'] == base_metrics['total_words']
            assert metrics['unique_words'] == base_metrics['unique_words']
            # Vocabulary richness should be precisely the same
            assert metrics['vocabulary_richness'] == base_metrics['vocabulary_richness']


@pytest.mark.integration
@pytest.mark.performance
class TestPerformanceIntegration:
    """Integration tests for performance characteristics."""
    
    def test_processing_time_scaling(self, integration_environment):
        """Test that processing time scales reasonably with input size."""
        # Create files of different sizes
        sizes = [1000, 5000, 10000]  # word counts
        times = []
        
        for size in sizes:
            content = "test word analysis " * (size // 3)
            test_file = integration_environment['data_dir'] / f'size_test_{size}.txt'
            test_file.write_text(content, encoding='utf-8')
            
            try:
                analyzer = TextAnalyzer()
                result = analyzer.analyze_file(test_file)
                
                assert result.success is True
                times.append(result.execution_time)
                
            finally:
                if test_file.exists():
                    test_file.unlink()
        
        # Processing time should generally increase with size
        # (allowing for some variance)
        assert times[1] >= times[0] * 0.5  # At least 50% of linear scaling
        assert times[2] >= times[1] * 0.5  # At least 50% of linear scaling
    
    def test_memory_usage_bounds(self, integration_environment, memory_monitor):
        """Test that memory usage stays within reasonable bounds."""
        input_file = integration_environment['test_files']['large']
        
        memory_monitor.reset()
        initial_memory = memory_monitor.get_memory_usage()
        
        analyzer = TextAnalyzer({'memory_limit_mb': 128})
        result = analyzer.analyze_file(input_file)
        
        final_memory = memory_monitor.get_memory_usage()
        memory_increase = final_memory - initial_memory
        
        assert result.success is True
        assert memory_increase < 256  # Should not use more than 256MB additional
    
    def test_concurrent_performance(self, integration_environment):
        """Test performance characteristics under concurrent load."""
        import threading
        import time
        
        test_files = list(integration_environment['test_files'].values())[:3]
        results = []
        start_time = time.time()
        
        def analyze_file_timed(file_path):
            analyzer = TextAnalyzer()
            result = analyzer.analyze_file(file_path)
            results.append(result)
        
        # Start concurrent analyses
        threads = []
        for file_path in test_files:
            thread = threading.Thread(target=analyze_file_timed, args=(file_path,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # All should complete successfully
        assert len(results) == len(test_files)
        assert all(r.success for r in results)
        
        # Should complete in reasonable time (allowing for concurrency overhead)
        expected_sequential_time = sum(r.execution_time for r in results)
        assert total_time < expected_sequential_time * 1.5  # Max 50% overhead


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--integration"])