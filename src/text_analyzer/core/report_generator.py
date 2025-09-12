"""
Strategic report generation system for Text Analyzer.
Comprehensive reporting with multiple formats and systematic presentation.
"""

import json
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from collections import Counter
from abc import ABC, abstractmethod

from config import get_reporter_logger, OutputConfig, LoggedOperation
from ..utils import (
    ReportFormattingError,
    ReportOutputError,
    FormattingHelper,
    FileSystemHelper,
    timing_decorator
)

class ReportFormatter(ABC):
    """Abstract base class for report formatters."""
    
    @abstractmethod
    def format_report(self, data: Dict[str, Any]) -> str:
        """Format report data into specific format."""
        pass
    
    @abstractmethod
    def get_file_extension(self) -> str:
        """Get file extension for this format."""
        pass

class TextReportFormatter(ReportFormatter):
    """Strategic text report formatting with clean presentation."""
    
    def __init__(self, width: int = 80):
        self.width = width
        self.separator_char = OutputConfig.SEPARATOR_CHAR
        self.sub_separator_char = OutputConfig.SUB_SEPARATOR_CHAR
    
    def format_report(self, data: Dict[str, Any]) -> str:
        """Generate formatted text report."""
        sections = []
        
        # Header section
        sections.append(self._format_header(data))
        sections.append("")
        
        # File information section
        if 'file_info' in data:
            sections.append(self._format_file_info(data['file_info']))
            sections.append("")
        
        # Analysis summary section
        if 'analysis_summary' in data:
            sections.append(self._format_analysis_summary(data['analysis_summary']))
            sections.append("")
        
        # Top words section
        if 'top_words' in data:
            sections.append(self._format_top_words(data['top_words'], data.get('total_words', 0)))
            sections.append("")
        
        # Processing statistics section
        if 'processing_stats' in data:
            sections.append(self._format_processing_stats(data['processing_stats']))
            sections.append("")
        
        # Word length distribution section
        if 'word_length_distribution' in data:
            sections.append(self._format_word_length_distribution(data['word_length_distribution']))
            sections.append("")
        
        # Footer section
        sections.append(self._format_footer())
        
        return "\n".join(sections)
    
    def _format_header(self, data: Dict[str, Any]) -> str:
        """Format report header."""
        lines = [
            FormattingHelper.create_separator_line(self.width, self.separator_char),
            FormattingHelper.center_text("TEXT ANALYSIS REPORT", self.width),
            FormattingHelper.create_separator_line(self.width, self.separator_char),
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Analyzer Version: {data.get('version', '1.0.0')}"
        ]
        return "\n".join(lines)
    
    def _format_file_info(self, file_info: Dict[str, Any]) -> str:
        """Format file information section."""
        lines = [
            "FILE INFORMATION:",
            FormattingHelper.create_separator_line(self.width // 2, self.sub_separator_char)
        ]
        
        info_items = [
            ("Source File", file_info.get('path', 'Unknown')),
            ("File Size", FormattingHelper.format_bytes(file_info.get('size', 0))),
            ("Encoding", file_info.get('encoding', 'Unknown')),
            ("Lines", FormattingHelper.format_number(file_info.get('lines', 0))),
            ("Checksum", file_info.get('checksum', 'N/A'))
        ]
        
        for label, value in info_items:
            lines.append(f"{label:<15}: {value}")
        
        return "\n".join(lines)
    
    def _format_analysis_summary(self, summary: Dict[str, Any]) -> str:
        """Format analysis summary section."""
        lines = [
            "ANALYSIS SUMMARY:",
            FormattingHelper.create_separator_line(self.width // 2, self.sub_separator_char)
        ]
        
        summary_items = [
            ("Total Words", FormattingHelper.format_number(summary.get('total_words', 0))),
            ("Unique Words", FormattingHelper.format_number(summary.get('unique_words', 0))),
            ("Vocabulary Richness", f"{summary.get('vocabulary_richness', 0):.4f}"),
            ("Average Word Length", f"{summary.get('avg_word_length', 0):.2f} characters"),
            ("Processing Time", FormattingHelper.format_duration(summary.get('processing_time', 0)))
        ]
        
        for label, value in summary_items:
            lines.append(f"{label:<20}: {value}")
        
        return "\n".join(lines)
    
    def _format_top_words(self, top_words: List[Tuple[str, int]], total_words: int) -> str:
        """Format top words section."""
        lines = [
            f"TOP {len(top_words)} MOST FREQUENT WORDS:",
            FormattingHelper.create_separator_line(self.width // 2, self.sub_separator_char)
        ]
        
        # Table header
        header_columns = ["Rank", "Word", "Count", "Percentage"]
        header_widths = [6, 20, 10, 12]
        lines.append(FormattingHelper.create_table_row(header_columns, header_widths, " | "))
        lines.append(FormattingHelper.create_separator_line(sum(header_widths) + 9, self.sub_separator_char))
        
        # Table rows
        for rank, (word, count) in enumerate(top_words, 1):
            percentage = FormattingHelper.format_percentage(count, total_words)
            row_columns = [
                str(rank),
                word,
                FormattingHelper.format_number(count),
                percentage
            ]
            lines.append(FormattingHelper.create_table_row(row_columns, header_widths, " | "))
        
        return "\n".join(lines)
    
    def _format_processing_stats(self, stats: Dict[str, Any]) -> str:
        """Format processing statistics section."""
        lines = [
            "PROCESSING STATISTICS:",
            FormattingHelper.create_separator_line(self.width // 2, self.sub_separator_char)
        ]
        
        stats_items = [
            ("Original Text Length", FormattingHelper.format_number(stats.get('original_length', 0))),
            ("Processed Text Length", FormattingHelper.format_number(stats.get('processed_length', 0))),
            ("Words per Second", FormattingHelper.format_number(int(stats.get('words_per_second', 0)))),
            ("Operations Applied", ", ".join(stats.get('operations_applied', []))),
            ("Memory Usage", f"{stats.get('memory_mb', 0):.2f} MB")
        ]
        
        for label, value in stats_items:
            lines.append(f"{label:<25}: {value}")
        
        return "\n".join(lines)
    
    def _format_word_length_distribution(self, distribution: Dict[int, int]) -> str:
        """Format word length distribution section."""
        lines = [
            "WORD LENGTH DISTRIBUTION:",
            FormattingHelper.create_separator_line(self.width // 2, self.sub_separator_char)
        ]
        
        # Sort by length and show top distributions
        sorted_dist = sorted(distribution.items())
        total_words = sum(distribution.values())
        
        for length, count in sorted_dist[:10]:  # Show top 10
            percentage = FormattingHelper.format_percentage(count, total_words)
            bar_length = int((count / max(distribution.values())) * 20) if distribution.values() else 0
            bar = "█" * bar_length
            lines.append(f"{length:2d} chars: {count:>6} {percentage:>7} {bar}")
        
        return "\n".join(lines)
    
    def _format_footer(self) -> str:
        """Format report footer."""
        return FormattingHelper.create_separator_line(self.width, self.separator_char)
    
    def get_file_extension(self) -> str:
        return ".txt"

class JSONReportFormatter(ReportFormatter):
    """JSON report formatting for structured data exchange."""
    
    def format_report(self, data: Dict[str, Any]) -> str:
        """Generate JSON formatted report."""
        # Create structured JSON report
        json_report = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "analyzer_version": data.get('version', '1.0.0'),
                "report_format": "json"
            },
            "file_information": data.get('file_info', {}),
            "analysis_summary": data.get('analysis_summary', {}),
            "top_words": [
                {"rank": i+1, "word": word, "count": count, 
                 "percentage": (count / data.get('total_words', 1)) * 100}
                for i, (word, count) in enumerate(data.get('top_words', []))
            ],
            "processing_statistics": data.get('processing_stats', {}),
            "word_length_distribution": data.get('word_length_distribution', {}),
            "character_distribution": data.get('character_distribution', {})
        }
        
        return json.dumps(json_report, indent=2, ensure_ascii=False)
    
    def get_file_extension(self) -> str:
        return ".json"

class CSVReportFormatter(ReportFormatter):
    """CSV report formatting for spreadsheet analysis."""
    
    def format_report(self, data: Dict[str, Any]) -> str:
        """Generate CSV formatted report."""
        output = []
        
        # Top words CSV section
        if 'top_words' in data:
            output.append("# Top Words")
            output.append("Rank,Word,Count,Percentage")
            
            total_words = data.get('total_words', 1)
            for rank, (word, count) in enumerate(data.get('top_words', []), 1):
                percentage = (count / total_words) * 100
                output.append(f"{rank},\"{word}\",{count},{percentage:.2f}")
            
            output.append("")
        
        # Word length distribution CSV section
        if 'word_length_distribution' in data:
            output.append("# Word Length Distribution")
            output.append("Length,Count")
            
            for length, count in sorted(data['word_length_distribution'].items()):
                output.append(f"{length},{count}")
            
            output.append("")
        
        # Summary statistics
        if 'analysis_summary' in data:
            output.append("# Summary Statistics")
            output.append("Metric,Value")
            
            summary = data['analysis_summary']
            metrics = [
                ("Total Words", summary.get('total_words', 0)),
                ("Unique Words", summary.get('unique_words', 0)),
                ("Vocabulary Richness", f"{summary.get('vocabulary_richness', 0):.4f}"),
                ("Average Word Length", f"{summary.get('avg_word_length', 0):.2f}"),
                ("Processing Time", f"{summary.get('processing_time', 0):.3f}")
            ]
            
            for metric, value in metrics:
                output.append(f"\"{metric}\",{value}")
        
        return "\n".join(output)
    
    def get_file_extension(self) -> str:
        return ".csv"

class HTMLReportFormatter(ReportFormatter):
    """HTML report formatting for web presentation."""
    
    def format_report(self, data: Dict[str, Any]) -> str:
        """Generate HTML formatted report."""
        html_template = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Text Analysis Report</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; text-align: center; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #34495e; border-left: 4px solid #3498db; padding-left: 15px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #3498db; color: white; font-weight: bold; }}
                tr:nth-child(even) {{ background-color: #f8f9fa; }}
                .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }}
                .stat-box {{ background-color: #ecf0f1; padding: 20px; border-radius: 6px; border-left: 4px solid #3498db; }}
                .stat-label {{ font-weight: bold; color: #2c3e50; }}
                .stat-value {{ font-size: 1.2em; color: #3498db; }}
                .footer {{ text-align: center; margin-top: 30px; color: #7f8c8d; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <div class="container">
            </div>
        </body>
        </html>
        """
        
        content_sections = []
        
        # Header
        content_sections.append(f"""
        <h1>Text Analysis Report</h1>
        <p style="text-align: center; color: #7f8c8d;">
            Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}
        </p>
        """)
        
        # File Information
        if 'file_info' in data:
            file_info = data['file_info']
            content_sections.append(f"""
            <h2>File Information</h2>
            <div class="stat-grid">
                <div class="stat-box">
                    <div class="stat-label">Source File</div>
                    <div class="stat-value">{file_info.get('path', 'Unknown')}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">File Size</div>
                    <div class="stat-value">{FormattingHelper.format_bytes(file_info.get('size', 0))}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Encoding</div>
                    <div class="stat-value">{file_info.get('encoding', 'Unknown')}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Lines</div>
                    <div class="stat-value">{FormattingHelper.format_number(file_info.get('lines', 0))}</div>
                </div>
            </div>
            """)
        
        # Analysis Summary
        if 'analysis_summary' in data:
            summary = data['analysis_summary']
            content_sections.append(f"""
            <h2>Analysis Summary</h2>
            <div class="stat-grid">
                <div class="stat-box">
                    <div class="stat-label">Total Words</div>
                    <div class="stat-value">{FormattingHelper.format_number(summary.get('total_words', 0))}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Unique Words</div>
                    <div class="stat-value">{FormattingHelper.format_number(summary.get('unique_words', 0))}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Vocabulary Richness</div>
                    <div class="stat-value">{summary.get('vocabulary_richness', 0):.4f}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Average Word Length</div>
                    <div class="stat-value">{summary.get('avg_word_length', 0):.2f} chars</div>
                </div>
            </div>
            """)
        
        # Top Words Table
        if 'top_words' in data:
            total_words = data.get('total_words', 1)
            top_words_html = """
            <h2>Most Frequent Words</h2>
            <table>
                <thead>
                    <tr><th>Rank</th><th>Word</th><th>Count</th><th>Percentage</th></tr>
                </thead>
                <tbody>
            """
            
            for rank, (word, count) in enumerate(data['top_words'], 1):
                percentage = FormattingHelper.format_percentage(count, total_words)
                top_words_html += f"""
                <tr>
                    <td>{rank}</td>
                    <td><strong>{word}</strong></td>
                    <td>{FormattingHelper.format_number(count)}</td>
                    <td>{percentage}</td>
                </tr>
                """
            
            top_words_html += "</tbody></table>"
            content_sections.append(top_words_html)
        
        # Footer
        content_sections.append("""
        <div class="footer">
            <p>Report generated by Text Analyzer | Strategic Analysis System</p>
        </div>
        """)
        
        return html_template.format(content="".join(content_sections))
    
    def get_file_extension(self) -> str:
        return ".html"

class ReportGenerator:
    """
    Strategic report generation system with multiple format support.
    Implements INTJ principles: systematic approach, comprehensive output, and flexibility.
    """
    
    def __init__(self):
        self.logger = get_reporter_logger()
        self.formatters = {
            'txt': TextReportFormatter(),
            'json': JSONReportFormatter(), 
            'csv': CSVReportFormatter(),
            'html': HTMLReportFormatter()
        }
        self.version = "1.0.0"
    
    @timing_decorator("report_generation")
    def generate_report(self, analysis_data: Dict[str, Any], 
                       output_format: str = 'txt',
                       output_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report in specified format.
        
        Args:
            analysis_data: Analysis results data
            output_format: Output format ('txt', 'json', 'csv', 'html')
            output_path: Optional output file path
            
        Returns:
            Dictionary with report generation results
            
        Raises:
            ReportFormattingError: If formatting fails
            ReportOutputError: If file output fails
        """
        with LoggedOperation(f"Report Generation ({output_format})", self.logger):
            try:
                # Validate format
                if output_format not in self.formatters:
                    raise ReportFormattingError(
                        output_format, 
                        f"Unsupported format. Available: {list(self.formatters.keys())}"
                    )
                
                # Prepare report data
                report_data = self._prepare_report_data(analysis_data)
                
                # Generate formatted report
                formatter = self.formatters[output_format]
                formatted_report = formatter.format_report(report_data)
                
                # Handle output
                result = {
                    'format': output_format,
                    'content': formatted_report,
                    'generated_at': datetime.now().isoformat(),
                    'success': True
                }
                
                if output_path:
                    output_file = self._save_report(formatted_report, output_path, formatter)
                    result['output_file'] = str(output_file)
                    result['file_size'] = len(formatted_report.encode('utf-8'))
                
                self.logger.info(f"Report generated successfully in {output_format} format")
                return result
                
            except Exception as e:
                self.logger.error(f"Report generation failed: {e}")
                raise ReportFormattingError(output_format, str(e))
    
    def _prepare_report_data(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare and structure data for report generation.
        
        Args:
            analysis_data: Raw analysis data
            
        Returns:
            Structured report data
        """
        # Extract core components
        word_frequencies = analysis_data.get('word_frequencies', Counter())
        file_info = analysis_data.get('file_info', {})
        processing_stats = analysis_data.get('processing_stats', {})
        
        # Calculate derived metrics
        total_words = sum(word_frequencies.values()) if word_frequencies else 0
        unique_words = len(word_frequencies)
        vocabulary_richness = unique_words / total_words if total_words > 0 else 0
        
        # Get top words
        top_words = word_frequencies.most_common(analysis_data.get('top_count', 10))
        
        # Calculate average word length
        if word_frequencies:
            total_chars = sum(len(word) * count for word, count in word_frequencies.items())
            avg_word_length = total_chars / total_words
        else:
            avg_word_length = 0
        
        # Prepare structured data
        report_data = {
            'version': self.version,
            'file_info': self._convert_file_info(file_info),
            'analysis_summary': {
                'total_words': total_words,
                'unique_words': unique_words,
                'vocabulary_richness': vocabulary_richness,
                'avg_word_length': avg_word_length,
                'processing_time': processing_stats.get('processing_time', 0)
            },
            'top_words': top_words,
            'total_words': total_words,
            'processing_stats': self._convert_processing_stats(processing_stats),
            'word_length_distribution': analysis_data.get('word_length_distribution', {}),
            'character_distribution': analysis_data.get('character_distribution', {})
        }
        
        return report_data
    
    def _convert_file_info(self, file_info: Any) -> Dict[str, Any]:
        """Convert file info to dictionary format."""
        if hasattr(file_info, '_asdict'):  # Named tuple
            return file_info._asdict()
        elif isinstance(file_info, dict):
            return file_info
        else:
            return {
                'path': str(file_info) if file_info else 'Unknown',
                'size': 0,
                'encoding': 'Unknown',
                'lines': 0,
                'checksum': 'N/A'
            }
    
    def _convert_processing_stats(self, stats: Any) -> Dict[str, Any]:
        """Convert processing stats to dictionary format."""
        if hasattr(stats, '__dict__'):  # Dataclass or object
            return stats.__dict__
        elif isinstance(stats, dict):
            return stats
        else:
            return {
                'original_length': 0,
                'processed_length': 0,
                'processing_time': 0,
                'operations_applied': [],
                'words_per_second': 0,
                'memory_mb': 0
            }
    
    def _save_report(self, content: str, output_path: Union[str, Path], 
                    formatter: ReportFormatter) -> Path:
        """
        Save report to file with proper extension and error handling.
        
        Args:
            content: Report content
            output_path: Output file path
            formatter: Report formatter (for extension)
            
        Returns:
            Final output file path
            
        Raises:
            ReportOutputError: If saving fails
        """
        try:
            output_file = Path(output_path)
            
            # Ensure proper extension
            expected_extension = formatter.get_file_extension()
            if not output_file.suffix:
                output_file = output_file.with_suffix(expected_extension)
            elif output_file.suffix != expected_extension:
                self.logger.warning(f"Extension mismatch: expected {expected_extension}, got {output_file.suffix}")
            
            # Create directory if needed
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Handle file conflicts
            if output_file.exists():
                output_file = FileSystemHelper.generate_unique_filename(
                    output_file.with_suffix(''), expected_extension
                )
                self.logger.info(f"File exists, using unique name: {output_file}")
            
            # Write content
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.logger.info(f"Report saved to: {output_file}")
            return output_file
            
        except IOError as e:
            raise ReportOutputError(str(output_path), f"File I/O error: {e}")
        except Exception as e:
            raise ReportOutputError(str(output_path), f"Unexpected error: {e}")
    
    @timing_decorator("batch_report_generation")
    def generate_multiple_reports(self, analysis_data: Dict[str, Any],
                                 formats: List[str],
                                 output_directory: Union[str, Path],
                                 base_filename: str = "analysis_report") -> Dict[str, Any]:
        """
        Generate multiple report formats efficiently.
        
        Args:
            analysis_data: Analysis results data
            formats: List of output formats
            output_directory: Output directory path
            base_filename: Base filename for reports
            
        Returns:
            Dictionary with results for each format
        """
        output_dir = FileSystemHelper.ensure_directory(output_directory)
        results = {}
        
        with LoggedOperation(f"Batch Report Generation ({len(formats)} formats)", self.logger):
            for fmt in formats:
                try:
                    # Generate timestamp for unique filenames
                    timestamp = datetime.now().strftime(OutputConfig.TIMESTAMP_FORMAT)
                    filename = f"{base_filename}_{timestamp}"
                    output_path = output_dir / filename
                    
                    # Generate report
                    result = self.generate_report(analysis_data, fmt, output_path)
                    results[fmt] = result
                    
                except Exception as e:
                    self.logger.error(f"Failed to generate {fmt} report: {e}")
                    results[fmt] = {
                        'format': fmt,
                        'success': False,
                        'error': str(e)
                    }
        
        # Summary statistics
        successful_formats = [fmt for fmt, result in results.items() if result.get('success', False)]
        
        summary = {
            'total_formats': len(formats),
            'successful_formats': len(successful_formats),
            'failed_formats': len(formats) - len(successful_formats),
            'output_directory': str(output_dir),
            'results': results
        }
        
        self.logger.info(f"Batch generation complete: {len(successful_formats)}/{len(formats)} successful")
        return summary
    
    def get_report_preview(self, analysis_data: Dict[str, Any], 
                          lines: int = 20) -> str:
        """
        Generate preview of text report for quick inspection.
        
        Args:
            analysis_data: Analysis results data
            lines: Number of lines to include in preview
            
        Returns:
            Report preview string
        """
        try:
            # Generate full text report
            full_report = self.generate_report(analysis_data, 'txt')['content']
            
            # Extract preview lines
            report_lines = full_report.split('\n')
            preview_lines = report_lines[:lines]
            
            if len(report_lines) > lines:
                preview_lines.append(f"... ({len(report_lines) - lines} more lines)")
            
            return '\n'.join(preview_lines)
            
        except Exception as e:
            return f"Preview generation failed: {e}"
    
    def validate_format_support(self, format_name: str) -> bool:
        """
        Validate if format is supported.
        
        Args:
            format_name: Format to check
            
        Returns:
            True if format is supported
        """
        return format_name.lower() in self.formatters
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported output formats.
        
        Returns:
            List of supported format names
        """
        return list(self.formatters.keys())