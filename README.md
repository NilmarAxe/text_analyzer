# Text Analyzer

A robust Python-based command-line tool for analyzing text files, computing word frequencies, statistical metrics, and generating detailed reports in multiple formats.

## Overview

Text Analyzer is designed to process text documents efficiently, providing insights into word usage, vocabulary diversity, and other linguistic metrics. It supports single-file and batch processing through a flexible CLI interface, with configurable settings for logging and analysis parameters. The core functionality focuses on reading text files, processing content, performing analysis, and outputting reports in TXT, JSON, CSV, or HTML formats.

The project structure emphasizes modularity:
- **CLI Layer**: Handles user input and program execution.
- **Core Modules**: Perform file handling, text processing, analysis, and report generation.
- **Utilities**: Provide helpers, validators, exceptions, and common functions.
- **Configuration**: Manages settings and logging.
- **Scripts**: Offer additional tools for batch processing and analysis runs.
- **Tests**: Ensure reliability through unit and integration testing.
- **Examples**: Include sample configuration files.

## Features

- **File Handling**: Reads text files with automatic encoding detection, size calculation, and line counting.
- **Text Processing**: Cleans and prepares text for analysis (e.g., tokenization, normalization).
- **Analysis Capabilities**:
  - Word frequency counting using efficient data structures.
  - Calculation of total words, unique words, vocabulary richness (unique/total ratio), and average word length.
  - Extraction of top N most frequent words with percentages.
- **Report Generation**: Supports multiple output formats:
  - TXT: Plain text summaries.
  - JSON: Structured data for programmatic use.
  - CSV: Tabular data for spreadsheets.
  - HTML: Formatted reports with tables and sections.
- **Configuration**: Customizable via JSON files for analysis parameters (e.g., top word count).
- **Logging**: Integrated logging with configurable levels.
- **Batch Processing**: Scripts for analyzing multiple files.
- **Performance Tracking**: Timed operations for key processes.
- **Error Handling**: Custom exceptions for formatting and output issues.
- **Testing**: Comprehensive test suite covering individual modules and integrations.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/NilmarAxe/text_analyzer.git
   cd text_analyzer
   ```

2. Install dependencies from `requirements.txt`:
   ```
   pip install -r requirements.txt
   ```

3. (Optional) Install the package locally for development:
   ```
   python setup.py install
   ```

Note: The project uses standard Python libraries like `collections`, `datetime`, `pathlib`, and `typing`, along with any additional dependencies listed in `requirements.txt` (e.g., for logging or advanced features).

## Usage

The primary interface is the CLI, executed via `src/text_analyzer/cli/main.py`.

### Basic Command
```
python src/text_analyzer/cli/main.py --input <file_path> --output <output_path> --format <txt|json|csv|html>
```

### Arguments
- `--input`: Path to the input text file (required).
- `--output`: Path for the output report (optional; defaults to stdout for some formats).
- `--format`: Report format (txt, json, csv, html; default: txt).
- `--config`: Path to custom configuration JSON (optional; see examples).
- `--top`: Number of top words to include (default: 10).
- `--verbose`: Enable detailed logging.

For full options, run:
```
python src/text_analyzer/cli/main.py --help
```

### Script Usage
- **Run Analysis**: Single-file analysis script.
  ```
  python scripts/run_analysis.py --input example.txt --output report.html --format html
  ```
- **Batch Process**: Analyze multiple files.
  ```
  python scripts/batch_process.py --input-dir ./texts --output-dir ./reports --format json
  ```

## Configuration

Settings are managed in `config/settings.py` and `config/logging_config.py`. Use `examples/sample_config.json` as a template for custom analysis parameters, such as:
```json
{
  "top_count": 20,
  "exclude_words": ["the", "a", "an"],
  "min_word_length": 3
}
```

Load custom configs via the `--config` flag.

## Examples

Analyze a sample text file and generate an HTML report:
```
python src/text_analyzer/cli/main.py --input examples/sample.txt --output report.html --format html
```

The report will include sections for file information, analysis summary, and a table of most frequent words.

## Development

- **Structure**:
  - `src/text_analyzer/cli/`: Command-line interface (argument parsing, main execution).
  - `src/text_analyzer/core/`: Core logic (analyzer, file handler, report generator, text processor).
  - `src/text_analyzer/utils/`: Supporting tools (exceptions, helpers, validators).
  - `config/`: Configuration files.
  - `examples/`: Sample inputs and configs.
  - `scripts/`: Utility scripts.
  - `tests/`: Pytest-based tests.

- **Running Tests**:
  ```
  pytest tests/
  ```

- **Logging**: Configured in `config/logging_config.py` for structured output.

## License

This project is licensed under the MIT License. See the LICENSE file for details (if not present, assume standard open-source terms).
