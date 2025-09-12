"""
CLI module initialization for Text Analyzer.
Provides centralized access to command-line interface components.
"""

from .main import CLIInterface, main
from .argument_parser import (
    ArgumentParser,
    ParsedArguments,
    create_argument_parser,
    parse_command_line
)

# Export CLI classes and functions
__all__ = [
    'CLIInterface',
    'main',
    'ArgumentParser',
    'ParsedArguments',
    'create_argument_parser',
    'parse_command_line'
]