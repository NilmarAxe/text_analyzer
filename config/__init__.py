"""
Configuration module initialization.
Provides centralized access to all configuration components.
"""

from .settings import (
    AnalysisConfig,
    OutputConfig,
    SystemConfig,
    EnvironmentConfig,
    get_settings,
    validate_configuration
)

from .logging_config import (
    LoggerSetup,
    get_analyzer_logger,
    get_file_handler_logger,
    get_processor_logger,
    get_reporter_logger,
    get_cli_logger,
    get_system_logger,
    PerformanceLogger,
    LoggedOperation,
    initialize_logging
)

# Initialize configuration system
_config_initialized = False

def initialize_config():
    """Initialize complete configuration system."""
    global _config_initialized
    
    if _config_initialized:
        return True
    
    # Validate configuration
    if not validate_configuration():
        return False
    
    # Initialize logging
    initialize_logging()
    
    _config_initialized = True
    return True

# Export configuration classes and functions
__all__ = [
    'AnalysisConfig',
    'OutputConfig', 
    'SystemConfig',
    'EnvironmentConfig',
    'get_settings',
    'validate_configuration',
    'LoggerSetup',
    'get_analyzer_logger',
    'get_file_handler_logger',
    'get_processor_logger',
    'get_reporter_logger',
    'get_cli_logger',
    'get_system_logger',
    'PerformanceLogger',
    'LoggedOperation',
    'initialize_logging',
    'initialize_config'
]