# Copyright 2025 Michael Maillet, Damien Davison, Sacha Davison
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
LOOM Logging Infrastructure.

This module provides structured logging for LOOM modules.
Logging is disabled by default (WARNING level) to avoid performance impact.

Example Usage:
    >>> import loom
    >>> loom.logging.set_log_level('DEBUG')  # Enable detailed logs
    >>> # Now all LOOM operations will log debug information

    >>> # In a module:
    >>> from loom.logging import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.debug("Processing tensor with shape %s", tensor.shape)
"""

import logging
import sys
from typing import Optional


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a LOOM module.
    
    Creates or retrieves a logger with appropriate configuration for LOOM.
    The logger inherits from the 'loom' root logger, allowing global
    log level control.
    
    Args:
        name: Module name (e.g., 'loom.linalg', 'loom.optimize').
              Typically use __name__ to get the current module name.
    
    Returns:
        Configured logger instance for the specified module.
    
    Example:
        >>> from loom.logging import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Starting computation")
        >>> logger.debug("Processing shape: %s", (3, 4))
    """
    logger = logging.getLogger(name)
    
    # Only configure root loom logger if not already configured
    root_logger = logging.getLogger('loom')
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.WARNING)  # Default: only warnings and errors
    
    return logger


def set_log_level(level: str) -> None:
    """
    Set global LOOM log level.
    
    Controls verbosity of all LOOM logging output. By default, only
    WARNING and ERROR level messages are shown.
    
    Args:
        level: Log level as string. One of:
            - 'DEBUG': Show all messages including detailed debug info
            - 'INFO': Show informational messages and above
            - 'WARNING': Show warnings and errors only (default)
            - 'ERROR': Show only error messages
            - 'CRITICAL': Show only critical errors
    
    Example:
        >>> import loom
        >>> loom.logging.set_log_level('DEBUG')
        >>> # Now see detailed logs from all LOOM operations
        >>>
        >>> loom.logging.set_log_level('WARNING')
        >>> # Back to quiet mode (default)
    
    Note:
        Setting to DEBUG will show all internal operations which may
        produce significant output. Use INFO for moderate verbosity.
    """
    root_logger = logging.getLogger('loom')
    
    # Ensure the logger is configured
    if not root_logger.handlers:
        get_logger('loom')  # This will configure the root logger
    
    level_value = getattr(logging, level.upper(), None)
    if level_value is None:
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        raise ValueError(
            f"Invalid log level: '{level}'. Must be one of: {', '.join(valid_levels)}"
        )
    
    root_logger.setLevel(level_value)


def get_log_level() -> str:
    """
    Get current global LOOM log level.
    
    Returns:
        Current log level as a string (e.g., 'DEBUG', 'INFO', 'WARNING').
    
    Example:
        >>> import loom
        >>> loom.logging.get_log_level()
        'WARNING'
        >>> loom.logging.set_log_level('DEBUG')
        >>> loom.logging.get_log_level()
        'DEBUG'
    """
    root_logger = logging.getLogger('loom')
    
    # Ensure the logger is configured if not already
    if not root_logger.handlers:
        get_logger('loom')
    
    return logging.getLevelName(root_logger.level)


def disable_logging() -> None:
    """
    Completely disable LOOM logging.
    
    Useful when running in production environments where logging
    overhead should be minimized.
    
    Example:
        >>> import loom
        >>> loom.logging.disable_logging()
        >>> # No log output from LOOM at all
    """
    logging.getLogger('loom').setLevel(logging.CRITICAL + 1)


def enable_logging(level: str = 'WARNING') -> None:
    """
    Re-enable LOOM logging after it has been disabled.
    
    Args:
        level: Log level to set. Defaults to 'WARNING'.
    
    Example:
        >>> import loom
        >>> loom.logging.disable_logging()
        >>> # ... run without logs ...
        >>> loom.logging.enable_logging('INFO')
        >>> # Logging is back on at INFO level
    """
    set_log_level(level)


__all__ = [
    'get_logger',
    'set_log_level',
    'get_log_level',
    'disable_logging',
    'enable_logging',
]
