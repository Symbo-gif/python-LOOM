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

# LOOM Makefile
# Build automation for development, testing, and release

.PHONY: install install-dev test test-cov lint format benchmark clean docs all help

# Default target
help:
	@echo "LOOM Build Commands:"
	@echo ""
	@echo "  make install      - Install package in development mode"
	@echo "  make install-dev  - Install with development dependencies"
	@echo "  make test         - Run test suite"
	@echo "  make test-cov     - Run tests with coverage report"
	@echo "  make lint         - Run type checking and linting"
	@echo "  make format       - Format code with black"
	@echo "  make benchmark    - Run performance benchmarks"
	@echo "  make docs         - Build documentation"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make all          - Install dev + test + lint"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# Testing
test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ --cov=loom --cov-report=html --cov-report=term

# Code quality
lint:
	mypy loom
	@echo "Linting complete"

format:
	black loom tests/

# Performance
benchmark:
	python -m pytest tests/benchmark/ -v --benchmark-only

# Documentation
docs:
	cd docs && make html

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Combined targets
all: install-dev test lint
	@echo "All checks passed!"

