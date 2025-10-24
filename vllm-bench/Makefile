# Makefile for vllm-bench

# Define the Python interpreter and any necessary commands
PYTHON = python3
PIP = pip

# Define the virtual environment directory
VENV_DIR = venv

# Define the default target
.DEFAULT_GOAL := help

# Create a virtual environment
.PHONY: venv
venv:
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/$(PIP) install -r requirements.txt

# Run benchmarks
.PHONY: run
run: venv
	$(VENV_DIR)/bin/$(PYTHON) -m bench

# Clean up results
.PHONY: clean
clean:
	rm -rf $(VENV_DIR)
	rm -rf results/*

# Display help
.PHONY: help
help:
	@echo "Makefile commands:"
	@echo "  venv    - Create a virtual environment and install dependencies"
	@echo "  run     - Run the benchmarks"
	@echo "  clean   - Clean up the virtual environment and results"
	@echo "  help    - Display this help message"