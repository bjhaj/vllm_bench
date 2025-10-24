# Makefile for vllm-bench
# Quality-of-life commands for setup, server, benchmarking, and analysis

SHELL := /bin/sh
PYTHON := python3
VENV_DIR := venv
VENV_PYTHON := $(VENV_DIR)/bin/python
VENV_PIP := $(VENV_DIR)/bin/pip

# Directories
BENCH_DIR := bench
RESULTS_DIR := results
PLOTS_DIR := $(RESULTS_DIR)/plots
SERVER_DIR := server

# Files
REQUIREMENTS := requirements.txt
SCENARIOS := $(BENCH_DIR)/scenarios.yaml
SERVER_SCRIPT := $(SERVER_DIR)/run_server.sh

.DEFAULT_GOAL := help

# Help target - display available commands
.PHONY: help
help:
	@echo "vLLM Benchmark - Available targets:"
	@echo ""
	@echo "  make install    - Create venv, upgrade pip, install dependencies"
	@echo "  make server     - Start vLLM server (use MODEL=, HF_TOKEN=, etc.)"
	@echo "  make bench      - Run all benchmark scenarios from YAML"
	@echo "  make summarize  - Generate summary CSV and Markdown from results"
	@echo "  make plots      - Generate CDF plots for all run CSVs"
	@echo "  make all        - Run bench + summarize + plots"
	@echo "  make clean      - Remove results directory"
	@echo "  make clean-all  - Remove venv and results"
	@echo ""
	@echo "Examples:"
	@echo "  make install"
	@echo "  HF_TOKEN=hf_xxx MODEL=gpt-oss-20 make server"
	@echo "  make bench"
	@echo "  make summarize plots"

# Install dependencies
.PHONY: install
install:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "Upgrading pip..."
	$(VENV_PIP) install --upgrade pip
	@echo "Installing requirements..."
	$(VENV_PIP) install -r $(REQUIREMENTS)
	@echo "✓ Installation complete! Activate with: source $(VENV_DIR)/bin/activate"

# Start vLLM server (inherits environment variables)
.PHONY: server
server:
	@echo "Starting vLLM server..."
	@echo "Environment: MODEL=$${MODEL:-gpt-oss-20}, HF_TOKEN=$${HF_TOKEN:-(not set)}"
	$(SERVER_SCRIPT)

# Run all benchmark scenarios
.PHONY: bench
bench:
	@echo "Running all benchmark scenarios..."
	$(VENV_PYTHON) $(BENCH_DIR)/bench.py --scenarios $(SCENARIOS) --output-dir $(RESULTS_DIR)

# Generate summary statistics
.PHONY: summarize
summarize:
	@echo "Generating summary from results..."
	$(VENV_PYTHON) $(BENCH_DIR)/summarize.py $(RESULTS_DIR)/*.csv --output-dir $(RESULTS_DIR)

# Generate plots for all runs
.PHONY: plots
plots:
	@echo "Generating CDF plots for all runs..."
	@mkdir -p $(PLOTS_DIR)
	@for csv in $(RESULTS_DIR)/*.csv; do \
		if [ -f "$$csv" ] && [ "$$(basename $$csv)" != "summary.csv" ]; then \
			echo "  Plotting $$csv..."; \
			$(VENV_PYTHON) $(BENCH_DIR)/plot.py "$$csv" --output-dir $(PLOTS_DIR); \
		fi; \
	done
	@echo "✓ All plots generated in $(PLOTS_DIR)"

# Run complete workflow: bench + summarize + plots
.PHONY: all
all: bench summarize plots
	@echo "✓ Complete benchmark workflow finished!"

# Clean results only
.PHONY: clean
clean:
	@echo "Cleaning results directory..."
	rm -rf $(RESULTS_DIR)/*
	@echo "✓ Results cleaned"

# Clean everything including venv
.PHONY: clean-all
clean-all: clean
	@echo "Removing virtual environment..."
	rm -rf $(VENV_DIR)
	@echo "✓ Everything cleaned"

# Check if venv exists
.PHONY: check-venv
check-venv:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Error: Virtual environment not found. Run 'make install' first."; \
		exit 1; \
	fi