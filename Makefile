# Makefile for vllm-bench
# Quality-of-life commands for setup, server, benchmarking, and analysis

SHELL := /bin/sh
PYTHON := python3
ENV_NAME := vllm-bench
MAMBA := mamba

# Directories
BENCH_DIR := bench
RESULTS_DIR := results
PLOTS_DIR := $(RESULTS_DIR)/plots
SERVER_DIR := server

# Files
REQUIREMENTS := requirements.txt
SCENARIOS := $(BENCH_DIR)/scenarios.yaml
SERVER_SCRIPT := $(SERVER_DIR)/run_server.sh
ENV_FILE := .env
ENV_EXAMPLE := .env.example

# Load .env file if it exists
ifneq (,$(wildcard $(ENV_FILE)))
    include $(ENV_FILE)
    export
endif

.DEFAULT_GOAL := help

# Help target - display available commands
.PHONY: help
help:
	@echo "vLLM Benchmark - Available targets:"
	@echo ""
	@echo "  make install    - Create mamba environment and install dependencies"
	@echo "  make init-env   - Create .env from .env.example (if not exists)"
	@echo "  make server     - Start vLLM server (reads from .env)"
	@echo "  make bench      - Run all benchmark scenarios from YAML"
	@echo "  make summarize  - Generate summary CSV and Markdown from results"
	@echo "  make plots      - Generate CDF plots for all run CSVs"
	@echo "  make all        - Run bench + summarize + plots"
	@echo "  make clean      - Remove results directory"
	@echo "  make clean-all  - Remove mamba environment and results"
	@echo ""
	@echo "Quick start workflow:"
	@echo "  1. make install      # Create mamba environment"
	@echo "  2. make init-env     # Create .env file"
	@echo "  3. Edit .env         # Add your HF_TOKEN and configure settings"
	@echo "  4. make server       # Start vLLM server (terminal 1)"
	@echo "  5. make bench        # Run benchmarks (terminal 2)"
	@echo "  6. make summarize    # Generate statistics"
	@echo "  7. make plots        # Create visualizations"
	@echo ""
	@echo "Environment variables (from .env or override via command line):"
	@echo "  HF_TOKEN, MODEL, PORT, HOST, GPU_MEM_UTIL, MAX_MODEL_LEN, KV_CACHE_DTYPE"
	@echo ""
	@echo "Note: You can override .env with: MODEL=other make server"

# Install dependencies
.PHONY: install
install:
	@echo "Creating mamba environment: $(ENV_NAME)..."
	$(MAMBA) create -n $(ENV_NAME) python=3.10 -y
	@echo "Installing requirements..."
	$(MAMBA) run -n $(ENV_NAME) pip install --upgrade pip
	$(MAMBA) run -n $(ENV_NAME) pip install -r $(REQUIREMENTS)
	@echo "✓ Installation complete! Activate with: mamba activate $(ENV_NAME)"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Run: make init-env"
	@echo "  2. Edit .env and add your HF_TOKEN"
	@echo "  3. Run: make server (in one terminal)"
	@echo "  4. Run: make bench (in another terminal)"

# Initialize .env from example
.PHONY: init-env
init-env:
	@if [ -f $(ENV_FILE) ]; then \
		echo "✓ .env already exists"; \
		echo "  Current configuration:"; \
		@grep -v '^#' $(ENV_FILE) | grep -v '^$$' || true; \
	else \
		cp $(ENV_EXAMPLE) $(ENV_FILE); \
		echo "✓ Created .env from .env.example"; \
		echo ""; \
		echo "IMPORTANT: Edit .env and add your HF_TOKEN"; \
		echo "  nano .env"; \
		echo "  # or"; \
		echo "  vim .env"; \
		echo ""; \
		echo "Get your token from: https://huggingface.co/settings/tokens"; \
	fi

# Start vLLM server (inherits environment variables)
.PHONY: server
server:
	@if [ ! -f $(ENV_FILE) ]; then \
		echo "⚠️  Warning: .env file not found!"; \
		echo "   Run 'make init-env' to create it from template."; \
		echo "   Continuing with environment variables from shell..."; \
		echo ""; \
	fi
	@echo "Starting vLLM server..."
	@echo "=================================================="
	@echo "Configuration:"
	@echo "  MODEL:           $${MODEL:-gpt-oss-20}"
	@echo "  HOST:            $${HOST:-0.0.0.0}"
	@echo "  PORT:            $${PORT:-8000}"
	@echo "  GPU_MEM_UTIL:    $${GPU_MEM_UTIL:-<default>}"
	@echo "  MAX_MODEL_LEN:   $${MAX_MODEL_LEN:-<auto>}"
	@echo "  KV_CACHE_DTYPE:  $${KV_CACHE_DTYPE:-<auto>}"
	@echo "  HF_TOKEN:        $${HF_TOKEN:+✓ Set}"
	@if [ -z "$$HF_TOKEN" ]; then \
		echo "  HF_TOKEN:        ✗ NOT SET (required for gpt-oss-20)"; \
	fi
	@echo "=================================================="
	@echo ""
	$(SERVER_SCRIPT)

# Run all benchmark scenarios
.PHONY: bench
bench:
	@echo "Running all benchmark scenarios..."
	$(MAMBA) run -n $(ENV_NAME) python $(BENCH_DIR)/bench.py --scenarios $(SCENARIOS) --output-dir $(RESULTS_DIR)

# Generate summary statistics
.PHONY: summarize
summarize:
	@echo "Generating summary from results..."
	$(MAMBA) run -n $(ENV_NAME) python $(BENCH_DIR)/summarize.py $(RESULTS_DIR)/*.csv --output-dir $(RESULTS_DIR)

# Generate plots for all runs
.PHONY: plots
plots:
	@echo "Generating CDF plots for all runs..."
	@mkdir -p $(PLOTS_DIR)
	@for csv in $(RESULTS_DIR)/*.csv; do \
		if [ -f "$$csv" ] && [ "$$(basename $$csv)" != "summary.csv" ]; then \
			echo "  Plotting $$csv..."; \
			$(MAMBA) run -n $(ENV_NAME) python $(BENCH_DIR)/plot.py "$$csv" --output-dir $(PLOTS_DIR); \
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

# Clean everything including mamba environment
.PHONY: clean-all
clean-all: clean
	@echo "Removing mamba environment..."
	$(MAMBA) env remove -n $(ENV_NAME) -y
	@echo "✓ Everything cleaned"

# Check if mamba environment exists
.PHONY: check-env
check-env:
	@if ! $(MAMBA) env list | grep -q "^$(ENV_NAME) "; then \
		echo "Error: Mamba environment '$(ENV_NAME)' not found. Run 'make install' first."; \
		exit 1; \
	fi