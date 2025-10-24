#!/usr/bin/env python3
"""
Wrapper script to run benchmark from project root.
Usage: python3 run_benchmark.py --name rag_realistic
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from bench.core.benchmark import main

if __name__ == "__main__":
    sys.exit(main())
