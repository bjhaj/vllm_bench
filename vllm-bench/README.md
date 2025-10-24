# vllm-bench

vllm-bench is a benchmarking tool designed to evaluate the performance of various models and configurations. This project provides a structured approach to running benchmarks, storing results, and visualizing performance metrics.

## Project Structure

```
vllm-bench/
├── server/          # Contains server-related code and configurations
├── bench/           # Contains benchmarking logic and utilities
├── results/         # Stores results of benchmarks
│   └── plots/       # Directory for storing generated plots
├── Makefile         # Instructions for building and managing the project
├── requirements.txt # Python dependencies required for the project
├── .gitignore       # Files and directories to be ignored by Git
└── README.md        # Documentation for the project
```

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd vllm-bench
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

- To run benchmarks, use the provided Makefile:
  ```bash
  make run
  ```

- Results will be stored in the `results/` directory, and plots can be found in `results/plots/`.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or features you'd like to add.

## License

This project is licensed under the MIT License. See the LICENSE file for details.