# GPU Performance Comparison Tool

A tool for comparing GPU performance across different parallel computing algorithms using interactive charts.

## Features

- Compare performance data from multiple CSV files
- Interactive line charts using Plotly
- Support for parallel computing algorithms (sort, reduce, scan, etc.)
- Command-line interface for easy batch processing
- Automatic report generation

## Data Format

CSV files should contain the following columns:
- `algorithm`: Algorithm name (e.g., sort, reduce, scan, matmul, conv2d)
- `data_size`: Data size (e.g., 1K, 10K, 100K, 1M, 10M)
- `throughput`: Performance throughput in ops/sec

Example CSV content:
```csv
algorithm,data_size,throughput
sort,1K,150.5
sort,10K,420.8
reduce,1K,280.3
reduce,10K,780.9
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Create Sample Data
```bash
python perf_analyzer_plotly.py --create-sample
```

### Compare Two GPU Performance Files
```bash
python perf_analyzer_plotly.py \
  --csv-files rtx4090_performance.csv a100_performance.csv \
  --labels "RTX 4090" "A100" \
  --output gpu_comparison.html
```

### Full Command Options
```bash
python perf_analyzer_plotly.py \
  --csv-files file1.csv file2.csv file3.csv \
  --labels "GPU1" "GPU2" "GPU3" \
  --output comparison.html \
  --width 1400 \
  --height 900
```

## Output

The tool generates:
1. Interactive HTML chart showing performance comparison lines
2. Console summary report with key statistics
3. Optional static image export (PNG, SVG, PDF)

## Algorithm Types Supported

- **sort**: Parallel sorting algorithms
- **reduce**: Reduction operations
- **scan**: Prefix sum/scan operations  
- **matmul**: Matrix multiplication
- **conv2d**: 2D convolution operations

## Example Output

The generated chart shows:
- Each algorithm as a different colored line
- Different GPUs distinguished by line styles
- Interactive hover information
- Zoomable and pannable interface
- Legend for easy identification