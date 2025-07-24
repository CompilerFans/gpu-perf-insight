python perf_analyzer_plotly.py --create-sample
python perf_analyzer_plotly.py --csv-files rtx_4090_performance.csv a100_performance.csv h100_performance.csv --labels "RTX 4090" "A100" "H100" --output perf_compare.html
python perf_analyzer_plotly.py --csv-files rtx_4090_performance.csv --labels "RTX 4090" --output perf_single.html

