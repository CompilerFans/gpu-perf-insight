python perf_analyzer_plotly.py --create-sample  --sample-scale 1
python perf_analyzer_plotly.py --csv-files a100_performance.csv x500_performance.csv --labels "A100" "X500" --reference-lines 80 --output perf_compare_a100_vs_x500.html
python perf_analyzer_plotly.py --csv-files a100_performance.csv x500_optimized_performance.csv  --labels "A100" "X500opt" --reference-lines 80 --output perf_compare_a100_vs_x500opt.html
python perf_analyzer_plotly.py --csv-files x500_performance.csv x500_optimized_performance.csv --labels "X500" "X500opt" --output perf_compare_x500_vs_x500opt.html

