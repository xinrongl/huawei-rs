python tools/plot_logs.py "$(find ./logs -name *.log -printf '%p\n' | sort -n | tail -1)"