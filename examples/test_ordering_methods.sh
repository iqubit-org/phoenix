#!/usr/bin/env bash
# Test all block-ordering methods exposed by phoenix.compile_hamiltonian_simulation
# on a single benchmark and print a comparison summary.
#
# Usage:
#   ./test_ordering_methods.sh [BENCHMARK_JSON] [DEVICE]
#
# Defaults:
#   BENCHMARK_JSON = ../benchmarks/uccsd_json/LiH_frz_BK_sto3g.json
#   DEVICE         = all2all
set -u

cd "$(dirname "$0")"

BENCHMARK="${1:-../benchmarks/uccsd_json/LiH_frz_BK_sto3g.json}"
DEVICE="${2:-all2all}"

# All ordering methods supported by phoenix/primitive/ordering.py::order_circuits.
# 'beam' and 'mcts' are intentionally last because they are very slow.
METHODS=(trivial greedy greedy_multistart tsp tsp_2opt mcts beam)

LOG_DIR="./_ordering_logs"
mkdir -p "$LOG_DIR"

if [[ ! -f "$BENCHMARK" ]]; then
    echo "ERROR: benchmark file not found: $BENCHMARK" >&2
    exit 1
fi

echo "============================================================"
echo "Benchmark : $BENCHMARK"
echo "Device    : $DEVICE"
echo "Methods   : ${METHODS[*]}"
echo "Logs in   : $LOG_DIR"
echo "============================================================"

# header for the summary table
printf '\n%-20s %12s %12s %12s\n' "method" "2Q gates" "2Q depth" "elapsed(s)"
printf '%-20s %12s %12s %12s\n' "--------------------" "--------" "--------" "----------"

for method in "${METHODS[@]}"; do
    log_file="$LOG_DIR/${method}.log"
    echo ">> Running order_method=$method ..." >&2

    # Run; capture both stdout and stderr to the log so we can grep the table later.
    if ! python phoenix_pass.py "$BENCHMARK" \
            --device "$DEVICE" \
            --order-method "$method" \
            > "$log_file" 2>&1; then
        printf '%-20s %12s %12s %12s\n' "$method" "FAIL" "FAIL" "-"
        continue
    fi

    # Pull the post-optimization stats from the rich table. The "Optimized circuit"
    # line contains: | num_qubits | num_gates | num_2q_gates | depth | depth_2q |
    # We read the data row that immediately follows the second header.
    stats_line=$(awk '
        /Optimized circuit/ { found=1; next }
        found && /^\|[[:space:]]*[0-9]/ { print; exit }
    ' "$log_file")

    # Extract elapsed time embedded in the title row, e.g. "..., 11.34s)"
    elapsed=$(grep -oE '[0-9]+\.[0-9]+s\)' "$log_file" | head -1 | tr -d 's)')

    if [[ -z "$stats_line" ]]; then
        printf '%-20s %12s %12s %12s\n' "$method" "??" "??" "${elapsed:--}"
        continue
    fi

    # stats_line looks like: "|     10     |    865    |     500      |  649  |   455    |"
    n2q=$( echo "$stats_line" | awk -F'|' '{gsub(/ /,"",$4); print $4}')
    d2q=$( echo "$stats_line" | awk -F'|' '{gsub(/ /,"",$6); print $6}')

    printf '%-20s %12s %12s %12s\n' "$method" "$n2q" "$d2q" "${elapsed:--}"
done

echo
echo "Done. Per-method full output saved under $LOG_DIR/."
