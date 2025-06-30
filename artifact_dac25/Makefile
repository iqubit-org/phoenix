OUTPUT_QASM_PATH = 'output_uccsd'
BENCH_SCRIPT = 'bench_uccsd.py'
ALL_COMPILERS = tket paulihedral tetris phoenix
HIGH_LEVEL_COMPILERS := paulihedral tetris phoenix
NON_TOPO_AWARE_COMPILERS := tket phoenix
TOPOLOGY_TYPES := hhex square


.PHONY: all clean

all:
	@echo "This is the comprehensive Makefile for benchmarking Phoenix and baselines on UCCSD benchmarks."


# Phoenix compiler
phoenix:
	@echo "Testing Phoenix on UCCSD benchmarks (all2all topology)..."
	@cd experiments && ./$(BENCH_SCRIPT) -c phoenix

# Paulihedral compiler
paulihedral:
	@echo "Testing Paulihedral on UCCSD benchmarks (all2all topology)..."
	@cd experiments && ./$(BENCH_SCRIPT) -c paulihedral
paulihedral_hhex:
	@echo "Testing Paulihedral on UCCSD benchmarks (heavy-hexagonal topology)..."
	@cd experiments && ./$(BENCH_SCRIPT) -c paulihedral -d hhex
paulihedral_square:
	@echo "Testing Paulihedral on UCCSD benchmarks (square topology)..."
	@cd experiments && ./$(BENCH_SCRIPT) -c paulihedral -d square
	
# Tetris compiler
tetris:
	@echo "Testing Tetris on UCCSD benchmarks (all2all topology)..."
	@cd experiments && ./$(BENCH_SCRIPT) -c tetris
tetris_hhex:
	@echo "Testing Tetris on UCCSD benchmarks (heavy-hexagonal topology)..."
	@cd experiments && ./$(BENCH_SCRIPT) -c tetris -d hhex
tetris_square:
	@echo "Testing Tetris on UCCSD benchmarks (square topology)..."
	@cd experiments && ./$(BENCH_SCRIPT) -c tetris -d square

# TKet compiler
tket:
	@echo "Testing TKet on UCCSD benchmarks (all2all topology)..."
	@cd experiments && ./$(BENCH_SCRIPT) -c tket
	@cp experiments/$(OUTPUT_QASM_PATH)/tket/all2all/*.qasm ./experiments/$(OUTPUT_QASM_PATH)/tket/all2all_opt/

# Perform local optimization by Qiskit O3 after high-level compilation
O3:
	@echo "Performing local optimization by Qiskit O3 on all compiled circuits..."
	@parallel --bar --group 'cd experiments && ./uccsd_all2all_qiskit_opt.py -c {}' ::: $(HIGH_LEVEL_COMPILERS)

# Perform hardware-aware benchmarking since some VQA compilers are not topology-aware
topology:
	@echo "Perform hardware-aware compilation for some non-topology-aware VQA compilers..."
	@parallel --bar --group 'cd experiments && ./uccsd_all2all_to_limited.py -c {1} -d {2}' ::: $(NON_TOPO_AWARE_COMPILERS) ::: $(TOPOLOGY_TYPES)
	@echo "\n✅ All hardware-aware compilations are complete."
	
sum_result:
	@echo "Summarizing benchmarking results..."
	@parallel --bar --group 'cd experiments && python summarize_results_uccsd.py -c {}' ::: $(ALL_COMPILERS)

disp_result:
	@cd experiments && python display_results_uccsd.py

clean:
	@echo "Cleaning up .QASM files..."
	@find ./experiments/$(OUTPUT_QASM_PATH) -type f -name "*.qasm" -exec rm -f {} +
