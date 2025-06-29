OUTPUT_QASM_PATH = 'output_uccsd'
BENCH_SCRIPT = 'bench_uccsd.py'

.PHONY: all clean

all:
	@echo "This is the comprehensive Makefile for benchmarking Phoenix and baselines on UCCSD benchmarks."


# Phoenix compiler
phoenix:
	@echo "Testing Phoenix on UCCSD benchmarks (all2all topology)..."
	@cd experiments && ./$(BENCH_SCRIPT) -c phoenix
phoenix_o3:
	@echo "Testing Phoenix with O3 optimization on UCCSD benchmarks (all2all topology)..."
	@cd experiments && ./$(BENCH_SCRIPT) -c phoenix --O3

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
	@cd experiments && ./uccsd_all2all_qiskit_opt.py -c phoenix
	@cd experiments && ./uccsd_all2all_qiskit_opt.py -c paulihedral
	@cd experiments && ./uccsd_all2all_qiskit_opt.py -c tetris

# Perform hardware-aware benchmarking since some VQA compilers are not topology-aware
topology:
	@echo "Perform hardware-aware compilation for some non-topology-aware VQA compilers..."
	@cd experiments && ./uccsd_all2all_to_limited.py -c phoenix -d hhex
	@cd experiments && ./uccsd_all2all_to_limited.py -c phoenix -d square
	@cd experiments && ./uccsd_all2all_to_limited.py -c tket -d hhex
	@cd experiments && ./uccsd_all2all_to_limited.py -c tket -d square

sum_result:
	@echo "Summarizing benchmarking results..."
	@cd experiments && python summarize_results_uccsd.py -c phoenix && echo "\n"
	@cd experiments && python summarize_results_uccsd.py -c paulihedral && echo "\n"
	@cd experiments && python summarize_results_uccsd.py -c tetris && echo "\n"
	@cd experiments && python summarize_results_uccsd.py -c tket && echo "\n"

disp_ressult:
	@cd experiments && python display_results_uccsd.py

clean:
	@echo "Cleaning up .QASM files..."
	@find ./$(OUTPUT_QASM_PATH) -type f -name "*.qasm" -exec rm -f {} +
