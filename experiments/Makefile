OUTPUT_QASM_PATH = output_uccsd
BENCH_SCRIPT = bench_uccsd.py
ALL_COMPILERS := qiskit tket paulihedral tetris pauliopt quclear phoenix
HIGH_LEVEL_COMPILERS := tket paulihedral tetris pauliopt quclear phoenix
NON_TOPO_AWARE_COMPILERS := tket pauliopt quclear phoenix
TOPOLOGY_TYPES := hhex square


.PHONY: all clean

all:
	@echo "This is the comprehensive Makefile for benchmarking Phoenix and baselines on UCCSD benchmarks."


# Phoenix compiler
phoenix:
	@echo "Testing Phoenix on UCCSD benchmarks (all2all topology)..."
	@cd scripts && ./$(BENCH_SCRIPT) -c phoenix
phoenix_o3:
	@echo "Testing Phoenix with O3 optimization on UCCSD benchmarks (all2all topology)..."
	@cd scripts && ./$(BENCH_SCRIPT) -c phoenix --O3
phoenix_hhex:
	@echo "Testing Phoenix on UCCSD benchmarks (heavy-hexagonal topology)..."
	@cd scripts && ./uccsd_all2all_to_limited.py -c phoenix -d hhex
phoenix_square:
	@echo "Testing Phoenix on UCCSD benchmarks (square topology)..."
	@cd scripts && ./uccsd_all2all_to_limited.py -c phoenix -d square

# Phoenix++ compiler
phoenix++:
	@echo "Testing Phoenix++ on UCCSD benchmarks (all2all topology)..."
	@cd scripts && ./$(BENCH_SCRIPT) -c phoenix++
phoenix++_o3:
	@echo "Testing Phoenix++ with O3 optimization on UCCSD benchmarks (all2all topology)..."
	@cd scripts && ./$(BENCH_SCRIPT) -c phoenix++ --O3
phoenix++_hhex:
	@echo "Testing Phoenix++ on UCCSD benchmarks (heavy-hexagonal topology)..."
	@cd scripts && ./uccsd_all2all_to_limited.py -c phoenix++ -d hhex
phoenix++_square:
	@echo "Testing Phoenix++ on UCCSD benchmarks (square topology)..."
	@cd scripts && ./uccsd_all2all_to_limited.py -c phoenix++ -d square

# Paulihedral compiler
paulihedral:
	@echo "Testing Paulihedral on UCCSD benchmarks (all2all topology)..."
	@cd scripts && ./$(BENCH_SCRIPT) -c paulihedral
paulihedral_o3:
	@echo "Testing Paulihedral with O3 optimization on UCCSD benchmarks (all2all topology)..."
	@cd scripts && ./$(BENCH_SCRIPT) -c paulihedral --O3
paulihedral_hhex:
	@echo "Testing Paulihedral on UCCSD benchmarks (heavy-hexagonal topology)..."
	@cd scripts && ./$(BENCH_SCRIPT) -c paulihedral -d hhex
paulihedral_square:
	@echo "Testing Paulihedral on UCCSD benchmarks (square topology)..."
	@cd scripts && ./$(BENCH_SCRIPT) -c paulihedral -d square
	
# Tetris compiler
tetris:
	@echo "Testing Tetris on UCCSD benchmarks (all2all topology)..."
	@cd scripts && ./$(BENCH_SCRIPT) -c tetris
tetris_o3:
	@echo "Testing Tetris with O3 optimization on UCCSD benchmarks (all2all topology)..."
	@cd scripts && ./$(BENCH_SCRIPT) -c tetris --O3
tetris_hhex:
	@echo "Testing Tetris on UCCSD benchmarks (heavy-hexagonal topology)..."
	@cd scripts && ./$(BENCH_SCRIPT) -c tetris -d hhex
tetris_square:
	@echo "Testing Tetris on UCCSD benchmarks (square topology)..."
	@cd scripts && ./$(BENCH_SCRIPT) -c tetris -d square

# Qiskit compiler
qiskit:
	@echo "Testing Qiskit on UCCSD benchmarks (all2all topology)..."
	@cd scripts && ./$(BENCH_SCRIPT) -c qiskit
qiskit_o3:
	@echo "Testing Qiskit with O3 optimization on UCCSD benchmarks (all2all topology)..."
	@cd scripts && ./$(BENCH_SCRIPT) -c qiskit --O3
qiskit_hhex:
	@echo "Testing Qiskit on UCCSD benchmarks (heavy-hexagonal topology)..."
	@cd scripts && ./$(BENCH_SCRIPT) -c qiskit -d hhex
qiskit_square:
	@echo "Testing Qiskit on UCCSD benchmarks (square topology)..."
	@cd scripts && ./$(BENCH_SCRIPT) -c qiskit -d square

# TKet compiler
tket:
	@echo "Testing TKet on UCCSD benchmarks (all2all topology)..."
	@cd scripts && ./$(BENCH_SCRIPT) -c tket
tket_o3:
	@echo "Testing TKet with O3 optimization on UCCSD benchmarks (all2all topology)..."
	@cd scripts && ./$(BENCH_SCRIPT) -c tket --O3
tket_hhex:
	@echo "Testing TKet on UCCSD benchmarks (heavy-hexagonal topology)..."
	@cd scripts && ./uccsd_all2all_to_limited.py -c tket -d hhex
tket_square:
	@echo "Testing TKet on UCCSD benchmarks (square topology)..."
	@cd scripts && ./uccsd_all2all_to_limited.py -c tket -d square

# PauliOpt compiler
pauliopt:
	@echo "Testing PauliOpt on UCCSD benchmarks (all2all topology)..."
	@cd scripts && ./$(BENCH_SCRIPT) -c pauliopt
pauliopt_o3:
	@echo "Testing PauliOpt with O3 optimization on UCCSD benchmarks (all2all topology)..."
	@cd scripts && ./$(BENCH_SCRIPT) -c pauliopt --O3
pauliopt_hhex:
	@echo "Testing PauliOpt on UCCSD benchmarks (heavy-hexagonal topology)..."
	@cd scripts && ./uccsd_all2all_to_limited.py -c pauliopt -d hhex
pauliopt_square:
	@echo "Testing PauliOpt on UCCSD benchmarks (square topology)..."
	@cd scripts && ./uccsd_all2all_to_limited.py -c pauliopt -d square

# QuCLEAR compiler
quclear:
	@echo "Testing QuCLEAR on UCCSD benchmarks (all2all topology)..."
	@cd scripts && ./$(BENCH_SCRIPT) -c quclear
quclear_o3:
	@echo "Testing QuCLEAR with O3 optimization on UCCSD benchmarks (all2all topology)..."
	@cd scripts && ./$(BENCH_SCRIPT) -c quclear --O3
quclear_hhex:
	@echo "Testing QuCLEAR on UCCSD benchmarks (heavy-hexagonal topology)..."
	@cd scripts && ./uccsd_all2all_to_limited.py -c quclear -d hhex
quclear_square:
	@echo "Testing QuCLEAR on UCCSD benchmarks (square topology)..."
	@cd scripts && ./uccsd_all2all_to_limited.py -c quclear -d square

# PauliRL compiler
paulirl:
	@echo "Testing PauliRL on UCCSD benchmarks (all2all topology)...
	@cd scripts && ./$(BENCH_SCRIPT) -c paulirl
paulirl_o3:
	@echo "Testing PauliRL with O3 optimization on UCCSD benchmarks (all2all topology)..."
	@cd scripts && ./$(BENCH_SCRIPT) -c paulirl --O3
paulirl_hhex:
	@echo "Testing PauliRL on UCCSD benchmarks (heavy-hexagonal topology)..."
	@cd scripts && ./uccsd_all2all_to_limited.py -c paulirl -d hhex
paulirl_square:
	@echo "Testing PauliRL on UCCSD benchmarks (square topology)..."
	@cd scripts && ./uccsd_all2all_to_limited.py -c paulirl -d square

# Perform local optimization by Qiskit O3 after high-level compilation
O3:
	@echo "Performing local optimization by Qiskit O3 on all compiled circuits..."
	@parallel --bar --group 'cd scripts && ./uccsd_all2all_qiskit_opt.py -c {}' ::: $(HIGH_LEVEL_COMPILERS)
	@echo "\n✅ All local optimizations are complete."

# Perform hardware-aware benchmarking since some VQA compilers are not topology-aware
topology:
	@echo "Performing hardware-aware compilation for some non-topology-aware VQA compilers..."
	@parallel --bar --group 'cd scripts && ./uccsd_all2all_to_limited.py -c {1} -d {2}' ::: $(NON_TOPO_AWARE_COMPILERS) ::: $(TOPOLOGY_TYPES)
	@echo "\n✅ All hardware-aware compilations are complete."

sum_result:
	@echo "Summarizing benchmarking results..."
	@parallel --bar --group 'python summarize_results_uccsd.py -c {}' ::: $(ALL_COMPILERS)
	@echo "\n✅ All summaries finished."

disp_result:
	@python display_results_uccsd.py

clean:
	@echo "Cleaning up .QASM files..."
	@find ./$(OUTPUT_QASM_PATH)/ -type f -name "*.qasm" -exec rm -f {} +


demo:
	@echo "Demo test on LiH_frz_BK_sto3g..."
	@cd scripts && ./bench_single.py ../../benchmarks/uccsd_json/LiH_frz_BK_sto3g.json --O3
