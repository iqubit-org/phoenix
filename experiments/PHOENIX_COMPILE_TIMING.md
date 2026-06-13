# Phoenix Compilation Timing on Hamlib Benchmarks

Per-program compilation time for the `make phoenix -f Makefile-Hamlib` pipeline (`compiler=phoenix`, `with_O3=True`), measured by isolating each benchmark in its own subprocess and timing the `phoenix_pass` call (Hamiltonian construction + `compile_hamiltonian_simulation` + O3 all-to-all optimization).

- **Timeout threshold:** 180 s (3 min). Programs exceeding it are marked `timeout`.
- **Program property:** `Qubits` = Pauli-string length, `Paulis` = number of Pauli terms.
- Measurements are sequential (one program at a time) for fair per-program timing.

## Summary

| Category | Programs | Completed | Timeout | Error |
| --- | ---: | ---: | ---: | ---: |
| binaryoptimization | 15 | 15 | 0 | 0 |
| chemistry | 35 | 34 | 1 | 0 |
| condensedmatter | 35 | 34 | 1 | 0 |
| discreteoptimization | 15 | 14 | 1 | 0 |
| **Total** | **100** | **97** | **3** | **0** |

## binaryoptimization

| Program | Qubits | Paulis | Compile time (s) |
| --- | ---: | ---: | ---: |
| graph-gnp_k-2-gnp-k_2_n-4_rinst-05 | 4 | 12 | 2.80 |
| graph-gnp_k-4-gnp-k_4_n-6_rinst-15 | 6 | 45 | 3.17 |
| uf20-7-uf20-0384.cnf-8-res | 8 | 29 | 2.82 |
| graph-complete_bipart-complbipart-n-10_a-5_b-5 | 10 | 75 | 2.85 |
| graph-gnp_k-5-gnp-k_5_n-24_rinst-04 | 24 | 188 | 3.09 |
| uuf100-0-uuf100-0810.cnf-40-res | 40 | 187 | 3.88 |
| uuf100-8-uuf100-0509.cnf-40-cc | 40 | 206 | 3.98 |
| graph-gnp_k-4-gnp-k_4_n-40_rinst-07 | 40 | 804 | 3.30 |
| uf100-2-uf100-0396.cnf-46-res | 46 | 271 | 4.98 |
| uf100-2-uf100-0601.cnf-46-res | 46 | 284 | 5.34 |
| graph-gnp_k-5-gnp-k_5_n-60_rinst-19 | 60 | 1971 | 4.12 |
| uf100-4-uf100-0246.cnf-70-res | 70 | 688 | 26.73 |
| flat100-4-flat100-7.cnf-90-cc | 90 | 223 | 4.94 |
| graph-regular_reg-4-reg-4_n-90_rinst-07 | 90 | 540 | 3.37 |
| graph-gnp_k-5-gnp-k_5_n-90_rinst-02 | 90 | 2958 | 5.18 |

## chemistry

| Program | Qubits | Paulis | Compile time (s) |
| --- | ---: | ---: | ---: |
| all-vib-bh-enc_stdbinary_dvalues_d-level-4 | 2 | 9 | 31.18 |
| all-vib-c2h-mu_y_prime_enc_stdbinary_dvalues_8-8-4-4 | 3 | 12 | 2.63 |
| all-vib-o3-mu_y_prime_enc_unary_dvalues_4-4-4 | 4 | 6 | 2.60 |
| all-vib-c2h-mu_z_prime_enc_gray_dvalues_4-4-4-4 | 4 | 8 | 2.67 |
| LiH-parity-4 | 4 | 26 | 2.75 |
| all-vib-ch2-mu_y_prime_enc_gray_dvalues_16-16-16 | 4 | 32 | 2.67 |
| all-vib-fccf-mu_y_prime_enc_stdbinary_dvalues_16-16-16-16-8-8-8 | 4 | 32 | 3.21 |
| all-vib-f2-enc_gray_dvalues_d-level-16 | 4 | 98 | 2.67 |
| all-vib-bhf2-mu_y_prime_enc_stdbinary_dvalues_8-8-8-8-8-8 | 6 | 24 | 2.88 |
| Be2-JW-6 | 6 | 61 | 2.91 |
| all-vib-fccf-mu_y_prime_enc_unary_dvalues_8-8-8-8-8-8-8 | 8 | 14 | 2.58 |
| H2-JW-8 | 8 | 184 | 3.15 |
| all-vib-o3-enc_gray_dvalues_8-8-8 | 9 | 2801 | 21.05 |
| BH-JW-10 | 10 | 275 | 3.68 |
| HF-parity10 | 10 | 275 | 4.09 |
| OH-JW10 | 10 | 275 | 3.87 |
| all-vib-cyclo_propene-mu_x_prime_enc_stdbinary_dvalues_8-8-8-8-8-8-8-8-8-4-4-4-4-4-4 | 11 | 40 | 4.31 |
| all-vib-hno-mu_x_prime_enc_unary_dvalues_4-4-4 | 12 | 18 | 2.58 |
| HF-BK12 | 12 | 630 | 5.02 |
| OH-JW12 | 12 | 630 | 5.40 |
| OH-parity12 | 12 | 630 | 5.26 |
| all-vib-hco-enc_unary_dvalues_4-4-4 | 12 | 1788 | 15.07 |
| Li2-JW14 | 14 | 669 | 6.75 |
| N2-parity-14 | 14 | 669 | 8.35 |
| all-vib-fccf-enc_gray_dvalues_4-4-4-4-4-4-4 | 14 | 1035 | 7.46 |
| NH-BK-14 | 14 | 1085 | 7.67 |
| NH-JW-14 | 14 | 1085 | 8.57 |
| all-vib-fccf-mu_z_prime_enc_unary_dvalues_16-16-16-16-16-16-16 | 16 | 30 | 6.29 |
| C2-JW-18 | 18 | 1883 | 21.52 |
| N2-JW-22 | 22 | 4577 | 101.32 |
| O3-BK22 | 22 | 5465 | 145.81 |
| all-vib-hc3h2cn-mu_y_prime_enc_stdbinary_dvalues_4-4-4-4-4-4-4-4-4-4-4-4-4-4-4-4-4-4 | 24 | 48 | 2.63 |
| Na2-JW24 | 24 | 6508 | **timeout** |
| all-vib-h2co-mu_y_prime_enc_unary_dvalues_16-16-16-16-16-16 | 32 | 60 | 2.87 |
| all-vib-cyclo_propene-mu_x_prime_enc_unary_dvalues_16-16-16-16-16-16-16-16-16-16-16-16-16-16-16 | 64 | 120 | 2.76 |

## condensedmatter

| Program | Qubits | Paulis | Compile time (s) |
| --- | ---: | ---: | ---: |
| BH_D-1_d-8-bh_graph-1D-grid-pbc-qubitnodes_Lx-4_U-40_enc-gray_d-8 | 12 | 1176 | 6.14 |
| tfim-graph-1D-grid-pbc-qubitnodes_Lx-16_h-2 | 16 | 32 | 2.67 |
| tfim-graph-1D-grid-pbc-qubitnodes_Lx-26_h-6 | 26 | 52 | 2.66 |
| BH_D-1_d-4-bh_graph-1D-grid-pbc-qubitnodes_Lx-14_U-10_enc-stdbinary_d-4 | 28 | 490 | 4.84 |
| BH_D-2_d-4-bh_graph-2D-triag-nonpbc-qubitnodes_Lx-3_Ly-5_U-30_enc-gray_d-4 | 28 | 938 | 5.61 |
| BH_D-3_d-4-bh_graph-3D-grid-nonpbc-qubitnodes_Lx-2_Ly-2_Lz-2_U-20_enc-unary_d-4 | 32 | 880 | 5.41 |
| BH_D-1_d-4-bh_graph-1D-grid-nonpbc-qubitnodes_Lx-12_U-40_enc-unary_d-4 | 48 | 816 | 26.98 |
| BH_D-1_d-4-bh_graph-1D-grid-pbc-qubitnodes_Lx-24_U-20_enc-stdbinary_d-4 | 48 | 840 | 11.91 |
| BH_D-2_d-4-bh_graph-2D-triag-pbc-qubitnodes_Lx-5_Ly-5_U-30_enc-unary_d-4 | 60 | 3270 | 16.03 |
| tfim-graph-2D-grid-nonpbc-qubitnodes_Lx-5_Ly-15_h-3 | 75 | 205 | 3.01 |
| BH_D-1_d-8-bh_graph-1D-grid-pbc-qubitnodes_Lx-10_U-30_enc-unary_d-8 | 80 | 3980 | 22.05 |
| tfim-graph-2D-triag-pbc-qubitnodes_Lx-13_Ly-13_h-0.1 | 91 | 364 | 3.34 |
| BH_D-1_d-4-bh_graph-1D-grid-pbc-qubitnodes_Lx-46_U-2_enc-gray_d-4 | 92 | 1610 | 107.97 |
| BH_D-2_d-4-bh_graph-2D-grid-nonpbc-qubitnodes_Lx-7_Ly-7_U-70_enc-gray_d-4 | 98 | 2835 | 15.36 |
| FH_D-1-fh-graph-1D-grid-pbc-qubitnodes_Lx-50_U-2_enc-jw | 100 | 350 | 3.98 |
| BH_D-2_d-4-bh_graph-2D-triag-pbc-qubitnodes_Lx-10_Ly-10_U-30_enc-stdbinary_d-4 | 100 | 4950 | 18.77 |
| BH_D-1_d-4-bh_graph-1D-grid-pbc-qubitnodes_Lx-60_U-10_enc-stdbinary_d-4 | 120 | 2100 | **timeout** |
| BH_D-2_d-4-bh_graph-2D-triag-pbc-qubitnodes_Lx-3_Ly-22_U-70_enc-unary_d-4 | 132 | 7194 | 50.56 |
| BH_D-2_d-4-bh_graph-2D-triag-nonpbc-qubitnodes_Lx-11_Ly-11_U-100_enc-gray_d-4 | 156 | 6570 | 65.84 |
| FH_D-1-fh-graph-1D-grid-nonpbc-qubitnodes_Lx-90_U-2_enc-jw | 180 | 626 | 3.80 |
| tfim-graph-2D-triag-pbc-qubitnodes_Lx-19_Ly-19_h-0.5 | 190 | 760 | 4.74 |
| heis-graph-2D-grid-nonpbc-qubitnodes_Lx-5_Ly-58_h-2 | 290 | 1841 | 6.09 |
| FH_D-2-fh-graph-2D-triag-nonpbc-qubitnodes_Lx-16_Ly-16_U-4_enc-bk | 306 | 2091 | 27.95 |
| FH_D-1-fh-graph-1D-grid-pbc-qubitnodes_Lx-160_U-12_enc-parity | 320 | 1120 | 14.89 |
| tfim-graph-2D-triag-nonpbc-qubitnodes_Lx-3_Ly-160_h-0.1 | 324 | 1127 | 6.87 |
| tfim-graph-2D-grid-nonpbc-qubitnodes_Lx-19_Ly-19_h-5 | 361 | 1045 | 7.03 |
| heis-graph-2D-grid-pbc-qubitnodes_Lx-2_Ly-185_h-0.5 | 370 | 2035 | 7.22 |
| FH_D-2-fh-graph-2D-grid-pbc-qubitnodes_Lx-2_Ly-105_U-8_enc-bk | 420 | 1890 | 20.10 |
| heis-graph-2D-grid-nonpbc-qubitnodes_Lx-4_Ly-133_h-0.1 | 532 | 3313 | 13.93 |
| tfim-graph-2D-grid-pbc-qubitnodes_Lx-4_Ly-148_h-6 | 592 | 1776 | 14.32 |
| FH_D-2-fh-graph-2D-grid-pbc-qubitnodes_Lx-5_Ly-72_U-0_enc-parity | 720 | 2880 | 133.91 |
| heis-graph-2D-grid-pbc-qubitnodes_Lx-4_Ly-190_h-0 | 760 | 4560 | 21.37 |
| heis-graph-2D-triag-nonpbc-qubitnodes_Lx-40_Ly-40_h-0.1 | 861 | 8241 | 42.72 |
| heis-graph-2D-grid-nonpbc-qubitnodes_Lx-5_Ly-186_h-0.5 | 930 | 5937 | 36.25 |
| heis-graph-2D-grid-pbc-qubitnodes_Lx-5_Ly-186_h-3 | 930 | 6510 | 37.88 |

## discreteoptimization

| Program | Qubits | Paulis | Compile time (s) |
| --- | ---: | ---: | ---: |
| TSP_Ncity-4-tsp_rand-002_Ncity-4_enc-stdbinary | 8 | 48 | 3.04 |
| TSP_Ncity-8-tsp_prob-ulysses22_Ncity-8_enc-stdbinary | 24 | 448 | 4.25 |
| TSP_Ncity-5-tsp_prob-ts225_Ncity-5_enc-unary | 25 | 125 | 2.90 |
| binary-graph-regular_k-5-reg-5_n-10_rinst-07 | 30 | 1295 | 9.24 |
| TSP_Ncity-7-tsp_prob-lin105_Ncity-7_enc-unary | 49 | 343 | 3.90 |
| TSP_Ncity-8-tsp_prob-d198_Ncity-8_enc-unary | 64 | 512 | 3.80 |
| TSP_Ncity-16-tsp_prob-fl417_Ncity-16_enc-stdbinary | 64 | 3840 | 30.47 |
| TSP_Ncity-16-tsp_prob-kroD100_Ncity-16_enc-stdbinary | 64 | 3840 | 40.12 |
| binary-color02-dsjc1000.1_k-3-dsjc1000.1 | 72 | 675 | 4.75 |
| TSP_Ncity-10-tsp_prob-pr76_Ncity-10_enc-unary | 100 | 1000 | 4.92 |
| TSP_Ncity-10-tsp_prob-st70_Ncity-10_enc-unary | 100 | 1000 | 5.03 |
| unary-color02-queen13_13_k-4-queen13_13 | 112 | 412 | 3.75 |
| gray-color02-will199gpia_k-3-will199gpia | 120 | 1512 | 10.60 |
| gray-color02-1-fullins_5_k-5-1-fullins_5 | 144 | 5334 | **timeout** |
| unary-color02-ash608gpia_k-3-ash608gpia | 480 | 1614 | 12.49 |
