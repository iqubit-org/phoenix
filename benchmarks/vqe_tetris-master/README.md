# vqe_tetris

Install pyscf, qiskit 0.43.2, qiskit_nature, qiskit_optimization
```
pip install qiskit==0.43.2
pip install pytket
```
Run experiments
```
cd artifact_evaluation
export PYTHONPATH=../core
python3 run_all.py -test_scale=6
python3 calculate_duration.py
python3 show.py
python3 test_fidelity.py
```
