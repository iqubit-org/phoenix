"""
CLI to use the QAOA compiler.
"""
import argparse
import os
from compile_qaoa import compile_qaoa_qiskit as cqq

parser = argparse.ArgumentParser(allow_abbrev=True)
parser.add_argument("-device_json", help="Target Device configuration File",
type=str, default=None, action='store', dest='QCD')
parser.add_argument("-circuit_json", help="Problem QAOA ZZ Interaction File",
type=str, default=None, action='store', dest='CKT')
parser.add_argument("-config_json", help="QAOA/Compiler configuration File",
type=str, default=None, action='store', dest='CON')
parser.add_argument("-policy_compilation",
help="Chosen Compilation Policy: Instruction Parallelization-only (IP)" +
"Iterative Compilation (IterC), Incremental Compilation (IC)," +
"Variation-aware Incremental Compilation (VIC)",
type=str, default='IC', action='store', dest='POL')
parser.add_argument("-target_IterC", help="Chosen minimization objective" +
"of Iterative Compilation (IterC). Using a branch and bound optimization " +
"heuristic, IterC minimizes: Depth (D) or Two-qubit gate-count (GC_2Q)" +
"or Estimated Success Probability (ESP)",
type=str, default='GC_2Q', action='store', dest='TAR')
parser.add_argument("-output_qasm_file_name", help="Name of the output file" +
"to write compiled circuit description in QASM format",
type=str, default='QAOA.qasm', action='store', dest='OUT')

args = parser.parse_args()

if __name__ == '__main__':
    if os.path.isfile(args.QCD) and os.path.isfile(args.CKT) and os.path.isfile(args.CON):
        comp_obj = cqq.CompileQAOAQiskit(circuit_json=args.CKT, \
            qc_json=args.QCD, config_json=args.CON, out_circuit_file_name=args.OUT)
        if args.POL == 'IP':
            comp_obj.run_ip()
        elif args.POL == 'IterC':
            comp_obj.run_iter_c(target=args.TAR)
        elif args.POL == 'IC':
            comp_obj.run_incr_c()
        elif args.POL == 'VIC':
            comp_obj.run_incr_c(variation_aware=True)
    else:
        print('Please provide valid paths to the input files.')
        