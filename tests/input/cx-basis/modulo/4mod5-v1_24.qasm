OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [0, 1, 2, 3, 4]
qreg q[5];

// Quantum gate operations
cx q[3], q[1];
h q[0];
cx q[4], q[0];
tdg q[0];
cx q[2], q[0];
t q[0];
cx q[4], q[0];
tdg q[0];
cx q[2], q[0];
t q[0];
t q[4];
cx q[2], q[4];
h q[0];
t q[2];
tdg q[4];
cx q[2], q[4];
cx q[1], q[4];
x q[1];
h q[4];
cx q[1], q[4];
tdg q[4];
cx q[0], q[4];
t q[4];
cx q[1], q[4];
tdg q[4];
cx q[0], q[4];
t q[4];
t q[1];
cx q[0], q[1];
h q[4];
t q[0];
tdg q[1];
cx q[0], q[1];
