OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [0, 1]
qreg q[2];

// Quantum gate operations
h q[0];
rz(0.02167442989333893*pi) q[0];
h q[0];
h q[0];
h q[1];
cx q[0], q[1];
rz(0.7623953032730673*pi) q[1];
cx q[0], q[1];
h q[0];
h q[1];
h q[0];
cx q[0], q[1];
rz(-0.9882517142458932*pi) q[1];
cx q[0], q[1];
h q[0];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0], q[1];
rz(0.15566357988952884*pi) q[1];
cx q[0], q[1];
h q[0];
s q[0];
h q[1];
s q[1];
rz(-0.4066327891083482*pi) q[0];
h q[1];
cx q[0], q[1];
rz(-0.3520713162301423*pi) q[1];
cx q[0], q[1];
h q[1];
cx q[0], q[1];
rz(-0.9974193077447818*pi) q[1];
cx q[0], q[1];
h q[1];
rz(0.1643044929295567*pi) q[1];
h q[1];
rz(-0.8132655782166964*pi) q[1];
