OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
u(pi/2, 0, pi/4) q[0];
cx q[0], q[1];
rz(pi/4) q[1];
