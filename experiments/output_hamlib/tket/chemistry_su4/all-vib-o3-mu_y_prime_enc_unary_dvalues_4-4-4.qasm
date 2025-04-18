OPENQASM 2.0;
include "qelib1.inc";


// Customized 'ryy' gate definition
gate ryy(param0) q0,q1 {
    rx(pi/2) q0;
    rx(pi/2) q1;
    cx q0, q1;
    rz(param0) q1;
    cx q0, q1;
    rx(-pi/2) q0;
    rx(-pi/2) q1;
}

// Customized 'can' gate definition
gate can (param0, param1, param2) q0,q1 {
    rxx(param0) q0, q1;
    ryy(param1) q0, q1;
    rzz(param2) q0, q1;
}

// Qubits: [0, 1, 2, 3]
qreg q[4];

// Quantum gate operations
u3(1.0*pi, 0.7499999999999999*pi, -0.7499999999999999*pi) q[0];
u3(1.0*pi, 0.46874257385364343*pi, -0.46874257385364343*pi) q[3];
u3(0.0*pi, 0.02168150208571636*pi, 0.02168150208571636*pi) q[2];
u3(1.0*pi, -0.2499999999999999*pi, 0.2499999999999999*pi) q[1];
can(0.07894626189394732*pi, 0.07894626189394716*pi, 0.0*pi) q[0], q[1];
u3(1.0*pi, -0.228318497914284*pi, 0.228318497914284*pi) q[1];
can(0.11164687426907838*pi, 0.11164687426907831*pi, 0.0*pi) q[1], q[2];
u3(1.0*pi, 0.49042407593935977*pi, -0.49042407593935977*pi) q[2];
can(0.13673893666795542*pi, 0.13673893666795514*pi, 0.0*pi) q[2], q[3];
u3(1.0*pi, -0.5312574261463567*pi, 0.5312574261463567*pi) q[2];
u3(1.0*pi, -0.5312574261463567*pi, 0.5312574261463567*pi) q[3];
u3(0.0*pi, -0.02168150208571595*pi, -0.02168150208571595*pi) q[1];
u3(1.0*pi, -0.25000000000000017*pi, 0.25000000000000017*pi) q[0];
