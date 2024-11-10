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

// Qubits: [0, 1]
qreg q[2];

// Quantum gate operations
u3(0.3957255277711788*pi, -0.9771061653275535*pi, 0.9926222694722789*pi) q[0];
u3(0.9929507464652355*pi, -0.9999999999999984*pi, -0.9999999999999984*pi) q[1];
can(0.39330901216757186*pi, 0.1556635798895295*pi, 0.010019213990780084*pi) q[0], q[1];
u3(0.24448273866629347*pi, 0.5933672108916518*pi, 0.0*pi) q[0];
u3(0.16431939372424356*pi, 0.6914352383723378*pi, -0.5040883204746388*pi) q[1];
