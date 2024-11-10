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
u3(0.6658644785179939*pi, 0.8243561911197484*pi, 0.29905273381363384*pi) q[0];
u3(0.6658644785179944*pi, 0.8243561911197478*pi, 0.299052733813633*pi) q[1];
can(0.4685450756900029*pi, 0.3183098861837905*pi, 0.16807469667757874*pi) q[0], q[1];
u3(0.10373831737166332*pi, 0.9999999999999997*pi, 0.0*pi) q[0];
u3(0.10373831737166384*pi, -0.9999999999999991*pi, 0.0*pi) q[1];
