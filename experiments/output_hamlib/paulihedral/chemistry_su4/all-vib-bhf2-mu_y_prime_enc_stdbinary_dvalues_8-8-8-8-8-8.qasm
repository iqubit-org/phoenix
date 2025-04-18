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

// Qubits: [0, 1, 2, 3, 4, 5]
qreg q[6];

// Quantum gate operations
u3(0.0*pi, -0.19755837506019736*pi, -0.19755837506019736*pi) q[0];
u3(0.0*pi, -0.19755837506019736*pi, -0.19755837506019736*pi) q[3];
u3(0.5*pi, 1.0*pi, -0.75*pi) q[5];
u3(0.5*pi, 1.0*pi, -0.75*pi) q[2];
u3(0.32492948056956616*pi, 0.9269948740900762*pi, -0.2679325600054321*pi) q[4];
can(0.4873043128971157*pi, 0.2597869142235667*pi, 0.0*pi) q[3], q[4];
u3(0.06830988618379322*pi, 0.49999999999999795*pi, -0.4463336690198571*pi) q[3];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[3], q[5];
u3(0.7499999999999993*pi, -0.5000000000000001*pi, 0.5000000000000002*pi) q[3];
u3(0.5*pi, -0.25*pi, 0.0*pi) q[5];
u3(0.6129733081768405*pi, -0.9250644252140644*pi, -0.8784593486673435*pi) q[4];
can(0.40570370600080885*pi, 0.3183098861837905*pi, 0.0*pi) q[3], q[4];
u3(1.0*pi, 0.0*pi, 0.0*pi) q[3];
u3(0.8423196945496314*pi, 0.0*pi, 0.0*pi) q[4];
u3(0.32492948056956616*pi, 0.9269948740900762*pi, -0.2679325600054321*pi) q[1];
can(0.4873043128971157*pi, 0.2597869142235667*pi, 0.0*pi) q[0], q[1];
u3(0.06830988618379322*pi, 0.49999999999999795*pi, -0.4463336690198571*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.7499999999999993*pi, -0.5000000000000001*pi, 0.5000000000000002*pi) q[0];
u3(0.5*pi, -0.25*pi, 0.0*pi) q[2];
u3(0.6129733081768405*pi, -0.9250644252140644*pi, -0.8784593486673435*pi) q[1];
can(0.40570370600080885*pi, 0.3183098861837905*pi, 0.0*pi) q[0], q[1];
u3(1.0*pi, 0.0*pi, 0.0*pi) q[0];
u3(0.8423196945496314*pi, 0.0*pi, 0.0*pi) q[1];
