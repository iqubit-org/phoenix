OPENQASM 2.0;
include "qelib1.inc";
gate cxz q0,q1 { h q0; h q1; cx q0,q1; h q0; h q1; }
gate czz q0,q1 { h q1; cx q0,q1; h q1; }
gate cyy q0,q1 { sdg q0; h q0; sdg q1; cx q0,q1; h q0; s q0; s q1; }
gate czy q0,q1 { sdg q1; cx q0,q1; s q1; }
gate cxx q0,q1 { h q0; cx q0,q1; h q0; }
qreg q[10];
x q[0];
x q[5];
cxz q[1],q[2];
cxz q[1],q[3];
cxz q[1],q[6];
cxz q[1],q[7];
cxz q[1],q[8];
czz q[0],q[1];
cxz q[0],q[4];
cxz q[4],q[5];
cyy q[5],q[9];
czy q[0],q[5];
czy q[4],q[5];
czy q[0],q[5];
cxx q[0],q[9];
cxz q[0],q[4];
rz(-0.024999999999999998) q[0];
u3(0.024999999999999994,pi/2,-pi/2) q[5];
h q[9];
cx q[4],q[9];
rz(0.024999999999999998) q[9];
cx q[4],q[9];
cxz q[0],q[4];
rz(0.024999999999999998) q[0];
h q[9];
cxx q[0],q[9];
czy q[0],q[5];
u3(0.024999999999999994,-pi/2,pi/2) q[5];
czy q[4],q[5];
u3(0.024999999999999994,pi/2,-pi/2) q[5];
czy q[0],q[5];
u3(0.024999999999999994,-pi/2,pi/2) q[5];
u3(0.024999999999999994,pi/2,-pi/2) q[9];
cyy q[5],q[9];
cxz q[4],q[5];
cxz q[0],q[4];
czz q[0],q[1];
cxz q[1],q[8];
cxz q[1],q[7];
cxz q[1],q[6];
cxz q[1],q[3];
u3(pi/2,pi/2,2.111215827065471) q[5];
u3(1.4696906269787315,pi/2,-2.601173153319218) q[6];
cx q[5],q[6];
u3(3.041592653589794,0,pi/2) q[5];
u3(1.5702946546377712,1.5607196157102798,-1.6702839605185331) q[6];
cx q[5],q[6];
u3(pi/2,-2.1112158270654713,-pi) q[5];
u3(0.1011056998161652,2.601173153319218,-pi/2) q[6];
u3(pi/2,pi/2,2.111215827065471) q[0];
u3(1.4696906269787315,pi/2,-2.601173153319218) q[1];
cx q[0],q[1];
u3(3.041592653589794,0,pi/2) q[0];
u3(1.5702946546377712,1.5607196157102798,-1.6702839605185331) q[1];
cx q[0],q[1];
u3(pi/2,-2.1112158270654713,-pi) q[0];
u3(0.1011056998161652,2.601173153319218,-pi/2) q[1];
czz q[5],q[6];
u3(pi/2,pi/2,1.1830261549195793) q[5];
u3(1.4696906269787187,pi/2,2.7538224817144723) q[7];
cx q[5],q[7];
u3(3.0415926535897935,0,pi/2) q[5];
u3(1.5702946546377712,1.5607196157102798,-1.6702839605185333) q[7];
cx q[5],q[7];
u3(pi/2,-1.1830261549195793,-pi) q[5];
czz q[5],q[6];
u3(0.10110569981617738,-2.753822481714459,-pi/2) q[7];
czz q[0],q[1];
u3(pi/2,pi/2,1.1830261549195793) q[0];
u3(1.4696906269787187,pi/2,2.7538224817144723) q[3];
cx q[0],q[3];
u3(3.0415926535897935,0,pi/2) q[0];
u3(1.5702946546377712,1.5607196157102798,-1.6702839605185333) q[3];
cx q[0],q[3];
u3(pi/2,-1.1830261549195793,-pi) q[0];
czz q[0],q[1];
u3(0.10110569981617738,-2.753822481714459,-pi/2) q[3];
cxz q[1],q[3];
czz q[0],q[1];
u3(pi/2,pi/2,1.1830261549195793) q[0];
u3(1.4696906269787187,pi/2,2.7538224817144723) q[4];
cx q[0],q[4];
u3(3.0415926535897935,0,pi/2) q[0];
u3(1.5702946546377712,1.5607196157102798,-1.6702839605185333) q[4];
cx q[0],q[4];
u3(pi/2,-1.1830261549195793,-pi) q[0];
u3(0.10110569981617738,-2.753822481714459,-pi/2) q[4];
cxz q[0],q[4];
cxz q[4],q[5];
cyy q[5],q[6];
czy q[0],q[5];
czy q[4],q[5];
czy q[0],q[5];
cxx q[0],q[6];
cxz q[0],q[4];
rz(-0.024999999999999998) q[0];
u3(0.024999999999999994,pi/2,-pi/2) q[5];
h q[6];
cx q[4],q[6];
rz(0.024999999999999998) q[6];
cx q[4],q[6];
cxz q[0],q[4];
rz(0.024999999999999998) q[0];
h q[6];
cxx q[0],q[6];
czy q[0],q[5];
u3(0.024999999999999994,-pi/2,pi/2) q[5];
czy q[4],q[5];
u3(0.024999999999999994,pi/2,-pi/2) q[5];
czy q[0],q[5];
u3(0.024999999999999994,-pi/2,pi/2) q[5];
u3(0.024999999999999994,pi/2,-pi/2) q[6];
cyy q[5],q[6];
cxz q[4],q[5];
cxz q[0],q[4];
czz q[0],q[1];
cxz q[1],q[3];
czz q[0],q[1];
cxz q[0],q[3];
cxz q[3],q[5];
cyy q[5],q[6];
czy q[0],q[5];
czy q[3],q[5];
czy q[0],q[5];
cxx q[0],q[6];
cxz q[0],q[3];
rz(-0.024999999999999998) q[0];
u3(0.024999999999999994,pi/2,-pi/2) q[5];
h q[6];
cx q[3],q[6];
rz(0.024999999999999998) q[6];
cx q[3],q[6];
cxz q[0],q[3];
rz(0.024999999999999998) q[0];
h q[6];
cxx q[0],q[6];
czy q[0],q[5];
u3(0.024999999999999994,-pi/2,pi/2) q[5];
czy q[3],q[5];
u3(0.024999999999999994,pi/2,-pi/2) q[5];
czy q[0],q[5];
u3(0.024999999999999994,-pi/2,pi/2) q[5];
u3(0.024999999999999994,pi/2,-pi/2) q[6];
cyy q[5],q[6];
cxz q[3],q[5];
cxz q[0],q[3];
czz q[0],q[1];
cxz q[1],q[6];
czz q[0],q[1];
cxz q[0],q[3];
cxz q[3],q[5];
cyy q[5],q[7];
czy q[0],q[5];
czy q[3],q[5];
czy q[0],q[5];
cxx q[0],q[7];
cxz q[0],q[3];
rz(-0.024999999999999998) q[0];
u3(0.024999999999999994,pi/2,-pi/2) q[5];
h q[7];
cx q[3],q[7];
rz(0.024999999999999998) q[7];
cx q[3],q[7];
cxz q[0],q[3];
rz(0.024999999999999998) q[0];
h q[7];
cxx q[0],q[7];
czy q[0],q[5];
u3(0.024999999999999994,-pi/2,pi/2) q[5];
czy q[3],q[5];
u3(0.024999999999999994,pi/2,-pi/2) q[5];
czy q[0],q[5];
u3(0.024999999999999994,-pi/2,pi/2) q[5];
u3(0.024999999999999994,pi/2,-pi/2) q[7];
cyy q[5],q[7];
cxz q[3],q[5];
cxz q[0],q[3];
czz q[0],q[1];
cxz q[1],q[7];
czz q[0],q[1];
cxz q[0],q[3];
cxz q[3],q[5];
cyy q[5],q[8];
czy q[0],q[5];
czy q[3],q[5];
czy q[0],q[5];
cxx q[0],q[8];
cxz q[0],q[3];
rz(-0.024999999999999998) q[0];
u3(0.024999999999999994,pi/2,-pi/2) q[5];
h q[8];
cx q[3],q[8];
rz(0.024999999999999998) q[8];
cx q[3],q[8];
cxz q[0],q[3];
rz(0.024999999999999998) q[0];
h q[8];
cxx q[0],q[8];
czy q[0],q[5];
u3(0.024999999999999994,-pi/2,pi/2) q[5];
czy q[3],q[5];
u3(0.024999999999999994,pi/2,-pi/2) q[5];
czy q[0],q[5];
u3(0.024999999999999994,-pi/2,pi/2) q[5];
u3(0.024999999999999994,pi/2,-pi/2) q[8];
cyy q[5],q[8];
cxz q[3],q[5];
cxz q[0],q[3];
czz q[0],q[1];
cxz q[1],q[8];
czz q[0],q[1];
cxz q[0],q[3];
cxz q[3],q[5];
cyy q[5],q[9];
czy q[0],q[5];
czy q[3],q[5];
czy q[0],q[5];
cxx q[0],q[9];
cxz q[0],q[3];
rz(-0.024999999999999998) q[0];
u3(0.024999999999999994,pi/2,-pi/2) q[5];
h q[9];
cx q[3],q[9];
rz(0.024999999999999998) q[9];
cx q[3],q[9];
cxz q[0],q[3];
rz(0.024999999999999998) q[0];
h q[9];
cxx q[0],q[9];
czy q[0],q[5];
u3(0.024999999999999994,-pi/2,pi/2) q[5];
czy q[3],q[5];
u3(0.024999999999999994,pi/2,-pi/2) q[5];
czy q[0],q[5];
u3(0.024999999999999994,-pi/2,pi/2) q[5];
u3(0.024999999999999994,pi/2,-pi/2) q[9];
cyy q[5],q[9];
cxz q[3],q[5];
cxz q[0],q[3];
czz q[0],q[1];
cxz q[1],q[8];
cxz q[1],q[7];
cxz q[1],q[6];
cxz q[1],q[2];
cxz q[6],q[7];
czz q[5],q[6];
u3(pi/2,pi/2,1.1830261549195793) q[5];
u3(1.4696906269787187,pi/2,2.7538224817144723) q[8];
cx q[5],q[8];
u3(3.0415926535897935,0,pi/2) q[5];
u3(1.5702946546377712,1.5607196157102798,-1.6702839605185333) q[8];
cx q[5],q[8];
u3(pi/2,-1.1830261549195793,-pi) q[5];
czz q[5],q[6];
u3(0.10110569981617738,-2.753822481714459,-pi/2) q[8];
cxz q[6],q[8];
czz q[5],q[6];
u3(pi/2,pi/2,1.1830261549195793) q[5];
u3(1.4696906269787187,pi/2,2.7538224817144723) q[9];
cx q[5],q[9];
u3(3.0415926535897935,0,pi/2) q[5];
u3(1.5702946546377712,1.5607196157102798,-1.6702839605185333) q[9];
cx q[5],q[9];
u3(pi/2,-1.1830261549195793,-pi) q[5];
czz q[5],q[6];
cxz q[6],q[8];
cxz q[6],q[7];
u3(0.10110569981617738,-2.753822481714459,-pi/2) q[9];
czz q[0],q[1];
u3(pi/2,pi/2,1.1830261549195793) q[0];
u3(1.4696906269787187,pi/2,2.7538224817144723) q[2];
cx q[0],q[2];
u3(3.0415926535897935,0,pi/2) q[0];
u3(1.5702946546377712,1.5607196157102798,-1.6702839605185333) q[2];
cx q[0],q[2];
u3(pi/2,-1.1830261549195793,-pi) q[0];
u3(0.10110569981617738,-2.753822481714459,-pi/2) q[2];
cxz q[0],q[2];
cxz q[2],q[5];
cyy q[5],q[6];
czy q[0],q[5];
czy q[2],q[5];
czy q[0],q[5];
cxx q[0],q[6];
cxz q[0],q[2];
rz(-0.024999999999999998) q[0];
u3(0.024999999999999994,pi/2,-pi/2) q[5];
h q[6];
cx q[2],q[6];
rz(0.024999999999999998) q[6];
cx q[2],q[6];
cxz q[0],q[2];
rz(0.024999999999999998) q[0];
h q[6];
cxx q[0],q[6];
czy q[0],q[5];
u3(0.024999999999999994,-pi/2,pi/2) q[5];
czy q[2],q[5];
u3(0.024999999999999994,pi/2,-pi/2) q[5];
czy q[0],q[5];
u3(0.024999999999999994,-pi/2,pi/2) q[5];
u3(0.024999999999999994,pi/2,-pi/2) q[6];
cyy q[5],q[6];
cxz q[2],q[5];
cxz q[0],q[2];
czz q[0],q[1];
cxz q[6],q[7];
czz q[0],q[6];
cxz q[0],q[1];
cxz q[1],q[5];
cyy q[5],q[8];
czy q[0],q[5];
czy q[1],q[5];
czy q[0],q[5];
cxx q[0],q[8];
cxz q[0],q[1];
rz(-0.024999999999999998) q[0];
u3(0.024999999999999994,pi/2,-pi/2) q[5];
h q[8];
cx q[1],q[8];
rz(0.024999999999999998) q[8];
cx q[1],q[8];
cxz q[0],q[1];
rz(0.024999999999999998) q[0];
h q[8];
cxx q[0],q[8];
czy q[0],q[5];
u3(0.024999999999999994,-pi/2,pi/2) q[5];
czy q[1],q[5];
u3(0.024999999999999994,pi/2,-pi/2) q[5];
czy q[0],q[5];
u3(0.024999999999999994,-pi/2,pi/2) q[5];
u3(0.024999999999999994,pi/2,-pi/2) q[8];
cyy q[5],q[8];
cxz q[1],q[5];
cxz q[0],q[1];
czz q[0],q[6];
cxz q[6],q[8];
czz q[0],q[6];
cxz q[0],q[1];
cxz q[1],q[5];
cyy q[5],q[9];
czy q[0],q[5];
czy q[1],q[5];
czy q[0],q[5];
cxx q[0],q[9];
cxz q[0],q[1];
rz(-0.024999999999999998) q[0];
u3(0.024999999999999994,pi/2,-pi/2) q[5];
h q[9];
cx q[1],q[9];
rz(0.024999999999999998) q[9];
cx q[1],q[9];
cxz q[0],q[1];
rz(0.024999999999999998) q[0];
h q[9];
cxx q[0],q[9];
czy q[0],q[5];
u3(0.024999999999999994,-pi/2,pi/2) q[5];
czy q[1],q[5];
u3(0.024999999999999994,pi/2,-pi/2) q[5];
czy q[0],q[5];
u3(0.024999999999999994,-pi/2,pi/2) q[5];
u3(0.024999999999999994,pi/2,-pi/2) q[9];
cyy q[5],q[9];
cxz q[1],q[5];
cxz q[0],q[1];
czz q[0],q[6];
cxz q[6],q[8];
cxz q[6],q[7];
cxz q[0],q[1];
cxz q[1],q[5];
cyy q[5],q[6];
czy q[0],q[5];
czy q[1],q[5];
czy q[0],q[5];
cxx q[0],q[6];
cxz q[0],q[1];
rz(-0.024999999999999998) q[0];
u3(0.024999999999999994,pi/2,-pi/2) q[5];
h q[6];
cx q[1],q[6];
rz(0.024999999999999998) q[6];
cx q[1],q[6];
cxz q[0],q[1];
rz(0.024999999999999998) q[0];
h q[6];
cxx q[0],q[6];
czy q[0],q[5];
u3(0.024999999999999994,-pi/2,pi/2) q[5];
czy q[1],q[5];
u3(0.024999999999999994,pi/2,-pi/2) q[5];
czy q[0],q[5];
u3(0.024999999999999994,-pi/2,pi/2) q[5];
u3(0.024999999999999994,pi/2,-pi/2) q[6];
cyy q[5],q[6];
cxz q[1],q[5];
cxz q[0],q[1];
czz q[0],q[6];
cxz q[0],q[1];
cxz q[1],q[5];
cyy q[5],q[7];
czy q[0],q[5];
czy q[1],q[5];
czy q[0],q[5];
cxx q[0],q[7];
cxz q[0],q[1];
rz(-0.024999999999999998) q[0];
u3(0.024999999999999994,pi/2,-pi/2) q[5];
h q[7];
cx q[1],q[7];
rz(0.024999999999999998) q[7];
cx q[1],q[7];
cxz q[0],q[1];
rz(0.024999999999999998) q[0];
h q[7];
cxx q[0],q[7];
czy q[0],q[5];
u3(0.024999999999999994,-pi/2,pi/2) q[5];
czy q[1],q[5];
u3(0.024999999999999994,pi/2,-pi/2) q[5];
czy q[0],q[5];
u3(0.024999999999999994,-pi/2,pi/2) q[5];
u3(0.024999999999999994,pi/2,-pi/2) q[7];
cyy q[5],q[7];
cxz q[1],q[5];
cxz q[0],q[1];
czz q[0],q[6];
cxz q[1],q[2];
cxz q[1],q[3];
cxz q[1],q[6];
czz q[0],q[1];
cxz q[0],q[4];
cxz q[4],q[5];
cyy q[5],q[7];
czy q[0],q[5];
czy q[4],q[5];
czy q[0],q[5];
cxx q[0],q[7];
cxz q[0],q[4];
rz(-0.024999999999999998) q[0];
u3(0.024999999999999994,pi/2,-pi/2) q[5];
h q[7];
cx q[4],q[7];
rz(0.024999999999999998) q[7];
cx q[4],q[7];
cxz q[0],q[4];
rz(0.024999999999999998) q[0];
h q[7];
cxx q[0],q[7];
czy q[0],q[5];
u3(0.024999999999999994,-pi/2,pi/2) q[5];
czy q[4],q[5];
u3(0.024999999999999994,pi/2,-pi/2) q[5];
czy q[0],q[5];
u3(0.024999999999999994,-pi/2,pi/2) q[5];
u3(0.024999999999999994,pi/2,-pi/2) q[7];
cyy q[5],q[7];
cxz q[4],q[5];
cxz q[0],q[4];
czz q[0],q[1];
cxz q[1],q[7];
czz q[0],q[1];
cxz q[0],q[4];
cxz q[4],q[5];
cyy q[5],q[8];
czy q[0],q[5];
czy q[4],q[5];
czy q[0],q[5];
cxx q[0],q[8];
cxz q[0],q[4];
rz(-0.024999999999999998) q[0];
u3(0.024999999999999994,pi/2,-pi/2) q[5];
h q[8];
cx q[4],q[8];
rz(0.024999999999999998) q[8];
cx q[4],q[8];
cxz q[0],q[4];
rz(0.024999999999999998) q[0];
h q[8];
cxx q[0],q[8];
czy q[0],q[5];
u3(0.024999999999999994,-pi/2,pi/2) q[5];
czy q[4],q[5];
u3(0.024999999999999994,pi/2,-pi/2) q[5];
czy q[0],q[5];
u3(0.024999999999999994,-pi/2,pi/2) q[5];
u3(0.024999999999999994,pi/2,-pi/2) q[8];
cyy q[5],q[8];
cxz q[4],q[5];
cxz q[0],q[4];
czz q[0],q[1];
cxz q[1],q[7];
cxz q[1],q[6];
cxz q[1],q[3];
cxz q[1],q[2];
cxz q[1],q[6];
czz q[0],q[1];
cxz q[0],q[2];
cxz q[2],q[5];
cyy q[5],q[7];
czy q[0],q[5];
czy q[2],q[5];
czy q[0],q[5];
cxx q[0],q[7];
cxz q[0],q[2];
rz(-0.024999999999999998) q[0];
u3(0.024999999999999994,pi/2,-pi/2) q[5];
h q[7];
cx q[2],q[7];
rz(0.024999999999999998) q[7];
cx q[2],q[7];
cxz q[0],q[2];
rz(0.024999999999999998) q[0];
h q[7];
cxx q[0],q[7];
czy q[0],q[5];
u3(0.024999999999999994,-pi/2,pi/2) q[5];
czy q[2],q[5];
u3(0.024999999999999994,pi/2,-pi/2) q[5];
czy q[0],q[5];
u3(0.024999999999999994,-pi/2,pi/2) q[5];
u3(0.024999999999999994,pi/2,-pi/2) q[7];
cyy q[5],q[7];
cxz q[2],q[5];
cxz q[0],q[2];
czz q[0],q[1];
cxz q[1],q[7];
czz q[0],q[1];
cxz q[0],q[2];
cxz q[2],q[5];
cyy q[5],q[8];
czy q[0],q[5];
czy q[2],q[5];
czy q[0],q[5];
cxx q[0],q[8];
cxz q[0],q[2];
rz(-0.024999999999999998) q[0];
u3(0.024999999999999994,pi/2,-pi/2) q[5];
h q[8];
cx q[2],q[8];
rz(0.024999999999999998) q[8];
cx q[2],q[8];
cxz q[0],q[2];
rz(0.024999999999999998) q[0];
h q[8];
cxx q[0],q[8];
czy q[0],q[5];
u3(0.024999999999999994,-pi/2,pi/2) q[5];
czy q[2],q[5];
u3(0.024999999999999994,pi/2,-pi/2) q[5];
czy q[0],q[5];
u3(0.024999999999999994,-pi/2,pi/2) q[5];
u3(0.024999999999999994,pi/2,-pi/2) q[8];
cyy q[5],q[8];
cxz q[2],q[5];
cxz q[0],q[2];
czz q[0],q[1];
cxz q[1],q[8];
czz q[0],q[1];
cxz q[0],q[2];
cxz q[2],q[5];
cyy q[5],q[9];
czy q[0],q[5];
czy q[2],q[5];
czy q[0],q[5];
cxx q[0],q[9];
cxz q[0],q[2];
rz(-0.024999999999999998) q[0];
u3(0.024999999999999994,pi/2,-pi/2) q[5];
h q[9];
cx q[2],q[9];
rz(0.024999999999999998) q[9];
cx q[2],q[9];
cxz q[0],q[2];
rz(0.024999999999999998) q[0];
h q[9];
cxx q[0],q[9];
czy q[0],q[5];
u3(0.024999999999999994,-pi/2,pi/2) q[5];
czy q[2],q[5];
u3(0.024999999999999994,pi/2,-pi/2) q[5];
czy q[0],q[5];
u3(0.024999999999999994,-pi/2,pi/2) q[5];
u3(0.024999999999999994,pi/2,-pi/2) q[9];
cyy q[5],q[9];
cxz q[2],q[5];
cxz q[0],q[2];
czz q[0],q[1];
cxz q[1],q[8];
cxz q[1],q[7];
cxz q[1],q[6];
