OPENQASM 2.0;
include "qelib1.inc";
qreg q[64];
creg c[16];
u2(0,pi/2) q[0];
u2(0,pi) q[1];
u2(0,pi) q[2];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[2];
u3(0.0013232938,-pi/2,0) q[1];
u3(0.0013232938,-pi/2,0) q[2];
u2(0,pi) q[3];
u2(0,pi) q[4];
cx q[3],q[4];
cx q[4],q[3];
cx q[3],q[4];
u3(0.0013232938,-pi/2,0) q[3];
u2(0,pi) q[5];
cx q[4],q[5];
cx q[5],q[4];
cx q[4],q[5];
u2(0,pi) q[6];
u2(0,pi) q[7];
cx q[6],q[7];
cx q[7],q[6];
cx q[6],q[7];
cx q[5],q[6];
cx q[6],q[5];
cx q[5],q[6];
cx q[4],q[5];
cx q[5],q[4];
cx q[4],q[5];
u3(0.0013232938,-pi/2,0) q[5];
u2(pi/2,-pi/2) q[7];
u3(1.5721196,-pi/2,-pi) q[8];
cx q[7],q[8];
u2(-pi/2,pi/2) q[7];
u3(0.0018714197,-3*pi/4,-3*pi/4) q[8];
cx q[7],q[8];
u3(1.4707963,-pi,pi/2) q[7];
u3(1.569473,1.569473,-pi) q[8];
cx q[7],q[8];
u2(-pi,pi/2) q[7];
u2(1.569473,-pi) q[8];
u2(0,pi) q[9];
u3(1.5721196,-pi/2,-pi) q[10];
cx q[0],q[10];
u2(-pi/2,pi/2) q[0];
u3(0.0018714197,-3*pi/4,-3*pi/4) q[10];
cx q[0],q[10];
u3(1.4707963,-pi,pi/2) q[0];
u3(1.569473,1.569473,-pi) q[10];
cx q[0],q[10];
u2(-pi,-pi) q[0];
cx q[0],q[1];
u2(-pi/2,pi/2) q[0];
u3(0.0018714197,-3*pi/4,-3*pi/4) q[1];
cx q[0],q[1];
u3(1.4707963,-pi,pi/2) q[0];
u3(1.569473,1.569473,-pi) q[1];
cx q[0],q[1];
u2(-pi,-pi) q[0];
u3(3.1402694,3.0642155e-13,pi/2) q[1];
cx q[1],q[2];
u2(-pi/2,pi/2) q[1];
u3(3.1402694,3.0642155e-13,pi/2) q[10];
u3(0.0018714197,-3*pi/4,-3*pi/4) q[2];
cx q[1],q[2];
u3(1.4707963,-pi,pi/2) q[1];
u3(1.569473,1.569473,-pi) q[2];
cx q[1],q[2];
u2(-pi,pi/2) q[1];
u3(3.1402694,3.0642155e-13,pi/2) q[2];
cx q[2],q[3];
u2(-pi/2,pi/2) q[2];
u3(0.0018714197,-3*pi/4,-3*pi/4) q[3];
cx q[2],q[3];
u3(1.4707963,-pi,pi/2) q[2];
u3(1.569473,1.569473,-pi) q[3];
cx q[2],q[3];
u2(-pi,pi/2) q[2];
u2(1.569473,-pi) q[3];
u2(0,pi) q[11];
u2(0,pi) q[12];
u3(1.5721196,-pi/2,-pi) q[13];
cx q[10],q[13];
u2(-pi/2,pi/2) q[10];
u3(0.0018714197,-3*pi/4,-3*pi/4) q[13];
cx q[10],q[13];
u3(1.4707963,-pi,pi/2) q[10];
u3(1.569473,1.569473,-pi) q[13];
cx q[10],q[13];
u3(1.569473,pi/2,pi/2) q[10];
cx q[0],q[10];
u2(-pi/2,pi/2) q[0];
u3(0.0018714197,-3*pi/4,-3*pi/4) q[10];
cx q[0],q[10];
u3(1.4707963,-pi,pi/2) q[0];
u3(1.569473,1.569473,-pi) q[10];
cx q[0],q[10];
u2(-pi,pi/2) q[0];
cx q[0],q[1];
u1(0.1) q[1];
cx q[0],q[1];
cx q[1],q[2];
u2(1.569473,-pi) q[10];
u2(1.569473,-pi) q[13];
cx q[2],q[1];
cx q[1],q[2];
u2(pi/2,-pi/2) q[1];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[3];
u2(pi/2,-pi/2) q[2];
cx q[3],q[4];
cx q[4],q[3];
cx q[3],q[4];
u3(0.0013232938,-pi/2,0) q[3];
cx q[2],q[3];
u2(-pi/2,pi/2) q[2];
u3(0.0018714197,-3*pi/4,-3*pi/4) q[3];
cx q[2],q[3];
u3(1.4707963,-pi,pi/2) q[2];
u3(1.569473,1.569473,-pi) q[3];
cx q[2],q[3];
u3(1.569473,pi/2,pi/2) q[2];
cx q[1],q[2];
u2(-pi/2,pi/2) q[1];
u3(0.0018714197,-3*pi/4,-3*pi/4) q[2];
cx q[1],q[2];
u3(1.4707963,-pi,pi/2) q[1];
u3(1.569473,1.569473,-pi) q[2];
cx q[1],q[2];
u2(-pi,pi/2) q[1];
cx q[0],q[1];
cx q[1],q[0];
cx q[0],q[1];
cx q[0],q[10];
u2(pi/2,-pi/2) q[1];
u1(0.1) q[10];
cx q[0],q[10];
u3(1.5707981,-0.0013232926,-3.1402694) q[2];
cx q[1],q[2];
u2(-pi/2,pi/2) q[1];
u3(0.0018714197,-3*pi/4,-3*pi/4) q[2];
cx q[1],q[2];
u3(1.4707963,-pi,pi/2) q[1];
u3(1.569473,1.569473,-pi) q[2];
cx q[1],q[2];
u2(-pi,pi/2) q[1];
u2(1.569473,-pi) q[2];
u2(1.569473,-pi) q[3];
u2(pi/2,-pi/2) q[4];
cx q[4],q[5];
u2(-pi/2,pi/2) q[4];
u3(0.0018714197,-3*pi/4,-3*pi/4) q[5];
cx q[4],q[5];
u3(1.4707963,-pi,pi/2) q[4];
u3(1.569473,1.569473,-pi) q[5];
cx q[4],q[5];
u2(-pi,pi/2) q[4];
cx q[3],q[4];
cx q[4],q[3];
cx q[3],q[4];
cx q[2],q[3];
u1(0.1) q[3];
cx q[2],q[3];
u2(pi/2,-pi/2) q[2];
u2(1.569473,-pi) q[5];
cx q[5],q[6];
cx q[6],q[5];
cx q[5],q[6];
cx q[4],q[5];
u1(0.1) q[5];
cx q[4],q[5];
cx q[4],q[11];
cx q[11],q[4];
cx q[4],q[11];
cx q[3],q[4];
cx q[4],q[3];
cx q[3],q[4];
u3(0.0013232938,-pi/2,0) q[3];
cx q[2],q[3];
u2(-pi/2,pi/2) q[2];
u3(0.0018714197,-3*pi/4,-3*pi/4) q[3];
cx q[2],q[3];
u3(1.4707963,-pi,pi/2) q[2];
u3(1.569473,1.569473,-pi) q[3];
cx q[2],q[3];
u2(-pi,pi/2) q[2];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[2];
cx q[0],q[1];
cx q[1],q[0];
cx q[0],q[1];
cx q[0],q[10];
u1(0.1) q[10];
cx q[0],q[10];
u2(pi/2,-pi/2) q[10];
u2(1.569473,-pi) q[3];
u2(pi/2,-pi/2) q[5];
cx q[6],q[7];
cx q[7],q[6];
cx q[6],q[7];
u3(0.0013232938,-pi/2,0) q[6];
cx q[5],q[6];
u2(-pi/2,pi/2) q[5];
u3(0.0018714197,-3*pi/4,-3*pi/4) q[6];
cx q[5],q[6];
u3(1.4707963,-pi,pi/2) q[5];
u3(1.569473,1.569473,-pi) q[6];
cx q[5],q[6];
u2(-pi,pi/2) q[5];
cx q[4],q[5];
cx q[5],q[4];
cx q[4],q[5];
cx q[4],q[11];
u1(0.1) q[11];
cx q[4],q[11];
u2(1.569473,-pi) q[6];
cx q[5],q[6];
cx q[6],q[5];
cx q[5],q[6];
u2(pi/2,-pi/2) q[6];
cx q[7],q[8];
cx q[8],q[7];
cx q[7],q[8];
cx q[8],q[12];
u1(0.1) q[12];
cx q[8],q[12];
cx q[8],q[9];
cx q[9],q[8];
cx q[8],q[9];
cx q[7],q[8];
cx q[8],q[7];
cx q[7],q[8];
u3(0.0013232938,-pi/2,0) q[7];
cx q[6],q[7];
u2(-pi/2,pi/2) q[6];
u3(0.0018714197,-3*pi/4,-3*pi/4) q[7];
cx q[6],q[7];
u3(1.4707963,-pi,pi/2) q[6];
u3(1.569473,1.569473,-pi) q[7];
cx q[6],q[7];
u2(-pi,pi/2) q[6];
cx q[5],q[6];
cx q[6],q[5];
cx q[5],q[6];
cx q[4],q[5];
cx q[5],q[4];
cx q[4],q[5];
cx q[4],q[11];
u1(0.1) q[11];
cx q[4],q[11];
u2(pi/2,-pi/2) q[4];
u3(0.0013232938,-pi/2,0) q[6];
u2(1.569473,-pi) q[7];
cx q[8],q[12];
u1(0.1) q[12];
cx q[8],q[12];
u3(0.0013232938,-pi/2,0) q[8];
u2(0,pi) q[14];
cx q[13],q[14];
u1(0.1) q[14];
cx q[13],q[14];
u3(0.0013232938,-pi/2,0) q[13];
cx q[10],q[13];
u2(-pi/2,pi/2) q[10];
u3(0.0018714197,-3*pi/4,-3*pi/4) q[13];
cx q[10],q[13];
u3(1.4707963,-pi,pi/2) q[10];
u3(1.569473,1.569473,-pi) q[13];
cx q[10],q[13];
u2(-pi,pi/2) q[10];
u2(1.569473,-pi) q[13];
u2(pi/2,-pi/2) q[14];
u2(0,pi) q[15];
cx q[11],q[17];
cx q[17],q[11];
cx q[11],q[17];
cx q[16],q[17];
cx q[17],q[16];
cx q[16],q[17];
cx q[15],q[16];
cx q[16],q[15];
cx q[15],q[16];
u3(0.0013232938,-pi/2,0) q[15];
cx q[14],q[15];
u2(-pi/2,pi/2) q[14];
u3(0.0018714197,-3*pi/4,-3*pi/4) q[15];
cx q[14],q[15];
u3(1.4707963,-pi,pi/2) q[14];
u3(1.569473,1.569473,-pi) q[15];
cx q[14],q[15];
u2(-pi,pi/2) q[14];
u2(1.569473,-pi) q[15];
cx q[16],q[17];
cx q[17],q[16];
cx q[16],q[17];
cx q[11],q[17];
cx q[17],q[11];
cx q[11],q[17];
u3(0.0013232938,-pi/2,0) q[11];
cx q[4],q[11];
u3(0.0018714197,-3*pi/4,-3*pi/4) q[11];
u2(-pi/2,pi/2) q[4];
cx q[4],q[11];
u3(1.569473,1.569473,-pi) q[11];
u3(1.4707963,-pi,pi/2) q[4];
cx q[4],q[11];
u2(1.569473,-pi) q[11];
u2(-pi,pi/2) q[4];
cx q[4],q[5];
cx q[5],q[4];
cx q[4],q[5];
u2(pi/2,-pi/2) q[5];
cx q[5],q[6];
u2(-pi/2,pi/2) q[5];
u3(0.0018714197,-3*pi/4,-3*pi/4) q[6];
cx q[5],q[6];
u3(1.4707963,-pi,pi/2) q[5];
u3(1.569473,1.569473,-pi) q[6];
cx q[5],q[6];
u2(-pi,pi/2) q[5];
u2(1.569473,-pi) q[6];
cx q[6],q[7];
cx q[7],q[6];
cx q[6],q[7];
u2(pi/2,-pi/2) q[7];
cx q[7],q[8];
u2(-pi/2,pi/2) q[7];
u3(0.0018714197,-3*pi/4,-3*pi/4) q[8];
cx q[7],q[8];
u3(1.4707963,-pi,pi/2) q[7];
u3(1.569473,1.569473,-pi) q[8];
cx q[7],q[8];
u2(-pi,pi/2) q[7];
u2(1.569473,-pi) q[8];
measure q[10] -> c[0];
measure q[9] -> c[1];
measure q[13] -> c[2];
measure q[5] -> c[3];
measure q[2] -> c[4];
measure q[6] -> c[5];
measure q[7] -> c[6];
measure q[1] -> c[7];
measure q[4] -> c[8];
measure q[11] -> c[9];
measure q[14] -> c[10];
measure q[0] -> c[11];
measure q[12] -> c[12];
measure q[3] -> c[13];
measure q[15] -> c[14];
measure q[8] -> c[15];
