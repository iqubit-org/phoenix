OPENQASM 2.0;
include "qelib1.inc";

qreg q[10];
u3(0.5*pi,0.0*pi,1.0*pi) q[1];
u3(1.5*pi,0.0*pi,1.0*pi) q[2];
u3(1.5*pi,0.0*pi,1.0*pi) q[3];
u3(1.0*pi,-0.5*pi,1.0*pi) q[4];
u3(1.0*pi,-0.5*pi,4.0*pi) q[5];
u3(0.5*pi,0.0*pi,4.0*pi) q[6];
u3(0.5*pi,0.0*pi,4.0*pi) q[7];
u3(0.5*pi,0.0*pi,4.0*pi) q[8];
u3(0.0*pi,-0.5*pi,1.0*pi) q[9];
cx q[1],q[2];
cx q[4],q[5];
cx q[1],q[3];
u3(0.5*pi,0.0*pi,1.0*pi) q[5];
cx q[1],q[6];
cx q[1],q[7];
cx q[1],q[8];
cx q[0],q[1];
cx q[4],q[0];
u3(0.5*pi,0.0*pi,1.0*pi) q[0];
u3(0.5*pi,0.0*pi,0.5*pi) q[4];
cx q[4],q[9];
u3(0.5079577471545939*pi,0.0*pi,1.0*pi) q[4];
u3(0.0*pi,-0.5*pi,1.0*pi) q[9];
cx q[0],q[9];
cx q[0],q[5];
u3(0.5*pi,0.0*pi,1.0*pi) q[9];
u3(2.007957747154594*pi,-0.5*pi,1.5*pi) q[0];
u3(0.5*pi,-0.5*pi,1.0*pi) q[5];
cx q[5],q[9];
u3(1.0*pi,-0.5*pi,3.507957747154595*pi) q[9];
cx q[5],q[9];
u3(0.5*pi,0.0*pi,0.5*pi) q[5];
u3(0.5*pi,0.0*pi,1.0*pi) q[9];
cx q[0],q[5];
u3(3.007957747154595*pi,0.0*pi,0.5*pi) q[0];
u3(0.5*pi,-0.5*pi,1.0*pi) q[5];
cx q[0],q[9];
u3(0.5*pi,-0.5*pi,0.5*pi) q[0];
u3(1.992042252845406*pi,0.0*pi,0.5*pi) q[9];
cx q[0],q[4];
u3(1.007957747154595*pi,0.0*pi,1.0*pi) q[4];
cx q[5],q[4];
u3(2.007957747154594*pi,0.0*pi,4.0*pi) q[4];
cx q[0],q[4];
u3(2.507957747154595*pi,-0.5*pi,1.0*pi) q[4];
cx q[4],q[9];
u3(0.5*pi,0.0*pi,0.5*pi) q[4];
cx q[5],q[4];
cx q[0],q[4];
u3(0.5*pi,0.0*pi,0.5*pi) q[5];
u3(0.0*pi,-0.5*pi,1.0*pi) q[4];
cx q[4],q[1];
cx q[1],q[8];
cx q[4],q[1];
u3(0.5*pi,-0.5*pi,1.0*pi) q[8];
u3(0.5*pi,-0.5*pi,0.5*pi) q[4];
cx q[4],q[8];
u3(0.5*pi,0.0*pi,0.5*pi) q[4];
u3(0.0*pi,-0.5*pi,1.0*pi) q[8];
cx q[0],q[4];
u3(0.5*pi,0.0*pi,0.5*pi) q[0];
u3(1.9920422528454065*pi,0.0*pi,4.0*pi) q[4];
cx q[0],q[8];
cx q[0],q[5];
u3(0.5*pi,0.0*pi,1.0*pi) q[8];
u3(2.007957747154594*pi,-0.5*pi,1.5*pi) q[0];
u3(0.5*pi,-0.5*pi,1.0*pi) q[5];
cx q[5],q[8];
u3(1.0*pi,-0.5*pi,2.507957747154595*pi) q[8];
cx q[5],q[8];
u3(0.5*pi,0.0*pi,0.5*pi) q[5];
u3(0.5*pi,0.0*pi,1.0*pi) q[8];
cx q[0],q[5];
u3(2.007957747154594*pi,1.0*pi,0.5*pi) q[0];
u3(0.5*pi,-0.5*pi,1.0*pi) q[5];
cx q[0],q[8];
u3(0.5*pi,-0.5*pi,0.5*pi) q[0];
u3(0.5*pi,-0.4920422528454056*pi,4.0*pi) q[8];
cx q[0],q[4];
u3(1.007957747154595*pi,0.0*pi,1.0*pi) q[4];
cx q[5],q[4];
u3(0.9920422528454064*pi,0.0*pi,1.0*pi) q[4];
u3(0.5*pi,0.0*pi,0.5*pi) q[5];
cx q[0],q[4];
u3(0.5*pi,-0.5*pi,0.5*pi) q[0];
u3(2.507957747154595*pi,0.0*pi,1.0*pi) q[4];
cx q[4],q[8];
cx q[5],q[4];
u3(0.0*pi,-0.5*pi,1.0*pi) q[8];
cx q[0],q[5];
u3(0.5*pi,0.0*pi,1.0*pi) q[4];
u3(0.5*pi,0.0*pi,0.5*pi) q[0];
u3(0.5*pi,-0.5*pi,1.0*pi) q[5];
cx q[0],q[1];
cx q[1],q[3];
cx q[1],q[8];
u3(0.5*pi,-0.5*pi,1.0*pi) q[3];
cx q[0],q[1];
cx q[3],q[4];
cx q[3],q[0];
u3(0.5*pi,0.0*pi,1.0*pi) q[4];
u3(0.5*pi,0.0*pi,1.0*pi) q[0];
u3(0.5*pi,0.0*pi,0.5*pi) q[3];
cx q[3],q[9];
u3(2.492042252845406*pi,0.0*pi,4.0*pi) q[3];
u3(0.0*pi,-0.5*pi,1.0*pi) q[9];
cx q[0],q[9];
cx q[0],q[4];
u3(0.5*pi,0.0*pi,1.0*pi) q[9];
u3(2.007957747154594*pi,-0.5*pi,1.5*pi) q[0];
u3(0.5*pi,-0.5*pi,1.0*pi) q[4];
cx q[4],q[9];
u3(1.0*pi,-0.5*pi,2.507957747154595*pi) q[9];
cx q[4],q[9];
u3(0.5*pi,0.0*pi,0.5*pi) q[4];
u3(0.5*pi,0.0*pi,1.0*pi) q[9];
cx q[0],q[4];
u3(2.007957747154594*pi,1.0*pi,0.5*pi) q[0];
u3(0.5*pi,-0.5*pi,1.0*pi) q[4];
cx q[0],q[9];
u3(0.5*pi,-0.5*pi,0.5*pi) q[0];
u3(3.9920422528454065*pi,0.0*pi,0.5*pi) q[9];
cx q[0],q[3];
u3(1.007957747154595*pi,0.0*pi,1.0*pi) q[3];
cx q[4],q[3];
u3(0.9920422528454064*pi,0.0*pi,1.0*pi) q[3];
u3(0.5*pi,0.0*pi,0.5*pi) q[4];
cx q[0],q[3];
u3(0.5*pi,-0.5*pi,0.5*pi) q[0];
u3(3.4920422528454056*pi,0.0*pi,4.0*pi) q[3];
cx q[3],q[9];
cx q[4],q[3];
cx q[0],q[4];
u3(0.5*pi,0.0*pi,1.0*pi) q[3];
u3(0.5*pi,0.0*pi,0.5*pi) q[0];
cx q[5],q[3];
cx q[0],q[1];
u3(0.5*pi,0.0*pi,1.0*pi) q[3];
cx q[1],q[8];
cx q[1],q[7];
cx q[1],q[4];
u3(0.5*pi,-0.5*pi,1.0*pi) q[7];
cx q[0],q[1];
cx q[5],q[0];
u3(0.5*pi,0.0*pi,1.0*pi) q[0];
u3(0.5*pi,0.0*pi,0.5*pi) q[5];
cx q[5],q[7];
u3(2.492042252845406*pi,0.0*pi,4.0*pi) q[5];
u3(0.0*pi,-0.5*pi,1.0*pi) q[7];
cx q[0],q[7];
cx q[0],q[3];
u3(0.5*pi,0.0*pi,1.0*pi) q[7];
u3(2.007957747154594*pi,-0.5*pi,1.5*pi) q[0];
u3(0.5*pi,-0.5*pi,1.0*pi) q[3];
cx q[3],q[7];
u3(1.0*pi,-0.5*pi,3.507957747154595*pi) q[7];
cx q[3],q[7];
u3(0.5*pi,0.0*pi,0.5*pi) q[3];
u3(0.5*pi,0.0*pi,1.0*pi) q[7];
cx q[0],q[3];
u3(1.0079577471545942*pi,1.0*pi,0.5*pi) q[0];
u3(0.5*pi,-0.5*pi,1.0*pi) q[3];
cx q[0],q[7];
u3(0.5*pi,-0.5*pi,0.5*pi) q[0];
u3(0.5*pi,2.5079577471545935*pi,4.0*pi) q[7];
cx q[0],q[5];
u3(1.007957747154595*pi,0.0*pi,1.0*pi) q[5];
cx q[3],q[5];
u3(0.5*pi,0.0*pi,0.5*pi) q[3];
u3(0.9920422528454064*pi,0.0*pi,1.0*pi) q[5];
cx q[0],q[5];
u3(0.5*pi,-0.5*pi,0.5*pi) q[0];
u3(1.5079577471545949*pi,0.0*pi,1.0*pi) q[5];
cx q[5],q[7];
cx q[3],q[5];
u3(0.0*pi,-0.5*pi,1.0*pi) q[7];
cx q[0],q[3];
u3(0.5*pi,0.0*pi,1.0*pi) q[5];
u3(0.5*pi,0.0*pi,0.5*pi) q[0];
u3(0.5*pi,-0.5*pi,1.0*pi) q[3];
cx q[0],q[1];
cx q[1],q[4];
cx q[1],q[2];
u3(0.5*pi,-0.5*pi,1.0*pi) q[4];
cx q[1],q[7];
u3(0.5*pi,-0.5*pi,1.0*pi) q[2];
cx q[1],q[8];
cx q[2],q[5];
cx q[0],q[1];
u3(0.5*pi,0.0*pi,1.0*pi) q[5];
cx q[2],q[0];
u3(0.5*pi,0.0*pi,1.0*pi) q[0];
u3(0.5*pi,0.0*pi,0.5*pi) q[2];
cx q[2],q[9];
u3(2.492042252845406*pi,0.0*pi,4.0*pi) q[2];
u3(0.0*pi,-0.5*pi,1.0*pi) q[9];
cx q[0],q[9];
cx q[0],q[5];
u3(0.5*pi,0.0*pi,1.0*pi) q[9];
u3(2.007957747154594*pi,-0.5*pi,1.5*pi) q[0];
u3(0.5*pi,-0.5*pi,1.0*pi) q[5];
cx q[5],q[9];
u3(1.0*pi,-0.5*pi,2.507957747154595*pi) q[9];
cx q[5],q[9];
u3(0.5*pi,0.0*pi,0.5*pi) q[5];
u3(0.5*pi,0.0*pi,1.0*pi) q[9];
cx q[0],q[5];
u3(2.007957747154594*pi,1.0*pi,0.5*pi) q[0];
u3(0.5*pi,-0.5*pi,1.0*pi) q[5];
cx q[0],q[9];
u3(0.5*pi,-0.5*pi,0.5*pi) q[0];
u3(1.992042252845406*pi,1.0*pi,0.5*pi) q[9];
cx q[0],q[2];
u3(1.007957747154595*pi,0.0*pi,1.0*pi) q[2];
cx q[5],q[2];
u3(0.9920422528454064*pi,0.0*pi,1.0*pi) q[2];
u3(0.5*pi,0.0*pi,0.5*pi) q[5];
cx q[0],q[2];
u3(0.5*pi,-0.5*pi,0.5*pi) q[0];
u3(2.507957747154595*pi,0.0*pi,1.0*pi) q[2];
cx q[2],q[9];
cx q[5],q[2];
cx q[0],q[5];
u3(0.5*pi,0.0*pi,1.0*pi) q[2];
u3(0.5*pi,0.0*pi,0.5*pi) q[0];
cx q[4],q[2];
cx q[0],q[1];
u3(0.5*pi,0.0*pi,1.0*pi) q[2];
cx q[1],q[8];
cx q[1],q[5];
u3(0.5*pi,-0.5*pi,1.0*pi) q[8];
cx q[0],q[1];
cx q[4],q[0];
u3(0.5*pi,0.0*pi,1.0*pi) q[0];
u3(0.5*pi,0.0*pi,0.5*pi) q[4];
cx q[4],q[8];
u3(2.492042252845406*pi,0.0*pi,4.0*pi) q[4];
u3(0.0*pi,-0.5*pi,1.0*pi) q[8];
cx q[0],q[8];
cx q[0],q[2];
u3(0.5*pi,0.0*pi,1.0*pi) q[8];
u3(2.007957747154594*pi,-0.5*pi,1.5*pi) q[0];
u3(0.5*pi,-0.5*pi,1.0*pi) q[2];
cx q[2],q[8];
u3(1.0*pi,-0.5*pi,2.507957747154595*pi) q[8];
cx q[2],q[8];
u3(0.5*pi,0.0*pi,0.5*pi) q[2];
u3(0.5*pi,0.0*pi,1.0*pi) q[8];
cx q[0],q[2];
u3(2.007957747154594*pi,1.0*pi,0.5*pi) q[0];
u3(0.5*pi,-0.5*pi,1.0*pi) q[2];
cx q[0],q[8];
u3(0.5*pi,-0.5*pi,0.5*pi) q[0];
u3(3.5*pi,0.4920422528454056*pi,4.0*pi) q[8];
cx q[0],q[4];
u3(1.007957747154595*pi,0.0*pi,1.0*pi) q[4];
cx q[2],q[4];
u3(0.5*pi,0.0*pi,0.5*pi) q[2];
u3(0.9920422528454064*pi,0.0*pi,1.0*pi) q[4];
cx q[0],q[4];
u3(0.5*pi,-0.5*pi,0.5*pi) q[0];
u3(3.4920422528454056*pi,0.0*pi,4.0*pi) q[4];
cx q[4],q[8];
cx q[2],q[4];
u3(0.0*pi,-0.5*pi,1.0*pi) q[8];
cx q[0],q[2];
u3(0.5*pi,0.0*pi,1.0*pi) q[4];
u3(0.5*pi,0.0*pi,0.5*pi) q[0];
cx q[3],q[4];
cx q[0],q[1];
u3(0.5*pi,0.0*pi,1.0*pi) q[4];
cx q[1],q[7];
cx q[1],q[6];
cx q[1],q[2];
u3(0.5*pi,-0.5*pi,1.0*pi) q[6];
cx q[0],q[1];
cx q[3],q[0];
u3(0.5*pi,0.0*pi,1.0*pi) q[0];
u3(0.5*pi,0.0*pi,0.5*pi) q[3];
cx q[3],q[6];
u3(0.5079577471545939*pi,0.0*pi,1.0*pi) q[3];
u3(0.0*pi,-0.5*pi,1.0*pi) q[6];
cx q[0],q[6];
cx q[0],q[4];
u3(0.5*pi,0.0*pi,1.0*pi) q[6];
u3(2.007957747154594*pi,-0.5*pi,1.5*pi) q[0];
u3(0.5*pi,-0.5*pi,1.0*pi) q[4];
cx q[4],q[6];
u3(1.0*pi,-0.5*pi,3.507957747154595*pi) q[6];
cx q[4],q[6];
u3(0.5*pi,0.0*pi,0.5*pi) q[4];
u3(0.5*pi,0.0*pi,1.0*pi) q[6];
cx q[0],q[4];
u3(1.0079577471545942*pi,0.0*pi,0.5*pi) q[0];
u3(0.5*pi,-0.5*pi,1.0*pi) q[4];
cx q[0],q[6];
u3(0.5*pi,-0.5*pi,0.5*pi) q[0];
u3(0.5*pi,2.5079577471545935*pi,4.0*pi) q[6];
cx q[0],q[3];
u3(1.007957747154595*pi,0.0*pi,1.0*pi) q[3];
cx q[4],q[3];
u3(2.007957747154594*pi,0.0*pi,4.0*pi) q[3];
u3(0.5*pi,0.0*pi,0.5*pi) q[4];
cx q[0],q[3];
u3(0.5*pi,-0.5*pi,0.5*pi) q[0];
u3(1.5079577471545949*pi,0.0*pi,1.0*pi) q[3];
cx q[3],q[6];
cx q[4],q[3];
u3(0.0*pi,-0.5*pi,1.0*pi) q[6];
u3(0.5*pi,0.0*pi,1.0*pi) q[3];
u3(0.5*pi,-0.5*pi,0.5*pi) q[4];
cx q[6],q[7];
cx q[0],q[4];
cx q[6],q[8];
u3(0.5*pi,0.0*pi,0.5*pi) q[0];
u3(0.0*pi,-0.5*pi,1.0*pi) q[4];
cx q[0],q[1];
cx q[0],q[6];
u3(0.0*pi,-0.5*pi,1.0*pi) q[1];
cx q[1],q[2];
cx q[1],q[5];
u3(0.5*pi,-0.5*pi,1.0*pi) q[2];
u3(0.5*pi,-0.5*pi,0.5*pi) q[1];
cx q[1],q[3];
cx q[1],q[0];
u3(0.5*pi,0.0*pi,1.0*pi) q[3];
u3(0.5*pi,0.0*pi,1.0*pi) q[0];
u3(0.5*pi,0.0*pi,0.5*pi) q[1];
cx q[1],q[9];
u3(0.5079577471545939*pi,0.0*pi,1.0*pi) q[1];
u3(0.0*pi,-0.5*pi,1.0*pi) q[9];
cx q[0],q[9];
cx q[0],q[3];
u3(0.5*pi,0.0*pi,1.0*pi) q[9];
u3(1.992042252845406*pi,-0.5*pi,0.5*pi) q[0];
u3(0.5*pi,-0.5*pi,1.0*pi) q[3];
cx q[3],q[9];
u3(0.0*pi,-0.5*pi,0.5079577471545947*pi) q[9];
cx q[3],q[9];
u3(0.5*pi,0.0*pi,0.5*pi) q[3];
u3(0.5*pi,0.0*pi,1.0*pi) q[9];
cx q[0],q[3];
u3(2.007957747154595*pi,0.0*pi,0.5*pi) q[0];
u3(0.5*pi,-0.5*pi,1.0*pi) q[3];
cx q[0],q[9];
u3(0.5*pi,-0.5*pi,0.5*pi) q[0];
u3(0.9920422528454056*pi,1.0*pi,0.5*pi) q[9];
cx q[0],q[1];
u3(1.9920422528454051*pi,0.0*pi,4.0*pi) q[1];
cx q[3],q[1];
u3(2.007957747154594*pi,0.0*pi,4.0*pi) q[1];
u3(0.5*pi,0.0*pi,0.5*pi) q[3];
cx q[0],q[1];
u3(0.5*pi,-0.5*pi,0.5*pi) q[0];
u3(3.5079577471545944*pi,0.0*pi,1.0*pi) q[1];
cx q[1],q[9];
cx q[3],q[1];
u3(0.5*pi,-0.5*pi,1.0*pi) q[9];
cx q[0],q[3];
u3(0.5*pi,0.0*pi,1.0*pi) q[1];
u3(0.5*pi,0.0*pi,0.5*pi) q[0];
cx q[2],q[1];
cx q[3],q[5];
cx q[0],q[6];
u3(0.5*pi,0.0*pi,1.0*pi) q[1];
cx q[6],q[8];
cx q[6],q[7];
u3(0.5*pi,-0.5*pi,1.0*pi) q[8];
cx q[3],q[6];
u3(0.5*pi,-0.5*pi,1.0*pi) q[7];
cx q[0],q[3];
cx q[2],q[0];
u3(0.5*pi,0.0*pi,1.0*pi) q[0];
u3(0.5*pi,0.0*pi,0.5*pi) q[2];
cx q[2],q[7];
u3(2.492042252845406*pi,0.0*pi,4.0*pi) q[2];
u3(0.0*pi,-0.5*pi,1.0*pi) q[7];
cx q[0],q[7];
cx q[0],q[1];
u3(0.5*pi,0.0*pi,1.0*pi) q[7];
u3(2.007957747154594*pi,-0.5*pi,1.5*pi) q[0];
u3(0.5*pi,-0.5*pi,1.0*pi) q[1];
cx q[1],q[7];
u3(1.0*pi,-0.5*pi,2.507957747154595*pi) q[7];
cx q[1],q[7];
u3(0.5*pi,0.0*pi,0.5*pi) q[1];
u3(0.5*pi,0.0*pi,1.0*pi) q[7];
cx q[0],q[1];
u3(2.007957747154594*pi,1.0*pi,0.5*pi) q[0];
u3(0.5*pi,-0.5*pi,1.0*pi) q[1];
cx q[0],q[7];
u3(0.5*pi,-0.5*pi,0.5*pi) q[0];
u3(3.5*pi,0.4920422528454056*pi,4.0*pi) q[7];
cx q[0],q[2];
u3(1.007957747154595*pi,0.0*pi,1.0*pi) q[2];
cx q[1],q[2];
u3(0.5*pi,0.0*pi,0.5*pi) q[1];
u3(0.9920422528454064*pi,0.0*pi,1.0*pi) q[2];
cx q[0],q[2];
u3(0.5*pi,-0.5*pi,0.5*pi) q[0];
u3(1.4920422528454051*pi,0.0*pi,4.0*pi) q[2];
cx q[2],q[7];
cx q[1],q[2];
u3(0.0*pi,-0.5*pi,1.0*pi) q[7];
cx q[0],q[1];
u3(0.5*pi,0.0*pi,1.0*pi) q[2];
u3(0.5*pi,0.0*pi,0.5*pi) q[0];
u3(0.5*pi,-0.5*pi,1.0*pi) q[1];
cx q[0],q[3];
cx q[3],q[5];
cx q[3],q[7];
u3(0.5*pi,-0.5*pi,1.0*pi) q[5];
cx q[0],q[3];
cx q[5],q[2];
cx q[5],q[0];
u3(0.5*pi,0.0*pi,1.0*pi) q[2];
u3(0.5*pi,0.0*pi,1.0*pi) q[0];
u3(0.5*pi,0.0*pi,0.5*pi) q[5];
cx q[5],q[8];
u3(2.492042252845406*pi,0.0*pi,4.0*pi) q[5];
u3(0.0*pi,-0.5*pi,1.0*pi) q[8];
cx q[0],q[8];
cx q[0],q[2];
u3(0.5*pi,0.0*pi,1.0*pi) q[8];
u3(2.007957747154594*pi,-0.5*pi,1.5*pi) q[0];
u3(0.5*pi,-0.5*pi,1.0*pi) q[2];
cx q[2],q[8];
u3(1.0*pi,-0.5*pi,2.507957747154595*pi) q[8];
cx q[2],q[8];
u3(0.5*pi,0.0*pi,0.5*pi) q[2];
u3(0.5*pi,0.0*pi,1.0*pi) q[8];
cx q[0],q[2];
u3(2.007957747154594*pi,1.0*pi,0.5*pi) q[0];
u3(0.5*pi,-0.5*pi,1.0*pi) q[2];
cx q[0],q[8];
u3(0.5*pi,-0.5*pi,0.5*pi) q[0];
u3(1.9920422528454065*pi,1.0*pi,0.5*pi) q[8];
cx q[0],q[5];
u3(1.007957747154595*pi,0.0*pi,1.0*pi) q[5];
cx q[2],q[5];
u3(0.5*pi,0.0*pi,0.5*pi) q[2];
u3(0.9920422528454064*pi,0.0*pi,1.0*pi) q[5];
cx q[0],q[5];
u3(0.5*pi,-0.5*pi,0.5*pi) q[0];
u3(2.507957747154594*pi,0.0*pi,1.0*pi) q[5];
cx q[5],q[8];
cx q[2],q[5];
cx q[0],q[2];
u3(0.5*pi,0.0*pi,1.0*pi) q[5];
u3(0.5*pi,0.0*pi,0.5*pi) q[0];
cx q[1],q[5];
cx q[0],q[3];
u3(0.5*pi,0.0*pi,1.0*pi) q[5];
cx q[3],q[7];
cx q[3],q[6];
cx q[3],q[2];
u3(0.5*pi,-0.5*pi,1.0*pi) q[6];
cx q[0],q[3];
cx q[1],q[0];
u3(0.5*pi,0.0*pi,1.0*pi) q[0];
u3(0.5*pi,0.0*pi,0.5*pi) q[1];
cx q[1],q[6];
u3(0.5079577471545939*pi,0.0*pi,1.0*pi) q[1];
u3(0.0*pi,-0.5*pi,1.0*pi) q[6];
cx q[0],q[6];
cx q[0],q[5];
u3(0.5*pi,0.0*pi,1.0*pi) q[6];
u3(2.007957747154594*pi,-0.5*pi,1.5*pi) q[0];
u3(0.5*pi,-0.5*pi,1.0*pi) q[5];
cx q[5],q[6];
u3(1.0*pi,-0.5*pi,2.507957747154595*pi) q[6];
cx q[5],q[6];
u3(0.5*pi,0.0*pi,0.5*pi) q[5];
u3(0.5*pi,0.0*pi,1.0*pi) q[6];
cx q[0],q[5];
u3(2.007957747154595*pi,0.0*pi,0.5*pi) q[0];
u3(0.5*pi,-0.5*pi,1.0*pi) q[5];
cx q[0],q[6];
u3(0.5*pi,-0.5*pi,0.5*pi) q[0];
u3(3.5*pi,0.4920422528454056*pi,4.0*pi) q[6];
cx q[0],q[1];
u3(1.007957747154595*pi,0.0*pi,1.0*pi) q[1];
cx q[5],q[1];
u3(2.007957747154594*pi,0.0*pi,4.0*pi) q[1];
u3(0.5*pi,0.0*pi,0.5*pi) q[5];
cx q[0],q[1];
u3(0.5*pi,-0.5*pi,0.5*pi) q[0];
u3(1.4920422528454051*pi,0.0*pi,4.0*pi) q[1];
cx q[1],q[6];
cx q[5],q[1];
u3(0.0*pi,-0.5*pi,1.0*pi) q[6];
u3(0.5*pi,0.0*pi,1.0*pi) q[1];
u3(0.5*pi,-0.5*pi,0.5*pi) q[5];
cx q[6],q[7];
cx q[0],q[5];
u3(0.5*pi,0.0*pi,0.5*pi) q[0];
cx q[0],q[3];
cx q[0],q[6];
u3(0.0*pi,-0.5*pi,1.0*pi) q[3];
cx q[3],q[2];
u3(0.5*pi,-0.5*pi,1.0*pi) q[2];
u3(0.5*pi,-0.5*pi,0.5*pi) q[3];
cx q[3],q[1];
cx q[3],q[0];
u3(0.5*pi,0.0*pi,1.0*pi) q[1];
u3(0.5*pi,0.0*pi,1.0*pi) q[0];
u3(0.5*pi,0.0*pi,0.5*pi) q[3];
cx q[3],q[8];
u3(0.5079577471545939*pi,0.0*pi,1.0*pi) q[3];
u3(0.0*pi,-0.5*pi,1.0*pi) q[8];
cx q[0],q[8];
cx q[0],q[1];
u3(0.5*pi,0.0*pi,1.0*pi) q[8];
u3(1.992042252845406*pi,-0.5*pi,0.5*pi) q[0];
u3(0.5*pi,-0.5*pi,1.0*pi) q[1];
cx q[1],q[8];
u3(0.0*pi,-0.5*pi,1.5079577471545949*pi) q[8];
cx q[1],q[8];
u3(0.5*pi,0.0*pi,0.5*pi) q[1];
u3(0.5*pi,0.0*pi,1.0*pi) q[8];
cx q[0],q[1];
u3(1.0079577471545942*pi,0.0*pi,0.5*pi) q[0];
u3(0.5*pi,-0.5*pi,1.0*pi) q[1];
cx q[0],q[8];
u3(0.5*pi,-0.5*pi,0.5*pi) q[0];
u3(0.9920422528454056*pi,0.0*pi,0.5*pi) q[8];
cx q[0],q[3];
u3(1.9920422528454051*pi,0.0*pi,4.0*pi) q[3];
cx q[1],q[3];
u3(0.5*pi,0.0*pi,0.5*pi) q[1];
u3(2.007957747154594*pi,0.0*pi,4.0*pi) q[3];
cx q[0],q[3];
u3(0.5*pi,-0.5*pi,0.5*pi) q[0];
u3(2.507957747154594*pi,0.0*pi,1.0*pi) q[3];
cx q[3],q[8];
cx q[1],q[3];
u3(0.5*pi,0.0*pi,1.0*pi) q[8];
cx q[0],q[1];
u3(0.5*pi,0.0*pi,1.0*pi) q[3];
u3(0.5*pi,0.0*pi,0.5*pi) q[0];
cx q[2],q[3];
cx q[0],q[6];
u3(0.5*pi,0.0*pi,1.0*pi) q[3];
cx q[6],q[7];
cx q[1],q[6];
u3(0.5*pi,-0.5*pi,1.0*pi) q[7];
cx q[0],q[1];
cx q[2],q[0];
u3(0.5*pi,0.0*pi,1.0*pi) q[0];
u3(0.5*pi,0.0*pi,0.5*pi) q[2];
cx q[2],q[7];
u3(0.5079577471545939*pi,0.0*pi,1.0*pi) q[2];
u3(0.0*pi,-0.5*pi,1.0*pi) q[7];
cx q[0],q[7];
cx q[0],q[3];
u3(0.5*pi,0.0*pi,1.0*pi) q[7];
u3(2.007957747154594*pi,-0.5*pi,1.5*pi) q[0];
u3(0.5*pi,-0.5*pi,1.0*pi) q[3];
cx q[3],q[7];
u3(1.0*pi,-0.5*pi,3.507957747154595*pi) q[7];
cx q[3],q[7];
u3(0.5*pi,0.0*pi,0.5*pi) q[3];
u3(0.5*pi,0.0*pi,1.0*pi) q[7];
cx q[0],q[3];
u3(3.007957747154595*pi,0.0*pi,0.5*pi) q[0];
u3(0.5*pi,-0.5*pi,1.0*pi) q[3];
cx q[0],q[7];
u3(0.5*pi,-0.5*pi,0.5*pi) q[0];
u3(3.5*pi,0.4920422528454056*pi,4.0*pi) q[7];
cx q[0],q[2];
u3(1.007957747154595*pi,0.0*pi,1.0*pi) q[2];
cx q[3],q[2];
u3(2.007957747154594*pi,0.0*pi,4.0*pi) q[2];
u3(0.5*pi,0.0*pi,0.5*pi) q[3];
cx q[0],q[2];
u3(0.5*pi,-0.5*pi,0.5*pi) q[0];
u3(0.5079577471545944*pi,0.0*pi,1.0*pi) q[2];
cx q[2],q[7];
u3(0.5*pi,-0.5*pi,0.5*pi) q[2];
u3(0.0*pi,-0.5*pi,1.0*pi) q[7];
cx q[3],q[2];
cx q[0],q[3];
u3(0.5*pi,-0.5*pi,1.0*pi) q[2];
u3(0.5*pi,-0.5*pi,0.5*pi) q[0];
cx q[0],q[1];
cx q[1],q[6];
cx q[1],q[3];
cx q[6],q[7];
cx q[1],q[5];
cx q[6],q[8];
cx q[0],q[1];
cx q[2],q[6];
u3(0.5*pi,0.0*pi,0.5*pi) q[0];
u3(0.5*pi,0.0*pi,0.5*pi) q[2];
cx q[0],q[4];
cx q[2],q[9];
u3(1.0318309886183792*pi,-0.5*pi,4.0*pi) q[0];
u3(1.968169011381621*pi,-0.5*pi,1.0*pi) q[2];
u3(1.5*pi,-0.5*pi,1.468169011381621*pi) q[4];
u3(0.5*pi,-0.5*pi,1.468169011381621*pi) q[9];
cx q[0],q[4];
cx q[2],q[9];
u3(0.5*pi,0.0*pi,0.5*pi) q[0];
u3(0.5*pi,-0.5*pi,0.5*pi) q[2];
cx q[0],q[1];
cx q[2],q[6];
cx q[1],q[5];
u3(0.0*pi,-0.5*pi,1.0*pi) q[6];
cx q[1],q[3];
u3(0.0*pi,-0.5*pi,1.0*pi) q[5];
cx q[6],q[8];
u3(0.5*pi,-0.5*pi,0.5*pi) q[1];
u3(0.5*pi,-0.5*pi,1.0*pi) q[3];
cx q[6],q[7];
u3(0.0*pi,-0.5*pi,1.0*pi) q[8];
cx q[0],q[1];
cx q[3],q[2];
u3(0.5*pi,-0.5*pi,0.5*pi) q[6];
u3(0.5*pi,-0.5*pi,1.0*pi) q[7];
cx q[3],q[0];
u3(0.5*pi,0.0*pi,1.0*pi) q[2];
u3(0.5*pi,0.0*pi,1.0*pi) q[0];
u3(0.5*pi,0.0*pi,0.5*pi) q[3];
cx q[3],q[6];
u3(0.5079577471545939*pi,0.0*pi,1.0*pi) q[3];
u3(0.0*pi,-0.5*pi,1.0*pi) q[6];
cx q[0],q[6];
cx q[0],q[2];
u3(0.5*pi,0.0*pi,1.0*pi) q[6];
u3(1.992042252845406*pi,-0.5*pi,0.5*pi) q[0];
u3(0.5*pi,-0.5*pi,1.0*pi) q[2];
cx q[2],q[6];
u3(0.0*pi,-0.5*pi,1.5079577471545949*pi) q[6];
cx q[2],q[6];
u3(0.5*pi,0.0*pi,0.5*pi) q[2];
u3(0.5*pi,0.0*pi,1.0*pi) q[6];
cx q[0],q[2];
u3(3.007957747154595*pi,0.0*pi,0.5*pi) q[0];
u3(0.5*pi,-0.5*pi,1.0*pi) q[2];
cx q[0],q[6];
u3(0.5*pi,-0.5*pi,0.5*pi) q[0];
u3(0.5*pi,2.5079577471545935*pi,4.0*pi) q[6];
cx q[0],q[3];
u3(1.9920422528454051*pi,0.0*pi,4.0*pi) q[3];
cx q[2],q[3];
u3(0.5*pi,0.0*pi,0.5*pi) q[2];
u3(2.007957747154594*pi,0.0*pi,4.0*pi) q[3];
cx q[0],q[3];
u3(0.5*pi,-0.5*pi,0.5*pi) q[0];
u3(0.4920422528454055*pi,0.0*pi,4.0*pi) q[3];
cx q[3],q[6];
cx q[2],q[3];
u3(0.0*pi,-0.5*pi,1.0*pi) q[6];
u3(0.5*pi,-0.5*pi,0.5*pi) q[2];
u3(0.5*pi,0.0*pi,1.0*pi) q[3];
cx q[0],q[2];
u3(0.5*pi,0.0*pi,0.5*pi) q[0];
cx q[0],q[6];
u3(0.5*pi,0.0*pi,0.5*pi) q[0];
cx q[0],q[1];
u3(0.5*pi,-0.5*pi,0.5*pi) q[0];
u3(0.5*pi,-0.5*pi,1.0*pi) q[1];
cx q[1],q[3];
u3(0.5*pi,-0.5*pi,0.5*pi) q[1];
u3(0.5*pi,0.0*pi,1.0*pi) q[3];
cx q[1],q[7];
u3(2.5079577471545935*pi,0.0*pi,0.5*pi) q[1];
u3(0.0*pi,-0.5*pi,1.0*pi) q[7];
cx q[0],q[7];
cx q[0],q[3];
u3(0.5*pi,0.0*pi,1.0*pi) q[7];
u3(2.007957747154594*pi,-0.5*pi,1.5*pi) q[0];
u3(0.5*pi,-0.5*pi,1.0*pi) q[3];
cx q[3],q[7];
u3(1.0*pi,-0.5*pi,3.507957747154595*pi) q[7];
cx q[3],q[7];
u3(0.5*pi,0.0*pi,0.5*pi) q[3];
u3(0.5*pi,0.0*pi,1.0*pi) q[7];
cx q[0],q[3];
u3(3.007957747154595*pi,0.0*pi,0.5*pi) q[0];
u3(0.5*pi,-0.5*pi,1.0*pi) q[3];
cx q[0],q[7];
u3(0.5*pi,-0.5*pi,0.5*pi) q[0];
u3(1.9920422528454065*pi,1.0*pi,0.5*pi) q[7];
cx q[0],q[1];
u3(1.007957747154595*pi,0.0*pi,1.0*pi) q[1];
cx q[3],q[1];
u3(2.007957747154594*pi,0.0*pi,4.0*pi) q[1];
u3(0.5*pi,0.0*pi,0.5*pi) q[3];
cx q[0],q[1];
u3(0.5*pi,-0.5*pi,0.5*pi) q[0];
u3(1.5079577471545949*pi,0.0*pi,1.0*pi) q[1];
cx q[1],q[7];
u3(0.5*pi,-0.5*pi,0.5*pi) q[1];
u3(0.5*pi,0.0*pi,1.0*pi) q[7];
cx q[3],q[1];
cx q[0],q[3];
u3(0.5*pi,-0.5*pi,1.0*pi) q[1];
u3(0.5*pi,-0.5*pi,0.5*pi) q[0];
cx q[3],q[2];
cx q[0],q[6];
cx q[0],q[3];
cx q[6],q[7];
u3(0.5*pi,0.0*pi,0.5*pi) q[0];
cx q[1],q[6];
cx q[0],q[5];
u3(0.5*pi,0.0*pi,0.5*pi) q[1];
u3(1.0318309886183792*pi,-0.5*pi,4.0*pi) q[0];
cx q[1],q[8];
u3(1.5*pi,-0.5*pi,1.468169011381621*pi) q[5];
cx q[0],q[5];
u3(1.031830988618379*pi,-0.5*pi,4.0*pi) q[1];
u3(1.5*pi,-0.5*pi,1.468169011381621*pi) q[8];
u3(0.5*pi,0.0*pi,0.5*pi) q[0];
cx q[1],q[8];
cx q[0],q[3];
u3(0.5*pi,-0.5*pi,0.5*pi) q[1];
cx q[1],q[6];
u3(0.0*pi,-0.5*pi,1.0*pi) q[3];
cx q[3],q[2];
u3(0.0*pi,-0.5*pi,1.0*pi) q[6];
u3(0.0*pi,-0.5*pi,1.0*pi) q[2];
u3(0.5*pi,-0.5*pi,0.5*pi) q[3];
cx q[6],q[7];
cx q[3],q[1];
u3(0.5*pi,-0.5*pi,0.5*pi) q[6];
u3(0.0*pi,-0.5*pi,1.0*pi) q[7];
cx q[3],q[0];
u3(0.5*pi,0.0*pi,1.0*pi) q[1];
u3(0.5*pi,0.0*pi,1.0*pi) q[0];
u3(0.5*pi,0.0*pi,0.5*pi) q[3];
cx q[3],q[6];
u3(0.5079577471545939*pi,0.0*pi,1.0*pi) q[3];
u3(0.0*pi,-0.5*pi,1.0*pi) q[6];
cx q[0],q[6];
cx q[0],q[1];
u3(0.5*pi,0.0*pi,1.0*pi) q[6];
u3(1.992042252845406*pi,-0.5*pi,0.5*pi) q[0];
u3(0.5*pi,-0.5*pi,1.0*pi) q[1];
cx q[1],q[6];
u3(0.0*pi,-0.5*pi,1.5079577471545949*pi) q[6];
cx q[1],q[6];
u3(0.5*pi,0.0*pi,0.5*pi) q[1];
u3(0.5*pi,0.0*pi,1.0*pi) q[6];
cx q[0],q[1];
u3(3.007957747154595*pi,0.0*pi,0.5*pi) q[0];
u3(0.5*pi,-0.5*pi,1.0*pi) q[1];
cx q[0],q[6];
u3(0.5*pi,0.0*pi,0.5*pi) q[0];
u3(3.5*pi,-0.007957747154594408*pi,4.0*pi) q[6];
cx q[0],q[3];
u3(1.9920422528454051*pi,0.0*pi,4.0*pi) q[3];
cx q[1],q[3];
u3(0.5*pi,0.0*pi,0.5*pi) q[1];
u3(2.007957747154594*pi,0.0*pi,4.0*pi) q[3];
cx q[0],q[3];
u3(0.5*pi,0.0*pi,0.5*pi) q[0];
u3(2.007957747154595*pi,-0.5*pi,1.0*pi) q[3];
cx q[3],q[6];
u3(0.0*pi,-0.5*pi,1.0*pi) q[6];
cx q[1],q[6];
cx q[0],q[1];
cx q[6],q[7];
u3(0.5*pi,-0.5*pi,0.5*pi) q[0];
u3(0.0*pi,-0.5*pi,1.0*pi) q[1];
u3(1.968169011381621*pi,0.0*pi,1.0*pi) q[6];
u3(0.5*pi,-0.5*pi,1.468169011381621*pi) q[7];
cx q[0],q[2];
cx q[6],q[7];
u3(1.968169011381621*pi,0.0*pi,1.0*pi) q[0];
u3(0.5*pi,-0.5*pi,1.468169011381621*pi) q[2];
cx q[6],q[3];
cx q[0],q[2];
u3(0.5*pi,-0.5*pi,1.031830988618379*pi) q[3];
u3(1.531830988618379*pi,-0.5*pi,1.0*pi) q[6];
cx q[0],q[1];
cx q[6],q[3];
u3(1.531830988618379*pi,-0.5*pi,1.0*pi) q[0];
u3(0.5*pi,-0.5*pi,1.031830988618379*pi) q[1];
u3(0.5*pi,0.0*pi,0.5*pi) q[6];
cx q[0],q[1];
u3(0.5*pi,0.0*pi,0.5*pi) q[0];
