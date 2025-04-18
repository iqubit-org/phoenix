OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]
qreg q[46];

// Quantum gate operations
rz(-0.238732414637843*pi) q[0];
cx q[0], q[4];
rz(-0.07957747154594767*pi) q[4];
cx q[0], q[4];
cx q[0], q[4];
cx q[4], q[11];
rz(-0.07957747154594767*pi) q[11];
cx q[4], q[11];
cx q[0], q[4];
cx q[0], q[7];
rz(0.07957747154594767*pi) q[7];
cx q[0], q[7];
cx q[0], q[7];
cx q[7], q[35];
rz(0.07957747154594767*pi) q[35];
cx q[7], q[35];
cx q[0], q[7];
cx q[0], q[11];
rz(-0.15915494309189535*pi) q[11];
cx q[0], q[11];
cx q[0], q[11];
cx q[11], q[45];
rz(-0.07957747154594767*pi) q[45];
cx q[11], q[45];
cx q[0], q[11];
cx q[0], q[17];
rz(-0.15915494309189535*pi) q[17];
cx q[0], q[17];
cx q[0], q[21];
rz(0.15915494309189535*pi) q[21];
cx q[0], q[21];
cx q[0], q[31];
rz(0.15915494309189535*pi) q[31];
cx q[0], q[31];
cx q[0], q[35];
rz(0.07957747154594767*pi) q[35];
cx q[0], q[35];
cx q[0], q[37];
rz(0.15915494309189535*pi) q[37];
cx q[0], q[37];
cx q[0], q[40];
rz(-0.15915494309189535*pi) q[40];
cx q[0], q[40];
cx q[0], q[45];
rz(-0.07957747154594767*pi) q[45];
cx q[0], q[45];
cx q[1], q[6];
rz(0.07957747154594767*pi) q[6];
cx q[1], q[6];
cx q[1], q[6];
cx q[6], q[10];
rz(-0.07957747154594767*pi) q[10];
cx q[6], q[10];
cx q[1], q[6];
cx q[1], q[10];
rz(0.07957747154594767*pi) q[10];
cx q[1], q[10];
cx q[1], q[12];
rz(0.07957747154594767*pi) q[12];
cx q[1], q[12];
cx q[1], q[12];
cx q[12], q[18];
rz(0.07957747154594767*pi) q[18];
cx q[12], q[18];
cx q[1], q[12];
cx q[1], q[16];
rz(-0.07957747154594767*pi) q[16];
cx q[1], q[16];
cx q[1], q[16];
cx q[16], q[21];
rz(0.07957747154594767*pi) q[21];
cx q[16], q[21];
cx q[1], q[16];
cx q[1], q[18];
rz(-0.07957747154594767*pi) q[18];
cx q[1], q[18];
cx q[1], q[21];
cx q[21], q[29];
rz(-0.07957747154594767*pi) q[29];
cx q[21], q[29];
cx q[1], q[21];
cx q[1], q[29];
rz(0.07957747154594767*pi) q[29];
cx q[1], q[29];
cx q[1], q[40];
rz(-0.15915494309189535*pi) q[40];
cx q[1], q[40];
rz(-0.5570423008216338*pi) q[2];
cx q[2], q[4];
rz(-0.238732414637843*pi) q[4];
cx q[2], q[4];
cx q[2], q[4];
cx q[4], q[41];
rz(0.07957747154594767*pi) q[41];
cx q[4], q[41];
cx q[2], q[4];
cx q[2], q[19];
rz(0.07957747154594767*pi) q[19];
cx q[2], q[19];
cx q[2], q[19];
cx q[19], q[27];
rz(-0.07957747154594767*pi) q[27];
cx q[19], q[27];
cx q[2], q[19];
cx q[2], q[27];
rz(0.07957747154594767*pi) q[27];
cx q[2], q[27];
cx q[2], q[29];
rz(0.07957747154594767*pi) q[29];
cx q[2], q[29];
cx q[2], q[29];
cx q[29], q[37];
rz(0.07957747154594767*pi) q[37];
cx q[29], q[37];
cx q[2], q[29];
cx q[2], q[37];
rz(0.07957747154594767*pi) q[37];
cx q[2], q[37];
cx q[2], q[41];
rz(0.07957747154594767*pi) q[41];
cx q[2], q[41];
rz(-0.15915494309189535*pi) q[3];
cx q[3], q[8];
rz(-0.15915494309189535*pi) q[8];
cx q[3], q[8];
cx q[3], q[10];
rz(-0.07957747154594767*pi) q[10];
cx q[3], q[10];
cx q[3], q[10];
cx q[10], q[27];
rz(-0.07957747154594767*pi) q[27];
cx q[10], q[27];
cx q[3], q[10];
cx q[3], q[12];
rz(0.07957747154594767*pi) q[12];
cx q[3], q[12];
cx q[3], q[12];
cx q[12], q[20];
rz(0.07957747154594767*pi) q[20];
cx q[12], q[20];
cx q[3], q[12];
cx q[3], q[15];
rz(-0.07957747154594767*pi) q[15];
cx q[3], q[15];
cx q[3], q[15];
cx q[15], q[25];
rz(0.07957747154594767*pi) q[25];
cx q[15], q[25];
cx q[3], q[15];
cx q[3], q[20];
rz(-0.07957747154594767*pi) q[20];
cx q[3], q[20];
cx q[3], q[21];
rz(0.07957747154594767*pi) q[21];
cx q[3], q[21];
cx q[3], q[21];
cx q[21], q[31];
rz(0.07957747154594767*pi) q[31];
cx q[21], q[31];
cx q[3], q[21];
cx q[3], q[25];
rz(0.15915494309189535*pi) q[25];
cx q[3], q[25];
cx q[3], q[25];
cx q[25], q[36];
rz(-0.07957747154594767*pi) q[36];
cx q[25], q[36];
cx q[3], q[25];
cx q[3], q[26];
rz(-0.07957747154594767*pi) q[26];
cx q[3], q[26];
cx q[3], q[26];
cx q[26], q[31];
rz(0.07957747154594767*pi) q[31];
cx q[26], q[31];
cx q[3], q[26];
cx q[3], q[27];
rz(-0.07957747154594767*pi) q[27];
cx q[3], q[27];
cx q[3], q[36];
rz(0.07957747154594767*pi) q[36];
cx q[3], q[36];
cx q[3], q[44];
rz(0.15915494309189535*pi) q[44];
cx q[3], q[44];
rz(0.477464829275686*pi) q[4];
cx q[4], q[11];
rz(-0.07957747154594767*pi) q[11];
cx q[4], q[11];
cx q[4], q[31];
rz(-0.07957747154594767*pi) q[31];
cx q[4], q[31];
cx q[4], q[31];
cx q[31], q[35];
rz(0.07957747154594767*pi) q[35];
cx q[31], q[35];
cx q[4], q[31];
cx q[4], q[32];
rz(0.15915494309189535*pi) q[32];
cx q[4], q[32];
cx q[4], q[35];
rz(0.07957747154594767*pi) q[35];
cx q[4], q[35];
cx q[4], q[39];
rz(-0.07957747154594767*pi) q[39];
cx q[4], q[39];
cx q[4], q[39];
cx q[39], q[45];
rz(-0.07957747154594767*pi) q[45];
cx q[39], q[45];
cx q[4], q[39];
cx q[4], q[41];
rz(0.07957747154594767*pi) q[41];
cx q[4], q[41];
cx q[4], q[45];
rz(0.07957747154594767*pi) q[45];
cx q[4], q[45];
rz(-0.6366197723675814*pi) q[5];
cx q[5], q[7];
rz(-0.15915494309189535*pi) q[7];
cx q[5], q[7];
cx q[5], q[8];
rz(-0.15915494309189535*pi) q[8];
cx q[5], q[8];
cx q[5], q[15];
rz(-0.07957747154594767*pi) q[15];
cx q[5], q[15];
cx q[5], q[15];
cx q[15], q[28];
rz(0.07957747154594767*pi) q[28];
cx q[15], q[28];
cx q[5], q[15];
cx q[5], q[16];
rz(-0.07957747154594767*pi) q[16];
cx q[5], q[16];
cx q[5], q[16];
cx q[16], q[41];
rz(-0.07957747154594767*pi) q[41];
cx q[16], q[41];
cx q[5], q[16];
cx q[5], q[28];
rz(0.07957747154594767*pi) q[28];
cx q[5], q[28];
cx q[5], q[30];
rz(-0.15915494309189535*pi) q[30];
cx q[5], q[30];
cx q[5], q[41];
rz(0.07957747154594767*pi) q[41];
cx q[5], q[41];
cx q[5], q[43];
rz(0.15915494309189535*pi) q[43];
cx q[5], q[43];
rz(0.3978873577297384*pi) q[6];
cx q[6], q[10];
rz(-0.07957747154594767*pi) q[10];
cx q[6], q[10];
cx q[6], q[18];
rz(0.15915494309189535*pi) q[18];
cx q[6], q[18];
cx q[6], q[30];
rz(0.15915494309189535*pi) q[30];
cx q[6], q[30];
rz(-0.3978873577297384*pi) q[7];
cx q[7], q[9];
rz(-0.07957747154594767*pi) q[9];
cx q[7], q[9];
cx q[7], q[9];
cx q[9], q[40];
rz(-0.07957747154594767*pi) q[40];
cx q[9], q[40];
cx q[7], q[9];
cx q[7], q[11];
rz(-0.07957747154594767*pi) q[11];
cx q[7], q[11];
cx q[7], q[11];
cx q[11], q[28];
rz(0.07957747154594767*pi) q[28];
cx q[11], q[28];
cx q[7], q[11];
cx q[7], q[27];
rz(0.15915494309189535*pi) q[27];
cx q[7], q[27];
cx q[7], q[28];
rz(0.07957747154594767*pi) q[28];
cx q[7], q[28];
cx q[7], q[35];
rz(-0.07957747154594767*pi) q[35];
cx q[7], q[35];
cx q[7], q[40];
rz(0.07957747154594767*pi) q[40];
cx q[7], q[40];
rz(-0.3978873577297384*pi) q[8];
cx q[8], q[12];
rz(0.15915494309189535*pi) q[12];
cx q[8], q[12];
cx q[8], q[13];
rz(0.07957747154594767*pi) q[13];
cx q[8], q[13];
cx q[8], q[13];
cx q[13], q[23];
rz(0.07957747154594767*pi) q[23];
cx q[13], q[23];
cx q[8], q[13];
cx q[8], q[23];
rz(0.07957747154594767*pi) q[23];
cx q[8], q[23];
cx q[8], q[30];
rz(-0.15915494309189535*pi) q[30];
cx q[8], q[30];
cx q[8], q[42];
rz(0.15915494309189535*pi) q[42];
cx q[8], q[42];
rz(0.3978873577297384*pi) q[9];
cx q[9], q[12];
rz(0.07957747154594767*pi) q[12];
cx q[9], q[12];
cx q[9], q[12];
cx q[12], q[34];
rz(-0.07957747154594767*pi) q[34];
cx q[12], q[34];
cx q[9], q[12];
cx q[9], q[16];
rz(0.15915494309189535*pi) q[16];
cx q[9], q[16];
cx q[9], q[23];
rz(0.07957747154594767*pi) q[23];
cx q[9], q[23];
cx q[9], q[23];
cx q[23], q[27];
rz(-0.07957747154594767*pi) q[27];
cx q[23], q[27];
cx q[9], q[23];
cx q[9], q[27];
rz(0.07957747154594767*pi) q[27];
cx q[9], q[27];
cx q[9], q[34];
rz(-0.07957747154594767*pi) q[34];
cx q[9], q[34];
cx q[9], q[38];
rz(0.15915494309189535*pi) q[38];
cx q[9], q[38];
cx q[9], q[40];
rz(0.07957747154594767*pi) q[40];
cx q[9], q[40];
rz(-0.716197243913529*pi) q[10];
cx q[10], q[16];
rz(0.07957747154594767*pi) q[16];
cx q[10], q[16];
cx q[10], q[16];
cx q[16], q[21];
rz(-0.07957747154594767*pi) q[21];
cx q[16], q[21];
cx q[10], q[16];
cx q[10], q[21];
rz(-0.07957747154594767*pi) q[21];
cx q[10], q[21];
cx q[10], q[27];
rz(-0.07957747154594767*pi) q[27];
cx q[10], q[27];
cx q[10], q[29];
rz(0.15915494309189535*pi) q[29];
cx q[10], q[29];
cx q[10], q[31];
rz(0.07957747154594767*pi) q[31];
cx q[10], q[31];
cx q[10], q[31];
cx q[31], q[39];
rz(-0.07957747154594767*pi) q[39];
cx q[31], q[39];
cx q[10], q[31];
cx q[10], q[37];
rz(0.07957747154594767*pi) q[37];
cx q[10], q[37];
cx q[10], q[37];
cx q[37], q[39];
rz(-0.07957747154594767*pi) q[39];
cx q[37], q[39];
cx q[10], q[37];
cx q[10], q[39];
rz(0.15915494309189535*pi) q[39];
cx q[10], q[39];
cx q[10], q[41];
rz(-0.15915494309189535*pi) q[41];
cx q[10], q[41];
rz(0.07957747154594767*pi) q[11];
cx q[11], q[13];
rz(-0.07957747154594767*pi) q[13];
cx q[11], q[13];
cx q[11], q[13];
cx q[13], q[27];
rz(-0.07957747154594767*pi) q[27];
cx q[13], q[27];
cx q[11], q[13];
cx q[11], q[14];
rz(0.15915494309189535*pi) q[14];
cx q[11], q[14];
cx q[11], q[15];
rz(-0.07957747154594767*pi) q[15];
cx q[11], q[15];
cx q[11], q[15];
cx q[15], q[45];
rz(0.07957747154594767*pi) q[45];
cx q[15], q[45];
cx q[11], q[15];
cx q[11], q[27];
rz(-0.07957747154594767*pi) q[27];
cx q[11], q[27];
cx q[11], q[28];
rz(0.07957747154594767*pi) q[28];
cx q[11], q[28];
rz(0.8753521870054244*pi) q[12];
cx q[12], q[18];
rz(0.07957747154594767*pi) q[18];
cx q[12], q[18];
cx q[12], q[20];
rz(0.07957747154594767*pi) q[20];
cx q[12], q[20];
cx q[12], q[22];
rz(-0.15915494309189535*pi) q[22];
cx q[12], q[22];
cx q[12], q[23];
rz(-0.15915494309189535*pi) q[23];
cx q[12], q[23];
cx q[12], q[34];
rz(0.07957747154594767*pi) q[34];
cx q[12], q[34];
rz(0.15915494309189535*pi) q[13];
cx q[13], q[20];
rz(0.07957747154594767*pi) q[20];
cx q[13], q[20];
cx q[13], q[20];
cx q[20], q[30];
rz(0.07957747154594767*pi) q[30];
cx q[20], q[30];
cx q[13], q[20];
cx q[13], q[23];
rz(-0.07957747154594767*pi) q[23];
cx q[13], q[23];
cx q[13], q[27];
rz(-0.07957747154594767*pi) q[27];
cx q[13], q[27];
cx q[13], q[30];
rz(-0.07957747154594767*pi) q[30];
cx q[13], q[30];
cx q[13], q[34];
rz(-0.07957747154594767*pi) q[34];
cx q[13], q[34];
cx q[13], q[34];
cx q[34], q[41];
rz(0.07957747154594767*pi) q[41];
cx q[34], q[41];
cx q[13], q[34];
cx q[13], q[41];
rz(-0.07957747154594767*pi) q[41];
cx q[13], q[41];
rz(0.07957747154594767*pi) q[14];
cx q[14], q[17];
rz(0.07957747154594767*pi) q[17];
cx q[14], q[17];
cx q[14], q[17];
cx q[17], q[24];
rz(0.07957747154594767*pi) q[24];
cx q[17], q[24];
cx q[14], q[17];
cx q[14], q[20];
rz(0.15915494309189535*pi) q[20];
cx q[14], q[20];
cx q[14], q[24];
rz(0.07957747154594767*pi) q[24];
cx q[14], q[24];
rz(-0.3978873577297384*pi) q[15];
cx q[15], q[18];
rz(0.07957747154594767*pi) q[18];
cx q[15], q[18];
cx q[15], q[18];
cx q[18], q[43];
rz(0.07957747154594767*pi) q[43];
cx q[18], q[43];
cx q[15], q[18];
cx q[15], q[20];
rz(-0.07957747154594767*pi) q[20];
cx q[15], q[20];
cx q[15], q[20];
cx q[20], q[34];
rz(-0.07957747154594767*pi) q[34];
cx q[20], q[34];
cx q[15], q[20];
cx q[15], q[25];
rz(0.07957747154594767*pi) q[25];
cx q[15], q[25];
cx q[15], q[28];
rz(0.07957747154594767*pi) q[28];
cx q[15], q[28];
cx q[15], q[34];
rz(0.07957747154594767*pi) q[34];
cx q[15], q[34];
cx q[15], q[36];
rz(-0.15915494309189535*pi) q[36];
cx q[15], q[36];
cx q[15], q[39];
rz(-0.15915494309189535*pi) q[39];
cx q[15], q[39];
cx q[15], q[43];
rz(0.07957747154594767*pi) q[43];
cx q[15], q[43];
cx q[15], q[45];
rz(0.07957747154594767*pi) q[45];
cx q[15], q[45];
rz(-0.3978873577297384*pi) q[16];
cx q[16], q[21];
rz(0.15915494309189535*pi) q[21];
cx q[16], q[21];
cx q[16], q[41];
rz(0.07957747154594767*pi) q[41];
cx q[16], q[41];
cx q[16], q[42];
rz(0.15915494309189535*pi) q[42];
cx q[16], q[42];
rz(0.15915494309189535*pi) q[17];
cx q[17], q[20];
rz(-0.07957747154594767*pi) q[20];
cx q[17], q[20];
cx q[17], q[20];
cx q[20], q[45];
rz(-0.07957747154594767*pi) q[45];
cx q[20], q[45];
cx q[17], q[20];
cx q[17], q[21];
rz(0.07957747154594767*pi) q[21];
cx q[17], q[21];
cx q[17], q[21];
cx q[21], q[25];
rz(0.07957747154594767*pi) q[25];
cx q[21], q[25];
cx q[17], q[21];
cx q[17], q[24];
rz(-0.07957747154594767*pi) q[24];
cx q[17], q[24];
cx q[17], q[25];
rz(0.07957747154594767*pi) q[25];
cx q[17], q[25];
cx q[17], q[31];
rz(0.07957747154594767*pi) q[31];
cx q[17], q[31];
cx q[17], q[31];
cx q[31], q[41];
rz(-0.07957747154594767*pi) q[41];
cx q[31], q[41];
cx q[17], q[31];
cx q[17], q[41];
rz(-0.07957747154594767*pi) q[41];
cx q[17], q[41];
cx q[17], q[45];
rz(-0.07957747154594767*pi) q[45];
cx q[17], q[45];
rz(-0.6366197723675814*pi) q[18];
cx q[18], q[43];
rz(-0.07957747154594767*pi) q[43];
cx q[18], q[43];
rz(-0.07957747154594767*pi) q[19];
cx q[19], q[27];
rz(-0.07957747154594767*pi) q[27];
cx q[19], q[27];
cx q[19], q[44];
rz(-0.15915494309189535*pi) q[44];
cx q[19], q[44];
rz(-0.6366197723675814*pi) q[20];
cx q[20], q[23];
rz(-0.07957747154594767*pi) q[23];
cx q[20], q[23];
cx q[20], q[23];
cx q[23], q[35];
rz(0.07957747154594767*pi) q[35];
cx q[23], q[35];
cx q[20], q[23];
cx q[20], q[29];
rz(0.15915494309189535*pi) q[29];
cx q[20], q[29];
cx q[20], q[30];
rz(0.07957747154594767*pi) q[30];
cx q[20], q[30];
cx q[20], q[34];
rz(0.07957747154594767*pi) q[34];
cx q[20], q[34];
cx q[20], q[35];
rz(0.07957747154594767*pi) q[35];
cx q[20], q[35];
cx q[20], q[36];
rz(0.07957747154594767*pi) q[36];
cx q[20], q[36];
cx q[20], q[36];
cx q[36], q[40];
rz(0.07957747154594767*pi) q[40];
cx q[36], q[40];
cx q[20], q[36];
cx q[20], q[40];
rz(0.07957747154594767*pi) q[40];
cx q[20], q[40];
cx q[20], q[41];
rz(0.15915494309189535*pi) q[41];
cx q[20], q[41];
cx q[20], q[45];
rz(-0.238732414637843*pi) q[45];
cx q[20], q[45];
rz(-0.15915494309189535*pi) q[21];
cx q[21], q[25];
rz(-0.07957747154594767*pi) q[25];
cx q[21], q[25];
cx q[21], q[29];
rz(0.07957747154594767*pi) q[29];
cx q[21], q[29];
cx q[21], q[31];
rz(0.07957747154594767*pi) q[31];
cx q[21], q[31];
cx q[21], q[32];
rz(-0.07957747154594767*pi) q[32];
cx q[21], q[32];
cx q[21], q[32];
cx q[32], q[35];
rz(-0.07957747154594767*pi) q[35];
cx q[32], q[35];
cx q[21], q[32];
cx q[21], q[35];
rz(0.07957747154594767*pi) q[35];
cx q[21], q[35];
rz(0.07957747154594767*pi) q[22];
cx q[22], q[27];
rz(0.07957747154594767*pi) q[27];
cx q[22], q[27];
cx q[22], q[27];
cx q[27], q[45];
rz(0.07957747154594767*pi) q[45];
cx q[27], q[45];
cx q[22], q[27];
cx q[22], q[45];
rz(-0.07957747154594767*pi) q[45];
cx q[22], q[45];
rz(-0.07957747154594767*pi) q[23];
cx q[23], q[25];
rz(-0.07957747154594767*pi) q[25];
cx q[23], q[25];
cx q[23], q[25];
cx q[25], q[43];
rz(0.07957747154594767*pi) q[43];
cx q[25], q[43];
cx q[23], q[25];
cx q[23], q[27];
rz(-0.07957747154594767*pi) q[27];
cx q[23], q[27];
cx q[23], q[33];
rz(0.15915494309189535*pi) q[33];
cx q[23], q[33];
cx q[23], q[35];
rz(0.07957747154594767*pi) q[35];
cx q[23], q[35];
cx q[23], q[37];
rz(0.07957747154594767*pi) q[37];
cx q[23], q[37];
cx q[23], q[37];
cx q[37], q[43];
rz(0.07957747154594767*pi) q[43];
cx q[37], q[43];
cx q[23], q[37];
cx q[23], q[43];
rz(-0.15915494309189535*pi) q[43];
cx q[23], q[43];
rz(0.07957747154594767*pi) q[24];
cx q[24], q[28];
rz(-0.15915494309189535*pi) q[28];
cx q[24], q[28];
rz(0.954929658551372*pi) q[25];
cx q[25], q[33];
rz(-0.15915494309189535*pi) q[33];
cx q[25], q[33];
cx q[25], q[34];
rz(-0.15915494309189535*pi) q[34];
cx q[25], q[34];
cx q[25], q[36];
rz(-0.07957747154594767*pi) q[36];
cx q[25], q[36];
cx q[25], q[42];
rz(-0.15915494309189535*pi) q[42];
cx q[25], q[42];
cx q[25], q[43];
rz(-0.07957747154594767*pi) q[43];
cx q[25], q[43];
rz(-0.238732414637843*pi) q[26];
cx q[26], q[31];
rz(0.238732414637843*pi) q[31];
cx q[26], q[31];
cx q[26], q[44];
rz(-0.15915494309189535*pi) q[44];
cx q[26], q[44];
cx q[26], q[45];
rz(0.15915494309189535*pi) q[45];
cx q[26], q[45];
rz(0.15915494309189535*pi) q[27];
cx q[27], q[35];
rz(-0.07957747154594767*pi) q[35];
cx q[27], q[35];
cx q[27], q[35];
cx q[35], q[36];
rz(-0.07957747154594767*pi) q[36];
cx q[35], q[36];
cx q[27], q[35];
cx q[27], q[36];
rz(-0.07957747154594767*pi) q[36];
cx q[27], q[36];
cx q[27], q[45];
rz(0.07957747154594767*pi) q[45];
cx q[27], q[45];
rz(0.477464829275686*pi) q[28];
cx q[28], q[34];
rz(0.07957747154594767*pi) q[34];
cx q[28], q[34];
cx q[28], q[34];
cx q[34], q[35];
rz(0.07957747154594767*pi) q[35];
cx q[34], q[35];
cx q[28], q[34];
cx q[28], q[35];
rz(-0.07957747154594767*pi) q[35];
cx q[28], q[35];
cx q[28], q[38];
rz(-0.07957747154594767*pi) q[38];
cx q[28], q[38];
cx q[28], q[38];
cx q[38], q[42];
rz(-0.07957747154594767*pi) q[42];
cx q[38], q[42];
cx q[28], q[38];
cx q[28], q[42];
rz(-0.07957747154594767*pi) q[42];
cx q[28], q[42];
rz(0.15915494309189535*pi) q[29];
cx q[29], q[37];
rz(-0.07957747154594767*pi) q[37];
cx q[29], q[37];
rz(0.3978873577297384*pi) q[30];
rz(0.238732414637843*pi) q[31];
cx q[31], q[35];
rz(0.07957747154594767*pi) q[35];
cx q[31], q[35];
cx q[31], q[39];
rz(-0.07957747154594767*pi) q[39];
cx q[31], q[39];
cx q[31], q[41];
rz(0.07957747154594767*pi) q[41];
cx q[31], q[41];
rz(-0.238732414637843*pi) q[32];
cx q[32], q[35];
rz(-0.07957747154594767*pi) q[35];
cx q[32], q[35];
cx q[32], q[38];
rz(0.3183098861837907*pi) q[38];
cx q[32], q[38];
rz(-0.3978873577297384*pi) q[33];
cx q[33], q[34];
rz(-0.07957747154594767*pi) q[34];
cx q[33], q[34];
cx q[33], q[34];
cx q[34], q[42];
rz(-0.07957747154594767*pi) q[42];
cx q[34], q[42];
cx q[33], q[34];
cx q[33], q[42];
rz(-0.07957747154594767*pi) q[42];
cx q[33], q[42];
rz(0.5570423008216338*pi) q[34];
cx q[34], q[35];
rz(0.07957747154594767*pi) q[35];
cx q[34], q[35];
cx q[34], q[41];
rz(-0.07957747154594767*pi) q[41];
cx q[34], q[41];
cx q[34], q[42];
rz(-0.07957747154594767*pi) q[42];
cx q[34], q[42];
rz(-0.7957747154594768*pi) q[35];
cx q[35], q[36];
rz(-0.238732414637843*pi) q[36];
cx q[35], q[36];
rz(0.3978873577297384*pi) q[36];
cx q[36], q[40];
rz(-0.07957747154594767*pi) q[40];
cx q[36], q[40];
cx q[36], q[45];
rz(0.15915494309189535*pi) q[45];
cx q[36], q[45];
rz(-0.07957747154594767*pi) q[37];
cx q[37], q[39];
rz(-0.07957747154594767*pi) q[39];
cx q[37], q[39];
cx q[37], q[43];
rz(0.07957747154594767*pi) q[43];
cx q[37], q[43];
rz(-0.5570423008216338*pi) q[38];
cx q[38], q[42];
rz(-0.07957747154594767*pi) q[42];
cx q[38], q[42];
rz(0.07957747154594767*pi) q[39];
cx q[39], q[41];
rz(-0.15915494309189535*pi) q[41];
cx q[39], q[41];
cx q[39], q[44];
rz(0.15915494309189535*pi) q[44];
cx q[39], q[44];
cx q[39], q[45];
rz(0.07957747154594767*pi) q[45];
cx q[39], q[45];
rz(-0.477464829275686*pi) q[40];
rz(-0.15915494309189535*pi) q[41];
cx q[41], q[44];
rz(0.15915494309189535*pi) q[44];
cx q[41], q[44];
rz(-0.3183098861837907*pi) q[42];
rz(-0.238732414637843*pi) q[43];
rz(-0.15915494309189535*pi) q[44];
rz(0.238732414637843*pi) q[45];
