OPENQASM 2.0;
include "qelib1.inc";
qreg q[24];
cx q[3],q[4];
cx q[2],q[4];
cx q[1],q[4];
cx q[0],q[4];
u1(1.0) q[4];
cx q[0],q[4];
cx q[1],q[4];
cx q[2],q[4];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[3];
cx q[0],q[3];
u1(1.0) q[3];
cx q[1],q[3];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[2];
u1(1.0) q[2];
cx q[0],q[2];
cx q[1],q[2];
cx q[0],q[1];
u1(1.0) q[1];
cx q[0],q[1];
cx q[1],q[3];
u1(1.0) q[3];
cx q[0],q[3];
u1(1.0) q[3];
cx q[1],q[3];
u1(1.0) q[1];
cx q[1],q[2];
u1(1.0) q[2];
cx q[1],q[2];
cx q[2],q[5];
cx q[1],q[5];
u1(1.0) q[5];
cx q[2],q[5];
u1(1.0) q[5];
cx q[1],q[5];
u1(1.0) q[5];
cx q[4],q[5];
u1(1.0) q[5];
cx q[3],q[5];
u1(1.0) q[5];
cx q[4],q[5];
u1(1.0) q[5];
cx q[2],q[5];
u1(1.0) q[5];
cx q[2],q[5];
cx q[3],q[5];
cx q[2],q[3];
u1(1.0) q[3];
cx q[2],q[3];
u1(1.0) q[2];
cx q[0],q[2];
u1(1.0) q[2];
cx q[0],q[2];
cx q[2],q[3];
cx q[0],q[3];
u1(1.0) q[3];
cx q[2],q[3];
u1(1.0) q[3];
cx q[0],q[3];
u1(1.0) q[0];
cx q[0],q[4];
u1(1.0) q[4];
cx q[1],q[4];
u1(1.0) q[4];
cx q[0],q[4];
u1(1.0) q[4];
cx q[1],q[4];
u1(1.0) q[4];
cx q[4],q[6];
u1(1.0) q[6];
cx q[3],q[6];
u1(1.0) q[6];
cx q[4],q[6];
u1(1.0) q[6];
cx q[3],q[6];
u1(1.0) q[3];
cx q[3],q[7];
u1(1.0) q[7];
cx q[4],q[7];
u1(1.0) q[7];
cx q[3],q[7];
u1(1.0) q[7];
cx q[4],q[7];
u1(1.0) q[7];
cx q[6],q[7];
u1(1.0) q[7];
cx q[6],q[7];
cx q[7],q[8];
cx q[6],q[8];
u1(1.0) q[8];
cx q[6],q[8];
u1(1.0) q[8];
cx q[5],q[8];
u1(1.0) q[8];
cx q[5],q[8];
cx q[7],q[8];
cx q[5],q[7];
u1(1.0) q[7];
cx q[6],q[7];
u1(1.0) q[7];
cx q[5],q[7];
cx q[6],q[7];
cx q[5],q[6];
u1(1.0) q[6];
cx q[5],q[6];
u1(1.0) q[6];
cx q[6],q[8];
u1(1.0) q[8];
cx q[5],q[8];
u1(1.0) q[8];
cx q[6],q[8];
u1(1.0) q[8];
cx q[5],q[8];
u1(1.0) q[8];
cx q[8],q[9];
u1(1.0) q[9];
cx q[8],q[9];
cx q[9],q[11];
cx q[8],q[11];
u1(1.0) q[11];
cx q[8],q[11];
u1(1.0) q[11];
cx q[10],q[11];
u1(1.0) q[11];
cx q[9],q[11];
u1(1.0) q[11];
cx q[8],q[11];
u1(1.0) q[11];
cx q[8],q[11];
cx q[10],q[11];
cx q[8],q[10];
u1(1.0) q[10];
cx q[8],q[10];
u1(1.0) q[10];
cx q[9],q[10];
u1(1.0) q[10];
cx q[8],q[10];
u1(1.0) q[10];
cx q[8],q[10];
cx q[9],q[10];
cx q[10],q[12];
cx q[9],q[12];
u1(1.0) q[12];
cx q[10],q[12];
u1(1.0) q[12];
cx q[9],q[12];
u1(1.0) q[9];
cx q[9],q[13];
u1(1.0) q[13];
cx q[10],q[13];
u1(1.0) q[13];
cx q[9],q[13];
u1(1.0) q[13];
cx q[10],q[13];
u1(1.0) q[13];
cx q[12],q[13];
u1(1.0) q[13];
cx q[12],q[13];
cx q[13],q[14];
cx q[12],q[14];
u1(1.0) q[14];
cx q[13],q[14];
u1(1.0) q[14];
cx q[11],q[14];
u1(1.0) q[14];
cx q[12],q[14];
u1(1.0) q[14];
cx q[11],q[14];
u1(1.0) q[11];
cx q[8],q[11];
u1(1.0) q[11];
cx q[6],q[11];
u1(1.0) q[11];
cx q[8],q[11];
u1(1.0) q[11];
cx q[7],q[11];
u1(1.0) q[11];
cx q[6],q[11];
u1(1.0) q[11];
cx q[10],q[11];
u1(1.0) q[11];
cx q[10],q[11];
cx q[7],q[10];
u1(1.0) q[10];
cx q[6],q[10];
u1(1.0) q[10];
cx q[7],q[10];
u1(1.0) q[10];
cx q[8],q[10];
u1(1.0) q[10];
cx q[6],q[10];
cx q[8],q[10];
cx q[8],q[9];
cx q[6],q[9];
u1(1.0) q[9];
cx q[8],q[9];
u1(1.0) q[9];
cx q[7],q[9];
u1(1.0) q[9];
cx q[6],q[9];
u1(1.0) q[9];
cx q[7],q[9];
cx q[9],q[11];
u1(1.0) q[11];
cx q[7],q[11];
cx q[9],q[11];
cx q[9],q[10];
cx q[7],q[10];
u1(1.0) q[10];
cx q[7],q[10];
cx q[9],q[10];
cx q[10],q[14];
cx q[9],q[14];
u1(1.0) q[14];
cx q[9],q[14];
u1(1.0) q[14];
cx q[10],q[14];
u1(1.0) q[14];
cx q[14],q[15];
u1(1.0) q[15];
cx q[14],q[15];
cx q[15],q[17];
cx q[14],q[17];
u1(1.0) q[17];
cx q[14],q[17];
u1(1.0) q[17];
cx q[16],q[17];
u1(1.0) q[17];
cx q[15],q[17];
u1(1.0) q[17];
cx q[14],q[17];
u1(1.0) q[17];
cx q[14],q[17];
cx q[16],q[17];
cx q[14],q[16];
u1(1.0) q[16];
cx q[14],q[16];
u1(1.0) q[16];
cx q[16],q[18];
u1(1.0) q[18];
cx q[15],q[18];
u1(1.0) q[18];
cx q[15],q[18];
cx q[16],q[18];
cx q[15],q[16];
u1(1.0) q[16];
cx q[14],q[16];
u1(1.0) q[16];
cx q[14],q[16];
cx q[15],q[16];
cx q[16],q[19];
cx q[15],q[19];
u1(1.0) q[19];
cx q[15],q[19];
u1(1.0) q[19];
cx q[16],q[19];
u1(1.0) q[19];
cx q[18],q[19];
u1(1.0) q[19];
cx q[18],q[19];
cx q[19],q[20];
cx q[18],q[20];
u1(1.0) q[20];
cx q[19],q[20];
u1(1.0) q[20];
cx q[17],q[20];
u1(1.0) q[20];
cx q[18],q[20];
cx q[17],q[18];
u1(1.0) q[18];
cx q[17],q[18];
u1(1.0) q[17];
cx q[17],q[19];
u1(1.0) q[19];
cx q[17],q[19];
cx q[19],q[20];
u1(1.0) q[20];
cx q[19],q[20];
u1(1.0) q[20];
cx q[17],q[20];
u1(1.0) q[20];
cx q[20],q[21];
u1(1.0) q[21];
cx q[20],q[21];
cx q[21],q[23];
cx q[20],q[23];
u1(1.0) q[23];
cx q[20],q[23];
u1(1.0) q[23];
cx q[22],q[23];
u1(1.0) q[23];
cx q[21],q[23];
u1(1.0) q[23];
cx q[20],q[23];
u1(1.0) q[23];
cx q[20],q[23];
cx q[22],q[23];
cx q[20],q[22];
u1(1.0) q[22];
cx q[20],q[22];
u1(1.0) q[22];
cx q[21],q[22];
u1(1.0) q[22];
cx q[20],q[22];
u1(1.0) q[22];
cx q[20],q[22];
cx q[2],q[22];
u1(1.0) q[22];
cx q[2],q[22];
cx q[21],q[22];
cx q[2],q[21];
u1(1.0) q[21];
cx q[2],q[21];
u1(1.0) q[21];
cx q[19],q[21];
u1(1.0) q[21];
cx q[18],q[21];
u1(1.0) q[21];
cx q[19],q[21];
u1(1.0) q[21];
cx q[18],q[21];
u1(1.0) q[18];
cx q[18],q[22];
u1(1.0) q[22];
cx q[19],q[22];
u1(1.0) q[22];
cx q[18],q[22];
u1(1.0) q[22];
cx q[19],q[22];
cx q[22],q[23];
cx q[19],q[23];
u1(1.0) q[23];
cx q[22],q[23];
u1(1.0) q[23];
cx q[19],q[23];
u1(1.0) q[23];
cx q[20],q[23];
u1(1.0) q[23];
cx q[18],q[23];
u1(1.0) q[23];
cx q[20],q[23];
u1(1.0) q[23];
cx q[19],q[23];
u1(1.0) q[23];
cx q[18],q[23];
cx q[19],q[23];
cx q[18],q[19];
cx q[17],q[19];
u1(1.0) q[19];
cx q[17],q[19];
cx q[18],q[19];
cx q[17],q[18];
cx q[15],q[18];
u1(1.0) q[18];
cx q[17],q[18];
u1(1.0) q[18];
cx q[15],q[18];
u1(1.0) q[15];
cx q[13],q[15];
u1(1.0) q[15];
cx q[12],q[15];
u1(1.0) q[15];
cx q[13],q[15];
u1(1.0) q[15];
cx q[12],q[15];
u1(1.0) q[12];
cx q[10],q[12];
u1(1.0) q[12];
cx q[10],q[12];
cx q[12],q[14];
cx q[10],q[14];
u1(1.0) q[14];
cx q[10],q[14];
cx q[12],q[14];
cx q[14],q[15];
cx q[12],q[15];
u1(1.0) q[15];
cx q[12],q[15];
cx q[14],q[15];
cx q[14],q[16];
cx q[12],q[16];
u1(1.0) q[16];
cx q[14],q[16];
u1(1.0) q[16];
cx q[13],q[16];
u1(1.0) q[16];
cx q[12],q[16];
u1(1.0) q[16];
cx q[13],q[16];
cx q[16],q[17];
cx q[13],q[17];
u1(1.0) q[17];
cx q[16],q[17];
u1(1.0) q[17];
cx q[12],q[17];
u1(1.0) q[17];
cx q[13],q[17];
u1(1.0) q[17];
cx q[14],q[17];
u1(1.0) q[17];
cx q[12],q[17];
u1(1.0) q[17];
cx q[13],q[17];
u1(1.0) q[17];
cx q[13],q[17];
cx q[14],q[17];
cx q[13],q[14];
u1(1.0) q[14];
cx q[11],q[14];
u1(1.0) q[14];
cx q[13],q[14];
cx q[11],q[13];
u1(1.0) q[13];
cx q[12],q[13];
u1(1.0) q[13];
cx q[12],q[13];
cx q[11],q[12];
u1(1.0) q[12];
cx q[9],q[12];
u1(1.0) q[12];
cx q[9],q[12];
cx q[11],q[12];
cx q[9],q[13];
u1(1.0) q[13];
cx q[11],q[13];
cx q[9],q[14];
u1(1.0) q[14];
cx q[11],q[14];
u1(1.0) q[14];
cx q[12],q[14];
u1(1.0) q[14];
cx q[9],q[14];
cx q[12],q[14];
cx q[12],q[13];
u1(1.0) q[13];
cx q[9],q[13];
cx q[10],q[13];
u1(1.0) q[13];
cx q[10],q[13];
cx q[12],q[13];
cx q[13],q[14];
cx q[10],q[14];
u1(1.0) q[14];
cx q[10],q[14];
cx q[13],q[14];
cx q[14],q[16];
cx q[13],q[16];
u1(1.0) q[16];
cx q[14],q[16];
cx q[15],q[16];
u1(1.0) q[16];
cx q[13],q[16];
cx q[15],q[16];
cx q[16],q[20];
cx q[15],q[20];
u1(1.0) q[20];
cx q[15],q[20];
u1(1.0) q[20];
cx q[19],q[20];
u1(1.0) q[20];
cx q[16],q[20];
u1(1.0) q[20];
cx q[19],q[20];
cx q[20],q[23];
cx q[19],q[23];
u1(1.0) q[23];
cx q[20],q[23];
cx q[21],q[23];
u1(1.0) q[23];
cx q[19],q[23];
cx q[2],q[23];
u1(1.0) q[23];
cx q[21],q[23];
u1(1.0) q[23];
cx q[22],q[23];
u1(1.0) q[23];
cx q[2],q[23];
cx q[22],q[23];
cx q[2],q[22];
u1(1.0) q[22];
cx q[0],q[22];
u1(1.0) q[22];
cx q[2],q[22];
u1(1.0) q[22];
cx q[1],q[22];
u1(1.0) q[22];
cx q[0],q[22];
u1(1.0) q[22];
cx q[1],q[22];
cx q[22],q[23];
cx q[1],q[23];
u1(1.0) q[23];
cx q[22],q[23];
u1(1.0) q[23];
cx q[0],q[23];
u1(1.0) q[23];
cx q[1],q[23];
u1(1.0) q[23];
cx q[2],q[23];
u1(1.0) q[23];
cx q[0],q[23];
cx q[2],q[23];
cx q[2],q[4];
cx q[0],q[4];
u1(1.0) q[4];
cx q[0],q[4];
u1(1.0) q[4];
cx q[2],q[4];
cx q[4],q[5];
cx q[2],q[5];
u1(1.0) q[5];
cx q[4],q[5];
u1(1.0) q[5];
cx q[0],q[5];
u1(1.0) q[5];
cx q[2],q[5];
u1(1.0) q[5];
cx q[1],q[5];
u1(1.0) q[5];
cx q[0],q[5];
cx q[1],q[5];
cx q[1],q[21];
cx q[0],q[21];
u1(1.0) q[21];
cx q[0],q[21];
u1(1.0) q[21];
cx q[1],q[21];
cx q[21],q[23];
cx q[1],q[23];
u1(1.0) q[23];
cx q[1],q[23];
cx q[21],q[23];
cx q[21],q[22];
cx q[1],q[22];
u1(1.0) q[22];
cx q[1],q[22];
cx q[19],q[22];
u1(1.0) q[22];
cx q[19],q[22];
cx q[18],q[22];
u1(1.0) q[22];
cx q[21],q[22];
cx q[20],q[21];
cx q[18],q[21];
u1(1.0) q[21];
cx q[18],q[21];
cx q[20],q[21];
cx q[20],q[22];
u1(1.0) q[22];
cx q[18],q[22];
cx q[20],q[22];
cx q[18],q[20];
cx q[16],q[20];
u1(1.0) q[20];
cx q[16],q[20];
cx q[18],q[20];
cx q[18],q[19];
cx q[16],q[19];
u1(1.0) q[19];
cx q[16],q[19];
cx q[15],q[19];
u1(1.0) q[19];
cx q[18],q[19];
u1(1.0) q[19];
cx q[17],q[19];
u1(1.0) q[19];
cx q[15],q[19];
cx q[17],q[19];
cx q[15],q[17];
cx q[13],q[17];
u1(1.0) q[17];
cx q[13],q[17];
cx q[15],q[17];
cx q[17],q[20];
cx q[15],q[20];
u1(1.0) q[20];
cx q[17],q[20];
u1(1.0) q[20];
cx q[18],q[20];
u1(1.0) q[20];
cx q[18],q[20];
cx q[19],q[20];
u1(1.0) q[20];
cx q[15],q[20];
cx q[19],q[20];
cx q[20],q[22];
cx q[19],q[22];
u1(1.0) q[22];
cx q[19],q[22];
cx q[20],q[22];
cx q[20],q[21];
cx q[19],q[21];
u1(1.0) q[21];
cx q[18],q[21];
u1(1.0) q[21];
cx q[18],q[21];
cx q[19],q[21];
cx q[20],q[21];
cx q[19],q[20];
cx q[18],q[20];
cx q[17],q[20];
u1(1.0) q[20];
cx q[17],q[20];
cx q[18],q[20];
cx q[19],q[20];
cx q[18],q[19];
cx q[17],q[19];
cx q[16],q[19];
u1(1.0) q[19];
cx q[18],q[19];
u1(1.0) q[19];
cx q[16],q[19];
cx q[17],q[19];
cx q[16],q[17];
cx q[12],q[17];
u1(1.0) q[17];
cx q[16],q[17];
cx q[15],q[16];
cx q[12],q[16];
u1(1.0) q[16];
cx q[12],q[16];
cx q[15],q[16];
cx q[15],q[17];
u1(1.0) q[17];
cx q[13],q[17];
u1(1.0) q[17];
cx q[12],q[17];
cx q[13],q[17];
cx q[15],q[17];
cx q[14],q[15];
cx q[13],q[15];
cx q[12],q[15];
u1(1.0) q[15];
cx q[12],q[15];
u1(1.0) q[15];
cx q[13],q[15];
cx q[14],q[15];
cx q[13],q[14];
cx q[9],q[14];
u1(1.0) q[14];
cx q[11],q[14];
u1(1.0) q[14];
cx q[9],q[14];
cx q[11],q[14];
cx q[13],q[14];
cx q[11],q[13];
cx q[10],q[13];
cx q[9],q[13];
u1(1.0) q[13];
cx q[9],q[13];
u1(1.0) q[13];
cx q[10],q[13];
cx q[11],q[13];
cx q[10],q[11];
cx q[6],q[11];
u1(1.0) q[11];
cx q[10],q[11];
cx q[9],q[10];
cx q[6],q[10];
u1(1.0) q[10];
cx q[6],q[10];
cx q[9],q[10];
cx q[9],q[11];
u1(1.0) q[11];
cx q[10],q[11];
u1(1.0) q[11];
cx q[6],q[11];
cx q[8],q[11];
u1(1.0) q[11];
cx q[8],q[11];
cx q[9],q[11];
cx q[10],q[11];
cx q[11],q[12];
cx q[10],q[12];
cx q[9],q[12];
u1(1.0) q[12];
cx q[9],q[12];
u1(1.0) q[12];
cx q[10],q[12];
cx q[11],q[12];
cx q[11],q[14];
cx q[10],q[14];
u1(1.0) q[14];
cx q[12],q[13];
cx q[9],q[14];
u1(1.0) q[14];
cx q[9],q[14];
cx q[10],q[14];
cx q[11],q[14];
cx q[10],q[11];
cx q[9],q[11];
cx q[7],q[11];
u1(1.0) q[11];
cx q[7],q[11];
cx q[9],q[11];
cx q[10],q[11];
cx q[9],q[10];
cx q[8],q[10];
cx q[7],q[10];
u1(1.0) q[10];
cx q[9],q[10];
u1(1.0) q[10];
cx q[7],q[10];
cx q[8],q[10];
cx q[7],q[8];
cx q[4],q[8];
u1(1.0) q[8];
cx q[7],q[8];
u1(1.0) q[8];
cx q[3],q[8];
u1(1.0) q[8];
cx q[4],q[8];
u1(1.0) q[8];
cx q[5],q[8];
u1(1.0) q[8];
cx q[3],q[8];
cx q[5],q[8];
cx q[3],q[5];
cx q[1],q[5];
u1(1.0) q[5];
cx q[3],q[5];
cx q[4],q[5];
u1(1.0) q[5];
cx q[1],q[5];
cx q[4],q[5];
cx q[3],q[4];
cx q[1],q[4];
u1(1.0) q[4];
cx q[1],q[4];
u1(1.0) q[4];
cx q[2],q[4];
u1(1.0) q[4];
cx q[2],q[4];
cx q[0],q[4];
u1(1.0) q[4];
cx q[0],q[4];
cx q[3],q[4];
cx q[3],q[5];
cx q[0],q[5];
u1(1.0) q[5];
cx q[0],q[5];
cx q[3],q[5];
cx q[5],q[6];
cx q[3],q[6];
u1(1.0) q[6];
cx q[3],q[6];
cx q[5],q[6];
cx q[5],q[7];
cx q[3],q[7];
u1(1.0) q[7];
cx q[5],q[7];
cx q[6],q[7];
u1(1.0) q[7];
cx q[3],q[7];
cx q[4],q[7];
u1(1.0) q[7];
cx q[6],q[7];
cx q[6],q[8];
cx q[4],q[8];
u1(1.0) q[8];
cx q[6],q[8];
cx q[5],q[8];
u1(1.0) q[8];
cx q[5],q[8];
cx q[4],q[5];
cx q[0],q[5];
u1(1.0) q[5];
cx q[0],q[5];
cx q[4],q[5];
cx q[5],q[7];
u1(1.0) q[7];
cx q[5],q[6];
cx q[4],q[6];
u1(1.0) q[6];
cx q[4],q[6];
cx q[5],q[6];
cx q[6],q[7];
u1(1.0) q[7];
cx q[4],q[7];
cx q[3],q[7];
u1(1.0) q[7];
cx q[5],q[7];
cx q[4],q[7];
u1(1.0) q[7];
cx q[3],q[7];
cx q[4],q[7];
cx q[6],q[7];
cx q[6],q[8];
cx q[3],q[8];
u1(1.0) q[8];
cx q[4],q[8];
u1(1.0) q[8];
cx q[6],q[8];
cx q[7],q[8];
u1(1.0) q[8];
cx q[3],q[8];
cx q[7],q[8];
cx q[8],q[11];
cx q[7],q[11];
u1(1.0) q[11];
cx q[8],q[9];
cx q[7],q[9];
u1(1.0) q[9];
cx q[6],q[9];
u1(1.0) q[9];
cx q[6],q[9];
cx q[6],q[11];
u1(1.0) q[11];
cx q[6],q[11];
cx q[7],q[9];
cx q[8],q[9];
cx q[9],q[11];
u1(1.0) q[11];
cx q[8],q[11];
cx q[6],q[11];
u1(1.0) q[11];
cx q[9],q[11];
cx q[10],q[11];
u1(1.0) q[11];
cx q[7],q[11];
cx q[10],q[11];
cx q[8],q[10];
cx q[7],q[10];
cx q[6],q[10];
u1(1.0) q[10];
cx q[7],q[10];
cx q[9],q[10];
u1(1.0) q[10];
cx q[6],q[10];
cx q[8],q[10];
cx q[9],q[10];
cx q[9],q[11];
cx q[8],q[11];
u1(1.0) q[11];
cx q[9],q[11];
cx q[10],q[11];
u1(1.0) q[11];
cx q[6],q[11];
cx q[7],q[11];
u1(1.0) q[11];
cx q[7],q[11];
cx q[8],q[11];
cx q[10],q[11];
cx q[11],q[13];
cx q[10],q[13];
u1(1.0) q[13];
cx q[10],q[13];
cx q[11],q[13];
cx q[12],q[13];
cx q[13],q[14];
cx q[12],q[14];
cx q[10],q[14];
u1(1.0) q[14];
cx q[10],q[14];
cx q[12],q[14];
cx q[13],q[14];
cx q[14],q[17];
cx q[13],q[17];
cx q[12],q[17];
u1(1.0) q[17];
cx q[12],q[17];
cx q[13],q[17];
cx q[14],q[17];
cx q[13],q[14];
cx q[12],q[14];
cx q[11],q[14];
u1(1.0) q[14];
cx q[11],q[14];
cx q[12],q[14];
cx q[13],q[14];
cx q[14],q[16];
cx q[13],q[16];
cx q[12],q[16];
u1(1.0) q[16];
cx q[12],q[16];
cx q[15],q[16];
u1(1.0) q[16];
cx q[13],q[16];
cx q[14],q[16];
cx q[15],q[16];
cx q[16],q[17];
cx q[15],q[17];
cx q[14],q[17];
u1(1.0) q[17];
cx q[14],q[17];
cx q[16],q[17];
cx q[15],q[16];
cx q[14],q[16];
cx q[12],q[16];
u1(1.0) q[16];
cx q[14],q[16];
cx q[13],q[16];
u1(1.0) q[16];
cx q[12],q[16];
cx q[13],q[16];
cx q[15],q[16];
cx q[16],q[17];
cx q[13],q[17];
u1(1.0) q[17];
cx q[13],q[17];
cx q[15],q[17];
cx q[16],q[17];
cx q[17],q[18];
cx q[16],q[18];
cx q[15],q[18];
u1(1.0) q[18];
cx q[15],q[18];
u1(1.0) q[18];
cx q[16],q[18];
cx q[17],q[18];
cx q[17],q[20];
cx q[16],q[20];
u1(1.0) q[20];
cx q[18],q[20];
u1(1.0) q[20];
cx q[18],q[20];
cx q[19],q[20];
u1(1.0) q[20];
cx q[17],q[20];
cx q[15],q[20];
u1(1.0) q[20];
cx q[16],q[20];
cx q[19],q[20];
cx q[17],q[19];
cx q[16],q[19];
cx q[15],q[19];
u1(1.0) q[19];
cx q[16],q[19];
cx q[18],q[19];
u1(1.0) q[19];
cx q[15],q[19];
cx q[17],q[19];
cx q[18],q[19];
cx q[19],q[20];
cx q[18],q[20];
u1(1.0) q[20];
cx q[15],q[20];
cx q[18],q[20];
cx q[19],q[20];
cx q[20],q[23];
cx q[19],q[23];
cx q[18],q[23];
u1(1.0) q[23];
cx q[19],q[23];
cx q[20],q[22];
cx q[19],q[22];
cx q[18],q[22];
u1(1.0) q[22];
cx q[18],q[22];
cx q[19],q[22];
cx q[20],q[22];
cx q[22],q[23];
u1(1.0) q[23];
cx q[20],q[23];
u1(1.0) q[23];
cx q[22],q[23];
cx q[21],q[23];
u1(1.0) q[23];
cx q[18],q[23];
cx q[0],q[23];
u1(1.0) q[23];
cx q[21],q[23];
cx q[0],q[21];
u1(1.0) q[21];
cx q[2],q[21];
u1(1.0) q[21];
cx q[0],q[21];
cx q[2],q[21];
cx q[21],q[22];
cx q[0],q[22];
u1(1.0) q[22];
cx q[0],q[22];
cx q[21],q[22];
cx q[22],q[23];
u1(1.0) q[23];
cx q[21],q[23];
u1(1.0) q[23];
cx q[0],q[23];
cx q[20],q[23];
u1(1.0) q[23];
cx q[20],q[23];
cx q[2],q[23];
u1(1.0) q[23];
cx q[21],q[23];
cx q[22],q[23];
cx q[21],q[22];
cx q[2],q[22];
cx q[1],q[22];
u1(1.0) q[22];
cx q[2],q[4];
cx q[1],q[4];
u1(1.0) q[4];
cx q[2],q[3];
cx q[1],q[3];
u1(1.0) q[3];
cx q[1],q[3];
cx q[1],q[23];
u1(1.0) q[23];
cx q[2],q[3];
cx q[21],q[22];
u1(1.0) q[22];
cx q[2],q[21];
cx q[1],q[21];
u1(1.0) q[21];
cx q[1],q[21];
cx q[2],q[21];
cx q[21],q[23];
u1(1.0) q[23];
cx q[2],q[23];
cx q[21],q[23];
cx q[2],q[21];
cx q[1],q[21];
cx q[0],q[21];
u1(1.0) q[21];
cx q[0],q[21];
cx q[0],q[4];
u1(1.0) q[4];
cx q[0],q[4];
cx q[1],q[21];
cx q[1],q[4];
cx q[2],q[21];
cx q[2],q[4];
cx q[2],q[5];
cx q[1],q[5];
cx q[0],q[5];
u1(1.0) q[5];
cx q[2],q[5];
cx q[3],q[5];
u1(1.0) q[5];
cx q[3],q[5];
cx q[4],q[5];
u1(1.0) q[5];
cx q[1],q[5];
cx q[2],q[5];
u1(1.0) q[5];
cx q[0],q[5];
cx q[3],q[5];
u1(1.0) q[5];
cx q[2],q[5];
cx q[3],q[5];
cx q[4],q[5];
cx q[5],q[8];
cx q[4],q[8];
cx q[3],q[8];
u1(1.0) q[8];
cx q[5],q[8];
cx q[5],q[6];
cx q[4],q[6];
cx q[3],q[6];
u1(1.0) q[6];
cx q[3],q[6];
cx q[4],q[6];
cx q[5],q[6];
cx q[5],q[7];
cx q[4],q[7];
cx q[3],q[7];
u1(1.0) q[7];
cx q[3],q[7];
cx q[4],q[7];
cx q[5],q[7];
cx q[7],q[8];
u1(1.0) q[8];
cx q[4],q[8];
cx q[5],q[8];
u1(1.0) q[8];
cx q[7],q[8];
cx q[6],q[8];
u1(1.0) q[8];
cx q[3],q[8];
cx q[7],q[8];
u1(1.0) q[8];
cx q[7],q[8];
cx q[4],q[8];
u1(1.0) q[8];
cx q[6],q[8];
cx q[7],q[8];
u1(1.0) q[8];
cx q[5],q[8];
cx q[6],q[8];
u1(1.0) q[8];
cx q[4],q[8];
cx q[3],q[8];
u1(1.0) q[8];
cx q[4],q[8];
u1(1.0) q[8];
cx q[3],q[8];
cx q[4],q[8];
cx q[6],q[8];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[7];
cx q[4],q[7];
cx q[3],q[7];
u1(1.0) q[7];
cx q[3],q[7];
cx q[4],q[7];
cx q[5],q[7];
cx q[4],q[5];
cx q[3],q[5];
cx q[1],q[5];
u1(1.0) q[5];
cx q[1],q[5];
cx q[4],q[5];
cx q[3],q[4];
cx q[1],q[4];
cx q[0],q[4];
u1(1.0) q[4];
cx q[0],q[4];
cx q[2],q[4];
u1(1.0) q[4];
cx q[1],q[4];
cx q[0],q[4];
u1(1.0) q[4];
cx q[0],q[4];
cx q[2],q[4];
cx q[2],q[5];
cx q[0],q[5];
u1(1.0) q[5];
cx q[0],q[5];
cx q[0],q[22];
u1(1.0) q[22];
cx q[0],q[22];
cx q[1],q[5];
u1(1.0) q[5];
cx q[1],q[22];
cx q[2],q[22];
cx q[22],q[23];
cx q[0],q[23];
u1(1.0) q[23];
cx q[22],q[23];
cx q[2],q[23];
u1(1.0) q[23];
cx q[1],q[23];
cx q[21],q[23];
u1(1.0) q[23];
cx q[21],q[23];
cx q[22],q[23];
u1(1.0) q[23];
cx q[0],q[23];
cx q[2],q[23];
cx q[22],q[23];
cx q[21],q[22];
cx q[2],q[22];
cx q[0],q[22];
u1(1.0) q[22];
cx q[2],q[22];
cx q[1],q[22];
u1(1.0) q[22];
cx q[0],q[22];
cx q[1],q[22];
cx q[21],q[22];
cx q[22],q[23];
cx q[21],q[23];
cx q[1],q[23];
u1(1.0) q[23];
cx q[1],q[23];
cx q[19],q[23];
u1(1.0) q[23];
cx q[19],q[23];
cx q[22],q[23];
cx q[21],q[22];
cx q[20],q[22];
cx q[19],q[22];
u1(1.0) q[22];
cx q[20],q[22];
cx q[18],q[22];
u1(1.0) q[22];
cx q[18],q[22];
cx q[19],q[22];
cx q[21],q[22];
cx q[22],q[23];
cx q[18],q[23];
u1(1.0) q[23];
cx q[22],q[23];
cx q[20],q[23];
u1(1.0) q[23];
cx q[20],q[23];
cx q[19],q[23];
u1(1.0) q[23];
cx q[21],q[23];
cx q[22],q[23];
u1(1.0) q[23];
cx q[18],q[23];
cx q[20],q[23];
u1(1.0) q[23];
cx q[22],q[23];
cx q[21],q[23];
u1(1.0) q[23];
cx q[19],q[23];
cx q[20],q[23];
cx q[19],q[20];
cx q[18],q[20];
cx q[16],q[20];
u1(1.0) q[20];
cx q[19],q[20];
cx q[15],q[20];
u1(1.0) q[20];
cx q[18],q[20];
cx q[17],q[20];
u1(1.0) q[20];
cx q[16],q[20];
cx q[18],q[20];
u1(1.0) q[20];
cx q[18],q[20];
cx q[19],q[20];
u1(1.0) q[20];
cx q[17],q[20];
cx q[15],q[17];
cx q[14],q[17];
cx q[12],q[17];
u1(1.0) q[17];
cx q[12],q[17];
cx q[13],q[17];
u1(1.0) q[17];
cx q[15],q[17];
cx q[16],q[17];
u1(1.0) q[17];
cx q[14],q[17];
cx q[12],q[17];
u1(1.0) q[17];
cx q[13],q[17];
cx q[14],q[17];
u1(1.0) q[17];
cx q[14],q[17];
cx q[15],q[17];
u1(1.0) q[17];
cx q[13],q[17];
u1(1.0) q[17];
cx q[13],q[17];
cx q[16],q[17];
cx q[15],q[16];
cx q[14],q[16];
cx q[13],q[16];
cx q[12],q[16];
u1(1.0) q[16];
cx q[12],q[16];
cx q[13],q[16];
cx q[14],q[16];
cx q[13],q[14];
cx q[12],q[14];
cx q[15],q[16];
cx q[19],q[20];
cx q[18],q[19];
cx q[21],q[22];
cx q[3],q[4];
cx q[3],q[5];
cx q[4],q[5];
u1(1.0) q[5];
cx q[4],q[5];
cx q[3],q[5];
cx q[6],q[7];
cx q[9],q[14];
u1(1.0) q[14];
cx q[13],q[14];
cx q[12],q[13];
cx q[10],q[13];
cx q[9],q[13];
u1(1.0) q[13];
cx q[9],q[13];
cx q[10],q[13];
cx q[10],q[14];
u1(1.0) q[14];
cx q[10],q[14];
cx q[11],q[14];
u1(1.0) q[14];
cx q[12],q[13];
cx q[9],q[14];
cx q[10],q[14];
u1(1.0) q[14];
cx q[12],q[14];
cx q[13],q[14];
u1(1.0) q[14];
cx q[11],q[14];
cx q[9],q[14];
u1(1.0) q[14];
cx q[13],q[14];
cx q[12],q[13];
cx q[11],q[13];
cx q[9],q[13];
u1(1.0) q[13];
cx q[10],q[13];
u1(1.0) q[13];
cx q[9],q[13];
cx q[10],q[13];
cx q[11],q[13];
cx q[12],q[13];
cx q[12],q[14];
cx q[11],q[14];
u1(1.0) q[14];
cx q[10],q[14];
cx q[11],q[14];
cx q[12],q[14];
cx q[13],q[14];
cx q[9],q[10];
cx q[7],q[10];
cx q[6],q[10];
u1(1.0) q[10];
cx q[8],q[10];
u1(1.0) q[10];
cx q[6],q[10];
cx q[7],q[10];
cx q[8],q[10];
cx q[9],q[10];
cx q[9],q[11];
cx q[8],q[11];
cx q[7],q[11];
cx q[6],q[11];
u1(1.0) q[11];
cx q[9],q[11];
cx q[10],q[11];
u1(1.0) q[11];
cx q[8],q[11];
cx q[9],q[11];
u1(1.0) q[11];
cx q[7],q[11];
cx q[8],q[11];
u1(1.0) q[11];
cx q[6],q[11];
cx q[7],q[11];
u1(1.0) q[11];
cx q[7],q[11];
cx q[8],q[11];
cx q[6],q[8];
cx q[9],q[11];
cx q[10],q[11];
cx q[11],q[14];
cx q[10],q[14];
u1(1.0) q[14];
cx q[11],q[14];
cx q[12],q[14];
u1(1.0) q[14];
cx q[10],q[14];
cx q[11],q[14];
u1(1.0) q[14];
cx q[9],q[14];
cx q[10],q[14];
u1(1.0) q[14];
cx q[10],q[14];
cx q[11],q[14];
cx q[12],q[14];
cx q[13],q[14];
cx q[14],q[17];
cx q[13],q[17];
u1(1.0) q[17];
cx q[15],q[17];
cx q[16],q[17];
u1(1.0) q[17];
cx q[12],q[17];
cx q[13],q[17];
cx q[14],q[17];
cx q[16],q[17];
cx q[16],q[19];
cx q[15],q[19];
u1(1.0) q[19];
cx q[17],q[19];
u1(1.0) q[19];
cx q[15],q[19];
cx q[16],q[19];
cx q[17],q[19];
cx q[18],q[19];
cx q[18],q[20];
cx q[17],q[20];
cx q[16],q[20];
u1(1.0) q[20];
cx q[15],q[20];
cx q[16],q[20];
cx q[17],q[20];
cx q[18],q[20];
cx q[20],q[22];
cx q[18],q[22];
u1(1.0) q[22];
cx q[19],q[22];
u1(1.0) q[22];
cx q[18],q[22];
cx q[19],q[22];
cx q[20],q[22];
cx q[20],q[23];
cx q[19],q[23];
cx q[18],q[23];
u1(1.0) q[23];
cx q[18],q[23];
cx q[19],q[23];
cx q[20],q[23];
cx q[1],q[23];
cx q[0],q[23];
u1(1.0) q[23];
cx q[0],q[23];
cx q[21],q[22];
cx q[21],q[23];
cx q[22],q[23];
cx q[2],q[23];
u1(1.0) q[23];
cx q[0],q[23];
u1(1.0) q[23];
cx q[22],q[23];
cx q[21],q[22];
cx q[2],q[22];
cx q[1],q[22];
cx q[0],q[22];
u1(1.0) q[22];
cx q[0],q[22];
cx q[1],q[22];
cx q[2],q[22];
cx q[21],q[22];
cx q[21],q[23];
u1(1.0) q[23];
cx q[2],q[23];
cx q[22],q[23];
u1(1.0) q[23];
cx q[1],q[23];
cx q[2],q[23];
u1(1.0) q[23];
cx q[0],q[23];
cx q[0],q[5];
u1(1.0) q[5];
cx q[1],q[23];
u1(1.0) q[23];
cx q[1],q[23];
cx q[1],q[5];
cx q[2],q[23];
cx q[19],q[23];
cx q[18],q[23];
u1(1.0) q[23];
cx q[2],q[5];
cx q[21],q[23];
cx q[20],q[23];
u1(1.0) q[23];
cx q[19],q[23];
cx q[21],q[23];
u1(1.0) q[23];
cx q[18],q[23];
cx q[19],q[23];
u1(1.0) q[23];
cx q[19],q[23];
cx q[20],q[23];
cx q[19],q[20];
cx q[17],q[20];
cx q[16],q[20];
cx q[15],q[20];
u1(1.0) q[20];
cx q[17],q[20];
cx q[18],q[20];
u1(1.0) q[20];
cx q[16],q[20];
cx q[17],q[20];
u1(1.0) q[20];
cx q[15],q[20];
cx q[16],q[20];
u1(1.0) q[20];
cx q[16],q[20];
cx q[17],q[20];
cx q[16],q[17];
cx q[15],q[17];
cx q[14],q[17];
cx q[12],q[17];
u1(1.0) q[17];
cx q[12],q[17];
cx q[13],q[17];
u1(1.0) q[17];
cx q[13],q[17];
cx q[14],q[17];
cx q[15],q[17];
cx q[16],q[17];
cx q[18],q[20];
cx q[19],q[20];
cx q[4],q[5];
u1(1.0) q[5];
cx q[1],q[5];
u1(1.0) q[5];
cx q[3],q[5];
cx q[2],q[5];
u1(1.0) q[5];
cx q[1],q[5];
cx q[3],q[5];
u1(1.0) q[5];
cx q[0],q[5];
cx q[1],q[5];
u1(1.0) q[5];
cx q[1],q[5];
cx q[2],q[5];
cx q[3],q[5];
cx q[4],q[5];
cx q[5],q[8];
cx q[4],q[8];
cx q[3],q[8];
u1(1.0) q[8];
cx q[6],q[8];
cx q[7],q[8];
u1(1.0) q[8];
cx q[4],q[8];
cx q[6],q[8];
u1(1.0) q[8];
cx q[3],q[8];
cx q[4],q[8];
u1(1.0) q[8];
cx q[3],q[8];
u1(1.0) q[8];
cx q[3],q[8];
cx q[4],q[8];
cx q[5],q[8];
cx q[4],q[5];
cx q[3],q[5];
cx q[2],q[5];
cx q[1],q[5];
cx q[0],q[5];
u1(1.0) q[5];
cx q[0],q[5];
cx q[1],q[5];
cx q[2],q[5];
cx q[2],q[23];
cx q[1],q[23];
cx q[0],q[23];
u1(1.0) q[23];
cx q[0],q[23];
cx q[1],q[23];
cx q[2],q[23];
cx q[20],q[23];
cx q[19],q[23];
cx q[18],q[23];
u1(1.0) q[23];
cx q[18],q[23];
cx q[19],q[23];
cx q[20],q[23];
cx q[19],q[20];
cx q[18],q[20];
cx q[17],q[20];
cx q[16],q[20];
cx q[15],q[20];
u1(1.0) q[20];
cx q[15],q[20];
cx q[16],q[20];
cx q[17],q[20];
cx q[16],q[17];
cx q[15],q[17];
cx q[14],q[17];
cx q[13],q[17];
cx q[12],q[17];
u1(1.0) q[17];
cx q[12],q[17];
cx q[13],q[17];
cx q[14],q[17];
cx q[13],q[14];
cx q[12],q[14];
cx q[11],q[14];
cx q[10],q[14];
cx q[15],q[17];
cx q[16],q[17];
cx q[18],q[20];
cx q[19],q[20];
cx q[21],q[23];
cx q[22],q[23];
cx q[3],q[5];
cx q[4],q[5];
cx q[6],q[8];
cx q[7],q[8];
cx q[9],q[14];
u1(1.0) q[14];
cx q[9],q[14];
cx q[10],q[14];
cx q[11],q[14];
cx q[10],q[11];
cx q[12],q[14];
cx q[13],q[14];
cx q[9],q[11];
cx q[8],q[11];
cx q[7],q[11];
cx q[6],q[11];
u1(1.0) q[11];
cx q[6],q[11];
cx q[7],q[11];
cx q[8],q[11];
cx q[9],q[11];
cx q[10],q[11];
