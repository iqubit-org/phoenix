OPENQASM 2.0;
include "qelib1.inc";
qreg q[49];
cx q[0],q[8];
u1(1.0) q[8];
cx q[0],q[8];
u1(1.0) q[0];
cx q[0],q[44];
u1(1.0) q[44];
cx q[0],q[44];
u1(1.0) q[44];
cx q[39],q[44];
u1(1.0) q[44];
cx q[39],q[44];
u1(1.0) q[39];
cx q[33],q[39];
u1(1.0) q[39];
cx q[33],q[39];
u1(1.0) q[33];
cx q[33],q[35];
u1(1.0) q[35];
cx q[33],q[35];
u1(1.0) q[35];
cx q[29],q[35];
u1(1.0) q[35];
cx q[29],q[35];
u1(1.0) q[29];
cx q[24],q[29];
u1(1.0) q[29];
cx q[24],q[29];
u1(1.0) q[24];
cx q[18],q[24];
u1(1.0) q[24];
cx q[18],q[24];
u1(1.0) q[18];
cx q[13],q[18];
u1(1.0) q[18];
cx q[13],q[18];
u1(1.0) q[13];
cx q[13],q[14];
u1(1.0) q[14];
cx q[13],q[14];
u1(1.0) q[14];
cx q[9],q[14];
u1(1.0) q[14];
cx q[9],q[14];
u1(1.0) q[9];
cx q[3],q[9];
u1(1.0) q[9];
cx q[3],q[9];
u1(1.0) q[3];
cx q[3],q[47];
u1(1.0) q[47];
cx q[3],q[47];
u1(1.0) q[47];
cx q[0],q[47];
u1(1.0) q[47];
cx q[0],q[47];
cx q[0],q[43];
u1(1.0) q[43];
cx q[0],q[43];
u1(1.0) q[43];
cx q[41],q[43];
u1(1.0) q[43];
cx q[41],q[43];
u1(1.0) q[41];
cx q[41],q[46];
u1(1.0) q[46];
cx q[41],q[46];
u1(1.0) q[46];
cx q[2],q[46];
u1(1.0) q[46];
cx q[2],q[46];
u1(1.0) q[2];
cx q[2],q[10];
u1(1.0) q[10];
cx q[2],q[10];
u1(1.0) q[10];
cx q[10],q[15];
u1(1.0) q[15];
cx q[10],q[15];
u1(1.0) q[15];
cx q[15],q[21];
u1(1.0) q[21];
cx q[15],q[21];
u1(1.0) q[21];
cx q[19],q[21];
u1(1.0) q[21];
cx q[19],q[21];
u1(1.0) q[19];
cx q[19],q[25];
u1(1.0) q[25];
cx q[19],q[25];
u1(1.0) q[25];
cx q[25],q[30];
u1(1.0) q[30];
cx q[25],q[30];
u1(1.0) q[30];
cx q[30],q[36];
u1(1.0) q[36];
cx q[30],q[36];
u1(1.0) q[36];
cx q[34],q[36];
u1(1.0) q[36];
cx q[34],q[36];
u1(1.0) q[34];
cx q[34],q[40];
u1(1.0) q[40];
cx q[34],q[40];
u1(1.0) q[40];
cx q[40],q[45];
u1(1.0) q[45];
cx q[40],q[45];
u1(1.0) q[45];
cx q[1],q[45];
u1(1.0) q[45];
cx q[1],q[45];
u1(1.0) q[1];
cx q[1],q[7];
u1(1.0) q[7];
cx q[1],q[7];
u1(1.0) q[7];
cx q[5],q[7];
u1(1.0) q[7];
cx q[5],q[7];
u1(1.0) q[5];
cx q[5],q[11];
u1(1.0) q[11];
cx q[5],q[11];
u1(1.0) q[11];
cx q[11],q[16];
u1(1.0) q[16];
cx q[11],q[16];
u1(1.0) q[16];
cx q[16],q[22];
u1(1.0) q[22];
cx q[16],q[22];
u1(1.0) q[22];
cx q[20],q[22];
u1(1.0) q[22];
cx q[20],q[22];
u1(1.0) q[20];
cx q[20],q[26];
u1(1.0) q[26];
cx q[20],q[26];
u1(1.0) q[26];
cx q[26],q[31];
u1(1.0) q[31];
cx q[26],q[31];
u1(1.0) q[31];
cx q[31],q[37];
u1(1.0) q[37];
cx q[31],q[37];
u1(1.0) q[37];
cx q[37],q[42];
u1(1.0) q[42];
cx q[37],q[42];
u1(1.0) q[42];
cx q[6],q[42];
u1(1.0) q[42];
cx q[6],q[42];
u1(1.0) q[6];
cx q[6],q[8];
u1(1.0) q[8];
cx q[6],q[8];
u1(1.0) q[8];
cx q[3],q[8];
u1(1.0) q[8];
cx q[3],q[8];
cx q[3],q[48];
u1(1.0) q[48];
cx q[3],q[48];
u1(1.0) q[48];
cx q[4],q[48];
u1(1.0) q[48];
cx q[4],q[48];
u1(1.0) q[4];
cx q[4],q[12];
u1(1.0) q[12];
cx q[4],q[12];
u1(1.0) q[12];
cx q[12],q[17];
u1(1.0) q[17];
cx q[12],q[17];
u1(1.0) q[17];
cx q[17],q[23];
u1(1.0) q[23];
cx q[17],q[23];
u1(1.0) q[23];
cx q[23],q[28];
u1(1.0) q[28];
cx q[23],q[28];
u1(1.0) q[28];
cx q[27],q[28];
u1(1.0) q[28];
cx q[27],q[28];
u1(1.0) q[27];
cx q[27],q[32];
u1(1.0) q[32];
cx q[27],q[32];
u1(1.0) q[32];
cx q[32],q[38];
u1(1.0) q[38];
cx q[32],q[38];
u1(1.0) q[38];
cx q[38],q[43];
u1(1.0) q[43];
cx q[38],q[43];
cx q[33],q[38];
u1(1.0) q[38];
cx q[33],q[38];
cx q[33],q[36];
u1(1.0) q[36];
cx q[33],q[36];
cx q[31],q[36];
u1(1.0) q[36];
cx q[31],q[36];
cx q[27],q[31];
u1(1.0) q[31];
cx q[27],q[31];
cx q[27],q[29];
u1(1.0) q[29];
cx q[27],q[29];
cx q[25],q[29];
u1(1.0) q[29];
cx q[25],q[29];
cx q[20],q[25];
u1(1.0) q[25];
cx q[20],q[25];
cx q[20],q[23];
u1(1.0) q[23];
cx q[20],q[23];
cx q[18],q[23];
u1(1.0) q[23];
cx q[18],q[23];
cx q[18],q[21];
u1(1.0) q[21];
cx q[18],q[21];
cx q[16],q[21];
u1(1.0) q[21];
cx q[16],q[21];
cx q[12],q[16];
u1(1.0) q[16];
cx q[12],q[16];
cx q[12],q[14];
u1(1.0) q[14];
cx q[12],q[14];
cx q[10],q[14];
u1(1.0) q[14];
cx q[10],q[14];
cx q[5],q[10];
u1(1.0) q[10];
cx q[5],q[10];
cx q[5],q[8];
u1(1.0) q[8];
cx q[5],q[8];
cx q[4],q[8];
u1(1.0) q[8];
cx q[4],q[8];
cx q[4],q[9];
u1(1.0) q[9];
cx q[4],q[9];
cx q[4],q[47];
u1(1.0) q[47];
cx q[4],q[47];
cx q[41],q[47];
u1(1.0) q[47];
cx q[41],q[47];
cx q[41],q[44];
u1(1.0) q[44];
cx q[41],q[44];
cx q[40],q[44];
u1(1.0) q[44];
cx q[40],q[44];
cx q[40],q[42];
u1(1.0) q[42];
cx q[40],q[42];
cx q[38],q[42];
u1(1.0) q[42];
cx q[38],q[42];
cx q[34],q[38];
u1(1.0) q[38];
cx q[34],q[38];
cx q[34],q[39];
u1(1.0) q[39];
cx q[34],q[39];
cx q[34],q[35];
u1(1.0) q[35];
cx q[34],q[35];
cx q[30],q[35];
u1(1.0) q[35];
cx q[30],q[35];
cx q[24],q[30];
u1(1.0) q[30];
cx q[24],q[30];
cx q[19],q[24];
u1(1.0) q[24];
cx q[19],q[24];
cx q[13],q[19];
u1(1.0) q[19];
cx q[13],q[19];
cx q[13],q[15];
u1(1.0) q[15];
cx q[13],q[15];
cx q[9],q[15];
u1(1.0) q[15];
cx q[9],q[15];
cx q[1],q[9];
u1(1.0) q[9];
cx q[1],q[9];
cx q[1],q[48];
u1(1.0) q[48];
cx q[1],q[48];
cx q[1],q[46];
u1(1.0) q[46];
cx q[1],q[46];
cx q[0],q[46];
u1(1.0) q[46];
cx q[0],q[46];
cx q[0],q[48];
u1(1.0) q[48];
cx q[0],q[48];
cx q[2],q[48];
u1(1.0) q[48];
cx q[2],q[48];
cx q[2],q[7];
u1(1.0) q[7];
cx q[2],q[7];
cx q[2],q[45];
u1(1.0) q[45];
cx q[2],q[45];
cx q[39],q[45];
u1(1.0) q[45];
cx q[39],q[45];
cx q[39],q[42];
u1(1.0) q[42];
cx q[39],q[42];
cx q[31],q[39];
u1(1.0) q[39];
cx q[31],q[39];
cx q[31],q[35];
u1(1.0) q[35];
cx q[31],q[35];
cx q[32],q[35];
u1(1.0) q[35];
cx q[32],q[35];
cx q[32],q[37];
u1(1.0) q[37];
cx q[32],q[37];
cx q[26],q[32];
u1(1.0) q[32];
cx q[26],q[32];
cx q[26],q[28];
u1(1.0) q[28];
cx q[26],q[28];
cx q[22],q[28];
u1(1.0) q[28];
cx q[22],q[28];
cx q[17],q[22];
u1(1.0) q[22];
cx q[17],q[22];
cx q[11],q[17];
u1(1.0) q[17];
cx q[11],q[17];
cx q[6],q[11];
u1(1.0) q[11];
cx q[6],q[11];
cx q[6],q[7];
u1(1.0) q[7];
cx q[6],q[7];
cx q[3],q[7];
u1(1.0) q[7];
cx q[3],q[7];
cx q[3],q[46];
u1(1.0) q[46];
cx q[3],q[46];
cx q[40],q[46];
u1(1.0) q[46];
cx q[40],q[46];
cx q[40],q[43];
u1(1.0) q[43];
cx q[40],q[43];
cx q[37],q[43];
u1(1.0) q[43];
cx q[37],q[43];
cx q[33],q[37];
u1(1.0) q[37];
cx q[33],q[37];
cx q[27],q[33];
u1(1.0) q[33];
cx q[27],q[33];
cx q[27],q[30];
u1(1.0) q[30];
cx q[27],q[30];
cx q[30],q[38];
u1(1.0) q[38];
cx q[30],q[38];
cx q[26],q[30];
u1(1.0) q[30];
cx q[26],q[30];
cx q[26],q[29];
u1(1.0) q[29];
cx q[26],q[29];
cx q[18],q[26];
u1(1.0) q[26];
cx q[18],q[26];
cx q[18],q[22];
u1(1.0) q[22];
cx q[18],q[22];
cx q[19],q[22];
u1(1.0) q[22];
cx q[19],q[22];
cx q[19],q[23];
u1(1.0) q[23];
cx q[19],q[23];
cx q[23],q[29];
u1(1.0) q[29];
cx q[23],q[29];
cx q[15],q[23];
u1(1.0) q[23];
cx q[15],q[23];
cx q[12],q[15];
u1(1.0) q[15];
cx q[12],q[15];
cx q[6],q[12];
u1(1.0) q[12];
cx q[6],q[12];
cx q[6],q[10];
u1(1.0) q[10];
cx q[6],q[10];
cx q[4],q[10];
u1(1.0) q[10];
cx q[4],q[10];
cx q[4],q[7];
u1(1.0) q[7];
cx q[4],q[7];
cx q[7],q[20];
u1(1.0) q[20];
cx q[7],q[20];
cx q[20],q[21];
u1(1.0) q[21];
cx q[20],q[21];
cx q[17],q[21];
u1(1.0) q[21];
cx q[17],q[21];
cx q[17],q[25];
u1(1.0) q[25];
cx q[17],q[25];
cx q[25],q[28];
u1(1.0) q[28];
cx q[25],q[28];
cx q[24],q[28];
u1(1.0) q[28];
cx q[24],q[28];
cx q[16],q[24];
u1(1.0) q[24];
cx q[16],q[24];
cx q[13],q[16];
u1(1.0) q[16];
cx q[13],q[16];
cx q[5],q[13];
u1(1.0) q[13];
cx q[5],q[13];
cx q[5],q[9];
u1(1.0) q[9];
cx q[5],q[9];
cx q[5],q[48];
u1(1.0) q[48];
cx q[5],q[48];
cx q[40],q[48];
u1(1.0) q[48];
cx q[40],q[48];
cx q[32],q[40];
u1(1.0) q[40];
cx q[32],q[40];
cx q[32],q[36];
u1(1.0) q[36];
cx q[32],q[36];
cx q[36],q[42];
u1(1.0) q[42];
cx q[36],q[42];
cx q[36],q[44];
u1(1.0) q[44];
cx q[36],q[44];
cx q[1],q[44];
u1(1.0) q[44];
cx q[1],q[44];
cx q[1],q[47];
u1(1.0) q[47];
cx q[1],q[47];
cx q[2],q[47];
u1(1.0) q[47];
cx q[2],q[47];
cx q[2],q[43];
u1(1.0) q[43];
cx q[2],q[43];
cx q[35],q[43];
u1(1.0) q[43];
cx q[35],q[43];
cx q[6],q[43];
u1(1.0) q[43];
cx q[6],q[43];
cx q[6],q[9];
u1(1.0) q[9];
cx q[6],q[9];
cx q[0],q[9];
u1(1.0) q[9];
cx q[0],q[9];
cx q[0],q[45];
u1(1.0) q[45];
cx q[0],q[45];
cx q[41],q[45];
u1(1.0) q[45];
cx q[41],q[45];
cx q[28],q[41];
u1(1.0) q[41];
cx q[28],q[41];
cx q[28],q[36];
u1(1.0) q[36];
cx q[28],q[36];
cx q[33],q[41];
u1(1.0) q[41];
cx q[33],q[41];
cx q[25],q[33];
u1(1.0) q[33];
cx q[25],q[33];
cx q[25],q[31];
u1(1.0) q[31];
cx q[25],q[31];
cx q[23],q[31];
u1(1.0) q[31];
cx q[23],q[31];
cx q[14],q[23];
u1(1.0) q[23];
cx q[14],q[23];
cx q[11],q[14];
u1(1.0) q[14];
cx q[11],q[14];
cx q[31],q[40];
u1(1.0) q[40];
cx q[31],q[40];
cx q[8],q[14];
u1(1.0) q[14];
cx q[8],q[14];
cx q[8],q[16];
u1(1.0) q[16];
cx q[8],q[16];
cx q[2],q[8];
u1(1.0) q[8];
cx q[2],q[8];
cx q[2],q[12];
u1(1.0) q[12];
cx q[2],q[12];
cx q[12],q[20];
u1(1.0) q[20];
cx q[12],q[20];
cx q[12],q[18];
u1(1.0) q[18];
cx q[12],q[18];
cx q[10],q[18];
u1(1.0) q[18];
cx q[10],q[18];
cx q[10],q[16];
u1(1.0) q[16];
cx q[10],q[16];
cx q[0],q[10];
u1(1.0) q[10];
cx q[0],q[10];
cx q[0],q[11];
u1(1.0) q[11];
cx q[0],q[11];
cx q[11],q[15];
u1(1.0) q[15];
cx q[11],q[15];
cx q[11],q[19];
u1(1.0) q[19];
cx q[11],q[19];
cx q[3],q[11];
u1(1.0) q[11];
cx q[3],q[11];
cx q[3],q[44];
u1(1.0) q[44];
cx q[3],q[44];
cx q[35],q[44];
u1(1.0) q[44];
cx q[35],q[44];
cx q[6],q[44];
u1(1.0) q[44];
cx q[6],q[44];
cx q[6],q[47];
u1(1.0) q[47];
cx q[6],q[47];
cx q[39],q[47];
u1(1.0) q[47];
cx q[39],q[47];
cx q[38],q[47];
u1(1.0) q[47];
cx q[38],q[47];
cx q[38],q[46];
u1(1.0) q[46];
cx q[38],q[46];
cx q[5],q[46];
u1(1.0) q[46];
cx q[5],q[46];
cx q[5],q[42];
u1(1.0) q[42];
cx q[5],q[42];
cx q[1],q[42];
u1(1.0) q[42];
cx q[1],q[42];
cx q[1],q[11];
u1(1.0) q[11];
cx q[1],q[11];
cx q[1],q[10];
u1(1.0) q[10];
cx q[1],q[10];
cx q[1],q[12];
u1(1.0) q[12];
cx q[1],q[12];
cx q[3],q[12];
u1(1.0) q[12];
cx q[3],q[12];
cx q[3],q[13];
u1(1.0) q[13];
cx q[3],q[13];
cx q[13],q[17];
u1(1.0) q[17];
cx q[13],q[17];
cx q[4],q[13];
u1(1.0) q[13];
cx q[4],q[13];
cx q[4],q[45];
u1(1.0) q[45];
cx q[4],q[45];
cx q[37],q[45];
u1(1.0) q[45];
cx q[37],q[45];
cx q[34],q[37];
u1(1.0) q[37];
cx q[34],q[37];
cx q[21],q[34];
u1(1.0) q[34];
cx q[21],q[34];
cx q[21],q[29];
u1(1.0) q[29];
cx q[21],q[29];
cx q[29],q[37];
u1(1.0) q[37];
cx q[29],q[37];
cx q[28],q[37];
u1(1.0) q[37];
cx q[28],q[37];
cx q[28],q[38];
u1(1.0) q[38];
cx q[28],q[38];
cx q[29],q[41];
u1(1.0) q[41];
cx q[29],q[41];
cx q[32],q[41];
u1(1.0) q[41];
cx q[32],q[41];
cx q[24],q[32];
u1(1.0) q[32];
cx q[24],q[32];
cx q[20],q[24];
u1(1.0) q[24];
cx q[20],q[24];
cx q[15],q[24];
u1(1.0) q[24];
cx q[15],q[24];
cx q[23],q[32];
u1(1.0) q[32];
cx q[23],q[32];
cx q[35],q[45];
u1(1.0) q[45];
cx q[35],q[45];
cx q[35],q[46];
u1(1.0) q[46];
cx q[35],q[46];
cx q[36],q[46];
u1(1.0) q[46];
cx q[36],q[46];
cx q[36],q[45];
u1(1.0) q[45];
cx q[36],q[45];
cx q[37],q[47];
u1(1.0) q[47];
cx q[37],q[47];
cx q[36],q[47];
u1(1.0) q[47];
cx q[36],q[47];
cx q[37],q[46];
u1(1.0) q[46];
cx q[37],q[46];
cx q[38],q[48];
u1(1.0) q[48];
cx q[38],q[48];
cx q[39],q[48];
u1(1.0) q[48];
cx q[39],q[48];
cx q[36],q[48];
u1(1.0) q[48];
cx q[36],q[48];
cx q[35],q[48];
u1(1.0) q[48];
cx q[35],q[48];
cx q[35],q[47];
u1(1.0) q[47];
cx q[35],q[47];
cx q[37],q[48];
u1(1.0) q[48];
cx q[37],q[48];
cx q[39],q[43];
u1(1.0) q[43];
cx q[39],q[43];
cx q[5],q[43];
u1(1.0) q[43];
cx q[5],q[43];
cx q[5],q[44];
u1(1.0) q[44];
cx q[5],q[44];
cx q[4],q[44];
u1(1.0) q[44];
cx q[4],q[44];
cx q[4],q[42];
u1(1.0) q[42];
cx q[4],q[42];
cx q[2],q[42];
u1(1.0) q[42];
cx q[2],q[42];
cx q[2],q[11];
u1(1.0) q[11];
cx q[2],q[11];
cx q[2],q[13];
u1(1.0) q[13];
cx q[2],q[13];
cx q[0],q[13];
u1(1.0) q[13];
cx q[0],q[13];
cx q[0],q[12];
u1(1.0) q[12];
cx q[0],q[12];
cx q[1],q[13];
u1(1.0) q[13];
cx q[1],q[13];
cx q[3],q[42];
u1(1.0) q[42];
cx q[3],q[42];
cx q[3],q[43];
u1(1.0) q[43];
cx q[3],q[43];
cx q[4],q[43];
u1(1.0) q[43];
cx q[4],q[43];
cx q[6],q[45];
u1(1.0) q[45];
cx q[6],q[45];
cx q[5],q[45];
u1(1.0) q[45];
cx q[5],q[45];
cx q[6],q[46];
u1(1.0) q[46];
cx q[6],q[46];
cx q[7],q[15];
u1(1.0) q[15];
cx q[7],q[15];
cx q[7],q[16];
u1(1.0) q[16];
cx q[7],q[16];
cx q[7],q[19];
u1(1.0) q[19];
cx q[7],q[19];
cx q[19],q[27];
u1(1.0) q[27];
cx q[19],q[27];
cx q[14],q[27];
u1(1.0) q[27];
cx q[14],q[27];
cx q[14],q[22];
u1(1.0) q[22];
cx q[14],q[22];
cx q[22],q[30];
u1(1.0) q[30];
cx q[22],q[30];
cx q[21],q[30];
u1(1.0) q[30];
cx q[21],q[30];
cx q[21],q[33];
u1(1.0) q[33];
cx q[21],q[33];
cx q[22],q[34];
u1(1.0) q[34];
cx q[22],q[34];
cx q[22],q[31];
u1(1.0) q[31];
cx q[22],q[31];
cx q[21],q[31];
u1(1.0) q[31];
cx q[21],q[31];
cx q[22],q[32];
u1(1.0) q[32];
cx q[22],q[32];
cx q[21],q[32];
u1(1.0) q[32];
cx q[21],q[32];
cx q[24],q[33];
u1(1.0) q[33];
cx q[24],q[33];
cx q[14],q[24];
u1(1.0) q[24];
cx q[14],q[24];
cx q[23],q[33];
u1(1.0) q[33];
cx q[23],q[33];
cx q[22],q[33];
u1(1.0) q[33];
cx q[22],q[33];
cx q[25],q[34];
u1(1.0) q[34];
cx q[25],q[34];
cx q[16],q[25];
u1(1.0) q[25];
cx q[16],q[25];
cx q[15],q[25];
u1(1.0) q[25];
cx q[15],q[25];
cx q[15],q[27];
u1(1.0) q[27];
cx q[15],q[27];
cx q[18],q[27];
u1(1.0) q[27];
cx q[18],q[27];
cx q[31],q[41];
u1(1.0) q[41];
cx q[31],q[41];
cx q[30],q[41];
u1(1.0) q[41];
cx q[30],q[41];
cx q[30],q[40];
u1(1.0) q[40];
cx q[30],q[40];
cx q[28],q[40];
u1(1.0) q[40];
cx q[28],q[40];
cx q[28],q[39];
u1(1.0) q[39];
cx q[28],q[39];
cx q[29],q[39];
u1(1.0) q[39];
cx q[29],q[39];
cx q[29],q[38];
u1(1.0) q[38];
cx q[29],q[38];
cx q[29],q[40];
u1(1.0) q[40];
cx q[29],q[40];
cx q[30],q[39];
u1(1.0) q[39];
cx q[30],q[39];
cx q[38],q[44];
u1(1.0) q[44];
cx q[38],q[44];
cx q[41],q[42];
u1(1.0) q[42];
cx q[41],q[42];
cx q[9],q[18];
u1(1.0) q[18];
cx q[9],q[18];
cx q[9],q[17];
u1(1.0) q[17];
cx q[9],q[17];
cx q[17],q[26];
u1(1.0) q[26];
cx q[17],q[26];
cx q[14],q[26];
u1(1.0) q[26];
cx q[14],q[26];
cx q[14],q[25];
u1(1.0) q[25];
cx q[14],q[25];
cx q[26],q[34];
u1(1.0) q[34];
cx q[26],q[34];
cx q[16],q[26];
u1(1.0) q[26];
cx q[16],q[26];
cx q[15],q[26];
u1(1.0) q[26];
cx q[15],q[26];
cx q[16],q[27];
u1(1.0) q[27];
cx q[16],q[27];
cx q[23],q[34];
u1(1.0) q[34];
cx q[23],q[34];
cx q[24],q[34];
u1(1.0) q[34];
cx q[24],q[34];
cx q[8],q[17];
u1(1.0) q[17];
cx q[8],q[17];
cx q[17],q[27];
u1(1.0) q[27];
cx q[17],q[27];
cx q[7],q[17];
u1(1.0) q[17];
cx q[7],q[17];
cx q[7],q[18];
u1(1.0) q[18];
cx q[7],q[18];
cx q[8],q[20];
u1(1.0) q[20];
cx q[8],q[20];
cx q[8],q[19];
u1(1.0) q[19];
cx q[8],q[19];
cx q[10],q[19];
u1(1.0) q[19];
cx q[10],q[19];
cx q[10],q[20];
u1(1.0) q[20];
cx q[10],q[20];
cx q[8],q[18];
u1(1.0) q[18];
cx q[8],q[18];
cx q[9],q[20];
u1(1.0) q[20];
cx q[9],q[20];
cx q[11],q[20];
u1(1.0) q[20];
cx q[11],q[20];
cx q[9],q[19];
u1(1.0) q[19];
cx q[9],q[19];
