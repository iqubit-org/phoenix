OPENQASM 2.0;
include "qelib1.inc";
qreg q[64];
cx q[0],q[9];
rz(1.0) q[9];
cx q[0],q[9];
rz(1.0) q[0];
cx q[0],q[10];
rz(1.0) q[10];
cx q[0],q[10];
rz(1.0) q[10];
cx q[3],q[10];
rz(1.0) q[10];
cx q[3],q[10];
rz(1.0) q[3];
cx q[3],q[9];
rz(1.0) q[9];
cx q[3],q[9];
rz(1.0) q[9];
cx q[2],q[9];
rz(1.0) q[9];
cx q[2],q[9];
rz(1.0) q[2];
cx q[2],q[11];
rz(1.0) q[11];
cx q[2],q[11];
rz(1.0) q[11];
cx q[1],q[11];
rz(1.0) q[11];
cx q[1],q[11];
rz(1.0) q[1];
cx q[1],q[8];
rz(1.0) q[8];
cx q[1],q[8];
rz(1.0) q[8];
cx q[2],q[8];
rz(1.0) q[8];
cx q[2],q[8];
cx q[3],q[8];
rz(1.0) q[8];
cx q[3],q[8];
cx q[3],q[15];
rz(1.0) q[15];
cx q[3],q[15];
rz(1.0) q[15];
cx q[6],q[15];
rz(1.0) q[15];
cx q[6],q[15];
rz(1.0) q[6];
cx q[6],q[12];
rz(1.0) q[12];
cx q[6],q[12];
rz(1.0) q[12];
cx q[5],q[12];
rz(1.0) q[12];
cx q[5],q[12];
rz(1.0) q[5];
cx q[5],q[15];
rz(1.0) q[15];
cx q[5],q[15];
cx q[4],q[15];
rz(1.0) q[15];
cx q[4],q[15];
rz(1.0) q[4];
cx q[4],q[13];
rz(1.0) q[13];
cx q[4],q[13];
rz(1.0) q[13];
cx q[7],q[13];
rz(1.0) q[13];
cx q[7],q[13];
rz(1.0) q[7];
cx q[7],q[14];
rz(1.0) q[14];
cx q[7],q[14];
rz(1.0) q[14];
cx q[4],q[14];
rz(1.0) q[14];
cx q[4],q[14];
cx q[5],q[14];
rz(1.0) q[14];
cx q[5],q[14];
cx q[2],q[14];
rz(1.0) q[14];
cx q[2],q[14];
cx q[2],q[15];
rz(1.0) q[15];
cx q[2],q[15];
cx q[1],q[15];
rz(1.0) q[15];
cx q[1],q[15];
cx q[1],q[10];
rz(1.0) q[10];
cx q[1],q[10];
cx q[1],q[13];
rz(1.0) q[13];
cx q[1],q[13];
cx q[6],q[13];
rz(1.0) q[13];
cx q[6],q[13];
cx q[6],q[10];
rz(1.0) q[10];
cx q[6],q[10];
cx q[6],q[11];
rz(1.0) q[11];
cx q[6],q[11];
cx q[0],q[11];
rz(1.0) q[11];
cx q[0],q[11];
cx q[0],q[12];
rz(1.0) q[12];
cx q[0],q[12];
cx q[7],q[12];
rz(1.0) q[12];
cx q[7],q[12];
cx q[7],q[11];
rz(1.0) q[11];
cx q[7],q[11];
cx q[7],q[10];
rz(1.0) q[10];
cx q[7],q[10];
cx q[4],q[10];
rz(1.0) q[10];
cx q[4],q[10];
cx q[4],q[8];
rz(1.0) q[8];
cx q[4],q[8];
cx q[4],q[9];
rz(1.0) q[9];
cx q[4],q[9];
cx q[5],q[9];
rz(1.0) q[9];
cx q[5],q[9];
cx q[5],q[8];
rz(1.0) q[8];
cx q[5],q[8];
cx q[5],q[11];
rz(1.0) q[11];
cx q[5],q[11];
cx q[4],q[11];
rz(1.0) q[11];
cx q[4],q[11];
cx q[4],q[61];
rz(1.0) q[61];
cx q[4],q[61];
rz(1.0) q[61];
cx q[52],q[61];
rz(1.0) q[61];
cx q[52],q[61];
rz(1.0) q[52];
cx q[52],q[62];
rz(1.0) q[62];
cx q[52],q[62];
rz(1.0) q[62];
cx q[55],q[62];
rz(1.0) q[62];
cx q[55],q[62];
rz(1.0) q[55];
cx q[55],q[61];
rz(1.0) q[61];
cx q[55],q[61];
cx q[54],q[61];
rz(1.0) q[61];
cx q[54],q[61];
rz(1.0) q[54];
cx q[54],q[63];
rz(1.0) q[63];
cx q[54],q[63];
rz(1.0) q[63];
cx q[53],q[63];
rz(1.0) q[63];
cx q[53],q[63];
rz(1.0) q[53];
cx q[53],q[60];
rz(1.0) q[60];
cx q[53],q[60];
rz(1.0) q[60];
cx q[54],q[60];
rz(1.0) q[60];
cx q[54],q[60];
cx q[55],q[60];
rz(1.0) q[60];
cx q[55],q[60];
cx q[48],q[60];
rz(1.0) q[60];
cx q[48],q[60];
rz(1.0) q[48];
cx q[48],q[57];
rz(1.0) q[57];
cx q[48],q[57];
rz(1.0) q[57];
cx q[51],q[57];
rz(1.0) q[57];
cx q[51],q[57];
rz(1.0) q[51];
cx q[51],q[58];
rz(1.0) q[58];
cx q[51],q[58];
rz(1.0) q[58];
cx q[48],q[58];
rz(1.0) q[58];
cx q[48],q[58];
cx q[48],q[59];
rz(1.0) q[59];
cx q[48],q[59];
rz(1.0) q[59];
cx q[50],q[59];
rz(1.0) q[59];
cx q[50],q[59];
rz(1.0) q[50];
cx q[50],q[56];
rz(1.0) q[56];
cx q[50],q[56];
rz(1.0) q[56];
cx q[49],q[56];
rz(1.0) q[56];
cx q[49],q[56];
rz(1.0) q[49];
cx q[49],q[59];
rz(1.0) q[59];
cx q[49],q[59];
cx q[49],q[58];
rz(1.0) q[58];
cx q[49],q[58];
cx q[49],q[61];
rz(1.0) q[61];
cx q[49],q[61];
cx q[48],q[61];
rz(1.0) q[61];
cx q[48],q[61];
cx q[48],q[62];
rz(1.0) q[62];
cx q[48],q[62];
cx q[53],q[62];
rz(1.0) q[62];
cx q[53],q[62];
cx q[50],q[62];
rz(1.0) q[62];
cx q[50],q[62];
cx q[50],q[57];
rz(1.0) q[57];
cx q[50],q[57];
cx q[53],q[57];
rz(1.0) q[57];
cx q[53],q[57];
cx q[52],q[57];
rz(1.0) q[57];
cx q[52],q[57];
cx q[52],q[63];
rz(1.0) q[63];
cx q[52],q[63];
cx q[51],q[63];
rz(1.0) q[63];
cx q[51],q[63];
cx q[51],q[56];
rz(1.0) q[56];
cx q[51],q[56];
cx q[52],q[56];
rz(1.0) q[56];
cx q[52],q[56];
cx q[53],q[56];
rz(1.0) q[56];
cx q[53],q[56];
cx q[53],q[59];
rz(1.0) q[59];
cx q[53],q[59];
cx q[55],q[59];
rz(1.0) q[59];
cx q[55],q[59];
cx q[54],q[59];
rz(1.0) q[59];
cx q[54],q[59];
cx q[54],q[58];
rz(1.0) q[58];
cx q[54],q[58];
cx q[55],q[58];
rz(1.0) q[58];
cx q[55],q[58];
cx q[52],q[58];
rz(1.0) q[58];
cx q[52],q[58];
cx q[52],q[59];
rz(1.0) q[59];
cx q[52],q[59];
cx q[2],q[59];
rz(1.0) q[59];
cx q[2],q[59];
cx q[2],q[12];
rz(1.0) q[12];
cx q[2],q[12];
cx q[1],q[12];
rz(1.0) q[12];
cx q[1],q[12];
cx q[1],q[14];
rz(1.0) q[14];
cx q[1],q[14];
cx q[3],q[14];
rz(1.0) q[14];
cx q[3],q[14];
cx q[0],q[14];
rz(1.0) q[14];
cx q[0],q[14];
cx q[0],q[13];
rz(1.0) q[13];
cx q[0],q[13];
cx q[3],q[13];
rz(1.0) q[13];
cx q[3],q[13];
cx q[2],q[13];
rz(1.0) q[13];
cx q[2],q[13];
cx q[13],q[20];
rz(1.0) q[20];
cx q[13],q[20];
rz(1.0) q[20];
cx q[20],q[29];
rz(1.0) q[29];
cx q[20],q[29];
rz(1.0) q[29];
cx q[23],q[29];
rz(1.0) q[29];
cx q[23],q[29];
rz(1.0) q[23];
cx q[23],q[30];
rz(1.0) q[30];
cx q[23],q[30];
rz(1.0) q[30];
cx q[20],q[30];
rz(1.0) q[30];
cx q[20],q[30];
cx q[20],q[31];
rz(1.0) q[31];
cx q[20],q[31];
rz(1.0) q[31];
cx q[22],q[31];
rz(1.0) q[31];
cx q[22],q[31];
rz(1.0) q[22];
cx q[22],q[28];
rz(1.0) q[28];
cx q[22],q[28];
rz(1.0) q[28];
cx q[21],q[28];
rz(1.0) q[28];
cx q[21],q[28];
rz(1.0) q[21];
cx q[21],q[31];
rz(1.0) q[31];
cx q[21],q[31];
cx q[21],q[30];
rz(1.0) q[30];
cx q[21],q[30];
cx q[18],q[30];
rz(1.0) q[30];
cx q[18],q[30];
rz(1.0) q[18];
cx q[18],q[27];
rz(1.0) q[27];
cx q[18],q[27];
rz(1.0) q[27];
cx q[17],q[27];
rz(1.0) q[27];
cx q[17],q[27];
rz(1.0) q[17];
cx q[17],q[24];
rz(1.0) q[24];
cx q[17],q[24];
rz(1.0) q[24];
cx q[18],q[24];
rz(1.0) q[24];
cx q[18],q[24];
cx q[18],q[25];
rz(1.0) q[25];
cx q[18],q[25];
rz(1.0) q[25];
cx q[16],q[25];
rz(1.0) q[25];
cx q[16],q[25];
rz(1.0) q[16];
cx q[16],q[26];
rz(1.0) q[26];
cx q[16],q[26];
rz(1.0) q[26];
cx q[19],q[26];
rz(1.0) q[26];
cx q[19],q[26];
rz(1.0) q[19];
cx q[19],q[25];
rz(1.0) q[25];
cx q[19],q[25];
cx q[19],q[24];
rz(1.0) q[24];
cx q[19],q[24];
cx q[19],q[31];
rz(1.0) q[31];
cx q[19],q[31];
cx q[18],q[31];
rz(1.0) q[31];
cx q[18],q[31];
cx q[17],q[31];
rz(1.0) q[31];
cx q[17],q[31];
cx q[17],q[26];
rz(1.0) q[26];
cx q[17],q[26];
cx q[17],q[29];
rz(1.0) q[29];
cx q[17],q[29];
cx q[22],q[29];
rz(1.0) q[29];
cx q[22],q[29];
cx q[22],q[26];
rz(1.0) q[26];
cx q[22],q[26];
cx q[22],q[27];
rz(1.0) q[27];
cx q[22],q[27];
cx q[16],q[27];
rz(1.0) q[27];
cx q[16],q[27];
cx q[16],q[28];
rz(1.0) q[28];
cx q[16],q[28];
cx q[23],q[28];
rz(1.0) q[28];
cx q[23],q[28];
cx q[23],q[27];
rz(1.0) q[27];
cx q[23],q[27];
cx q[23],q[26];
rz(1.0) q[26];
cx q[23],q[26];
cx q[20],q[26];
rz(1.0) q[26];
cx q[20],q[26];
cx q[20],q[24];
rz(1.0) q[24];
cx q[20],q[24];
cx q[20],q[25];
rz(1.0) q[25];
cx q[20],q[25];
cx q[21],q[25];
rz(1.0) q[25];
cx q[21],q[25];
cx q[21],q[24];
rz(1.0) q[24];
cx q[21],q[24];
cx q[21],q[27];
rz(1.0) q[27];
cx q[21],q[27];
cx q[20],q[27];
rz(1.0) q[27];
cx q[20],q[27];
cx q[27],q[34];
rz(1.0) q[34];
cx q[27],q[34];
rz(1.0) q[34];
cx q[34],q[43];
rz(1.0) q[43];
cx q[34],q[43];
rz(1.0) q[43];
cx q[33],q[43];
rz(1.0) q[43];
cx q[33],q[43];
rz(1.0) q[33];
cx q[33],q[40];
rz(1.0) q[40];
cx q[33],q[40];
rz(1.0) q[40];
cx q[34],q[40];
rz(1.0) q[40];
cx q[34],q[40];
cx q[34],q[41];
rz(1.0) q[41];
cx q[34],q[41];
rz(1.0) q[41];
cx q[32],q[41];
rz(1.0) q[41];
cx q[32],q[41];
rz(1.0) q[32];
cx q[32],q[42];
rz(1.0) q[42];
cx q[32],q[42];
rz(1.0) q[42];
cx q[35],q[42];
rz(1.0) q[42];
cx q[35],q[42];
rz(1.0) q[35];
cx q[35],q[41];
rz(1.0) q[41];
cx q[35],q[41];
cx q[35],q[40];
rz(1.0) q[40];
cx q[35],q[40];
cx q[35],q[47];
rz(1.0) q[47];
cx q[35],q[47];
rz(1.0) q[47];
cx q[38],q[47];
rz(1.0) q[47];
cx q[38],q[47];
rz(1.0) q[38];
cx q[38],q[44];
rz(1.0) q[44];
cx q[38],q[44];
rz(1.0) q[44];
cx q[37],q[44];
rz(1.0) q[44];
cx q[37],q[44];
rz(1.0) q[37];
cx q[37],q[47];
rz(1.0) q[47];
cx q[37],q[47];
cx q[36],q[47];
rz(1.0) q[47];
cx q[36],q[47];
rz(1.0) q[36];
cx q[36],q[45];
rz(1.0) q[45];
cx q[36],q[45];
rz(1.0) q[45];
cx q[39],q[45];
rz(1.0) q[45];
cx q[39],q[45];
rz(1.0) q[39];
cx q[39],q[46];
rz(1.0) q[46];
cx q[39],q[46];
rz(1.0) q[46];
cx q[36],q[46];
rz(1.0) q[46];
cx q[36],q[46];
cx q[37],q[46];
rz(1.0) q[46];
cx q[37],q[46];
cx q[34],q[46];
rz(1.0) q[46];
cx q[34],q[46];
cx q[34],q[47];
rz(1.0) q[47];
cx q[34],q[47];
cx q[33],q[47];
rz(1.0) q[47];
cx q[33],q[47];
cx q[33],q[42];
rz(1.0) q[42];
cx q[33],q[42];
cx q[33],q[45];
rz(1.0) q[45];
cx q[33],q[45];
cx q[38],q[45];
rz(1.0) q[45];
cx q[38],q[45];
cx q[38],q[42];
rz(1.0) q[42];
cx q[38],q[42];
cx q[38],q[43];
rz(1.0) q[43];
cx q[38],q[43];
cx q[32],q[43];
rz(1.0) q[43];
cx q[32],q[43];
cx q[32],q[44];
rz(1.0) q[44];
cx q[32],q[44];
cx q[39],q[44];
rz(1.0) q[44];
cx q[39],q[44];
cx q[39],q[43];
rz(1.0) q[43];
cx q[39],q[43];
cx q[39],q[42];
rz(1.0) q[42];
cx q[39],q[42];
cx q[36],q[42];
rz(1.0) q[42];
cx q[36],q[42];
cx q[36],q[40];
rz(1.0) q[40];
cx q[36],q[40];
cx q[36],q[41];
rz(1.0) q[41];
cx q[36],q[41];
cx q[37],q[41];
rz(1.0) q[41];
cx q[37],q[41];
cx q[37],q[40];
rz(1.0) q[40];
cx q[37],q[40];
cx q[37],q[43];
rz(1.0) q[43];
cx q[37],q[43];
cx q[36],q[43];
rz(1.0) q[43];
cx q[36],q[43];
cx q[29],q[36];
rz(1.0) q[36];
cx q[29],q[36];
cx q[16],q[29];
rz(1.0) q[29];
cx q[16],q[29];
cx q[16],q[30];
rz(1.0) q[30];
cx q[16],q[30];
cx q[19],q[30];
rz(1.0) q[30];
cx q[19],q[30];
cx q[19],q[29];
rz(1.0) q[29];
cx q[19],q[29];
cx q[18],q[29];
rz(1.0) q[29];
cx q[18],q[29];
cx q[18],q[28];
rz(1.0) q[28];
cx q[18],q[28];
cx q[17],q[28];
rz(1.0) q[28];
cx q[17],q[28];
cx q[17],q[30];
rz(1.0) q[30];
cx q[17],q[30];
cx q[8],q[17];
rz(1.0) q[17];
cx q[8],q[17];
cx q[6],q[8];
rz(1.0) q[8];
cx q[6],q[8];
cx q[6],q[9];
rz(1.0) q[9];
cx q[6],q[9];
cx q[7],q[9];
rz(1.0) q[9];
cx q[7],q[9];
cx q[7],q[8];
rz(1.0) q[8];
cx q[7],q[8];
cx q[7],q[62];
rz(1.0) q[62];
cx q[7],q[62];
cx q[51],q[62];
rz(1.0) q[62];
cx q[51],q[62];
cx q[51],q[61];
rz(1.0) q[61];
cx q[51],q[61];
cx q[50],q[61];
rz(1.0) q[61];
cx q[50],q[61];
cx q[50],q[63];
rz(1.0) q[63];
cx q[50],q[63];
cx q[49],q[63];
rz(1.0) q[63];
cx q[49],q[63];
cx q[49],q[60];
rz(1.0) q[60];
cx q[49],q[60];
cx q[50],q[60];
rz(1.0) q[60];
cx q[50],q[60];
cx q[51],q[60];
rz(1.0) q[60];
cx q[51],q[60];
cx q[5],q[60];
rz(1.0) q[60];
cx q[5],q[60];
cx q[5],q[10];
rz(1.0) q[10];
cx q[5],q[10];
cx q[10],q[19];
rz(1.0) q[19];
cx q[10],q[19];
cx q[19],q[28];
rz(1.0) q[28];
cx q[19],q[28];
cx q[28],q[37];
rz(1.0) q[37];
cx q[28],q[37];
cx q[37],q[42];
rz(1.0) q[42];
cx q[37],q[42];
cx q[42],q[51];
rz(1.0) q[51];
cx q[42],q[51];
cx q[41],q[51];
rz(1.0) q[51];
cx q[41],q[51];
cx q[39],q[41];
rz(1.0) q[41];
cx q[39],q[41];
cx q[38],q[41];
rz(1.0) q[41];
cx q[38],q[41];
cx q[38],q[40];
rz(1.0) q[40];
cx q[38],q[40];
cx q[39],q[40];
rz(1.0) q[40];
cx q[39],q[40];
cx q[30],q[39];
rz(1.0) q[39];
cx q[30],q[39];
cx q[29],q[39];
rz(1.0) q[39];
cx q[29],q[39];
cx q[28],q[39];
rz(1.0) q[39];
cx q[28],q[39];
cx q[28],q[38];
rz(1.0) q[38];
cx q[28],q[38];
cx q[31],q[38];
rz(1.0) q[38];
cx q[31],q[38];
cx q[16],q[31];
rz(1.0) q[31];
cx q[16],q[31];
cx q[9],q[16];
rz(1.0) q[16];
cx q[9],q[16];
cx q[9],q[19];
rz(1.0) q[19];
cx q[9],q[19];
cx q[8],q[19];
rz(1.0) q[19];
cx q[8],q[19];
cx q[8],q[18];
rz(1.0) q[18];
cx q[8],q[18];
cx q[11],q[18];
rz(1.0) q[18];
cx q[11],q[18];
cx q[11],q[17];
rz(1.0) q[17];
cx q[11],q[17];
cx q[10],q[17];
rz(1.0) q[17];
cx q[10],q[17];
cx q[10],q[16];
rz(1.0) q[16];
cx q[10],q[16];
cx q[11],q[16];
rz(1.0) q[16];
cx q[11],q[16];
cx q[11],q[23];
rz(1.0) q[23];
cx q[11],q[23];
cx q[23],q[25];
rz(1.0) q[25];
cx q[23],q[25];
cx q[22],q[25];
rz(1.0) q[25];
cx q[22],q[25];
cx q[22],q[24];
rz(1.0) q[24];
cx q[22],q[24];
cx q[23],q[24];
rz(1.0) q[24];
cx q[23],q[24];
cx q[14],q[23];
rz(1.0) q[23];
cx q[14],q[23];
cx q[13],q[23];
rz(1.0) q[23];
cx q[13],q[23];
cx q[12],q[23];
rz(1.0) q[23];
cx q[12],q[23];
cx q[3],q[12];
rz(1.0) q[12];
cx q[3],q[12];
cx q[3],q[58];
rz(1.0) q[58];
cx q[3],q[58];
cx q[53],q[58];
rz(1.0) q[58];
cx q[53],q[58];
cx q[44],q[53];
rz(1.0) q[53];
cx q[44],q[53];
cx q[33],q[44];
rz(1.0) q[44];
cx q[33],q[44];
cx q[34],q[44];
rz(1.0) q[44];
cx q[34],q[44];
cx q[34],q[45];
rz(1.0) q[45];
cx q[34],q[45];
cx q[32],q[45];
rz(1.0) q[45];
cx q[32],q[45];
cx q[32],q[46];
rz(1.0) q[46];
cx q[32],q[46];
cx q[35],q[46];
rz(1.0) q[46];
cx q[35],q[46];
cx q[35],q[45];
rz(1.0) q[45];
cx q[35],q[45];
cx q[35],q[44];
rz(1.0) q[44];
cx q[35],q[44];
cx q[26],q[35];
rz(1.0) q[35];
cx q[26],q[35];
cx q[21],q[26];
rz(1.0) q[26];
cx q[21],q[26];
cx q[12],q[21];
rz(1.0) q[21];
cx q[12],q[21];
cx q[12],q[22];
rz(1.0) q[22];
cx q[12],q[22];
cx q[15],q[22];
rz(1.0) q[22];
cx q[15],q[22];
cx q[0],q[15];
rz(1.0) q[15];
cx q[0],q[15];
cx q[0],q[57];
rz(1.0) q[57];
cx q[0],q[57];
cx q[55],q[57];
rz(1.0) q[57];
cx q[55],q[57];
cx q[54],q[57];
rz(1.0) q[57];
cx q[54],q[57];
cx q[54],q[56];
rz(1.0) q[56];
cx q[54],q[56];
cx q[55],q[56];
rz(1.0) q[56];
cx q[55],q[56];
cx q[1],q[56];
rz(1.0) q[56];
cx q[1],q[56];
cx q[1],q[59];
rz(1.0) q[59];
cx q[1],q[59];
cx q[0],q[59];
rz(1.0) q[59];
cx q[0],q[59];
cx q[0],q[58];
rz(1.0) q[58];
cx q[0],q[58];
cx q[1],q[58];
rz(1.0) q[58];
cx q[1],q[58];
cx q[1],q[61];
rz(1.0) q[61];
cx q[1],q[61];
cx q[7],q[61];
rz(1.0) q[61];
cx q[7],q[61];
cx q[6],q[61];
rz(1.0) q[61];
cx q[6],q[61];
cx q[6],q[63];
rz(1.0) q[63];
cx q[6],q[63];
cx q[48],q[63];
rz(1.0) q[63];
cx q[48],q[63];
cx q[41],q[48];
rz(1.0) q[48];
cx q[41],q[48];
cx q[42],q[48];
rz(1.0) q[48];
cx q[42],q[48];
cx q[42],q[49];
rz(1.0) q[49];
cx q[42],q[49];
cx q[49],q[62];
rz(1.0) q[62];
cx q[49],q[62];
cx q[40],q[49];
rz(1.0) q[49];
cx q[40],q[49];
cx q[40],q[50];
rz(1.0) q[50];
cx q[40],q[50];
cx q[43],q[50];
rz(1.0) q[50];
cx q[43],q[50];
cx q[43],q[49];
rz(1.0) q[49];
cx q[43],q[49];
cx q[43],q[48];
rz(1.0) q[48];
cx q[43],q[48];
cx q[43],q[55];
rz(1.0) q[55];
cx q[43],q[55];
cx q[46],q[55];
rz(1.0) q[55];
cx q[46],q[55];
cx q[33],q[46];
rz(1.0) q[46];
cx q[33],q[46];
cx q[24],q[33];
rz(1.0) q[33];
cx q[24],q[33];
cx q[24],q[34];
rz(1.0) q[34];
cx q[24],q[34];
cx q[24],q[35];
rz(1.0) q[35];
cx q[24],q[35];
cx q[25],q[35];
rz(1.0) q[35];
cx q[25],q[35];
cx q[25],q[32];
rz(1.0) q[32];
cx q[25],q[32];
cx q[32],q[47];
rz(1.0) q[47];
cx q[32],q[47];
cx q[47],q[54];
rz(1.0) q[54];
cx q[47],q[54];
cx q[44],q[54];
rz(1.0) q[54];
cx q[44],q[54];
cx q[44],q[55];
rz(1.0) q[55];
cx q[44],q[55];
cx q[45],q[55];
rz(1.0) q[55];
cx q[45],q[55];
cx q[45],q[52];
rz(1.0) q[52];
cx q[45],q[52];
cx q[46],q[52];
rz(1.0) q[52];
cx q[46],q[52];
cx q[46],q[53];
rz(1.0) q[53];
cx q[46],q[53];
cx q[47],q[53];
rz(1.0) q[53];
cx q[47],q[53];
cx q[47],q[52];
rz(1.0) q[52];
cx q[47],q[52];
cx q[40],q[52];
rz(1.0) q[52];
cx q[40],q[52];
cx q[40],q[51];
rz(1.0) q[51];
cx q[40],q[51];
cx q[47],q[51];
rz(1.0) q[51];
cx q[47],q[51];
cx q[46],q[51];
rz(1.0) q[51];
cx q[46],q[51];
cx q[46],q[50];
rz(1.0) q[50];
cx q[46],q[50];
cx q[41],q[50];
rz(1.0) q[50];
cx q[41],q[50];
cx q[41],q[53];
rz(1.0) q[53];
cx q[41],q[53];
cx q[40],q[53];
rz(1.0) q[53];
cx q[40],q[53];
cx q[40],q[54];
rz(1.0) q[54];
cx q[40],q[54];
cx q[45],q[54];
rz(1.0) q[54];
cx q[45],q[54];
cx q[42],q[54];
rz(1.0) q[54];
cx q[42],q[54];
cx q[42],q[55];
rz(1.0) q[55];
cx q[42],q[55];
cx q[41],q[55];
rz(1.0) q[55];
cx q[41],q[55];
cx q[41],q[52];
rz(1.0) q[52];
cx q[41],q[52];
cx q[42],q[52];
rz(1.0) q[52];
cx q[42],q[52];
cx q[42],q[53];
rz(1.0) q[53];
cx q[42],q[53];
cx q[43],q[53];
rz(1.0) q[53];
cx q[43],q[53];
cx q[43],q[54];
rz(1.0) q[54];
cx q[43],q[54];
cx q[41],q[54];
rz(1.0) q[54];
cx q[41],q[54];
cx q[2],q[56];
rz(1.0) q[56];
cx q[2],q[56];
cx q[2],q[57];
rz(1.0) q[57];
cx q[2],q[57];
cx q[3],q[57];
rz(1.0) q[57];
cx q[3],q[57];
cx q[3],q[56];
rz(1.0) q[56];
cx q[3],q[56];
cx q[3],q[63];
rz(1.0) q[63];
cx q[3],q[63];
cx q[5],q[63];
rz(1.0) q[63];
cx q[5],q[63];
cx q[4],q[63];
rz(1.0) q[63];
cx q[4],q[63];
cx q[4],q[62];
rz(1.0) q[62];
cx q[4],q[62];
cx q[5],q[62];
rz(1.0) q[62];
cx q[5],q[62];
cx q[2],q[62];
rz(1.0) q[62];
cx q[2],q[62];
cx q[2],q[63];
rz(1.0) q[63];
cx q[2],q[63];
cx q[1],q[63];
rz(1.0) q[63];
cx q[1],q[63];
cx q[1],q[60];
rz(1.0) q[60];
cx q[1],q[60];
cx q[6],q[60];
rz(1.0) q[60];
cx q[6],q[60];
cx q[7],q[60];
rz(1.0) q[60];
cx q[7],q[60];
cx q[0],q[60];
rz(1.0) q[60];
cx q[0],q[60];
cx q[0],q[61];
rz(1.0) q[61];
cx q[0],q[61];
cx q[0],q[62];
rz(1.0) q[62];
cx q[0],q[62];
cx q[3],q[62];
rz(1.0) q[62];
cx q[3],q[62];
cx q[3],q[61];
rz(1.0) q[61];
cx q[3],q[61];
cx q[2],q[61];
rz(1.0) q[61];
cx q[2],q[61];
cx q[2],q[60];
rz(1.0) q[60];
cx q[2],q[60];
cx q[3],q[60];
rz(1.0) q[60];
cx q[3],q[60];
cx q[14],q[20];
rz(1.0) q[20];
cx q[14],q[20];
cx q[14],q[21];
rz(1.0) q[21];
cx q[14],q[21];
cx q[15],q[21];
rz(1.0) q[21];
cx q[15],q[21];
cx q[15],q[20];
rz(1.0) q[20];
cx q[15],q[20];
cx q[8],q[20];
rz(1.0) q[20];
cx q[8],q[20];
cx q[8],q[21];
rz(1.0) q[21];
cx q[8],q[21];
cx q[9],q[21];
rz(1.0) q[21];
cx q[9],q[21];
cx q[9],q[18];
rz(1.0) q[18];
cx q[9],q[18];
cx q[14],q[18];
rz(1.0) q[18];
cx q[14],q[18];
cx q[14],q[19];
rz(1.0) q[19];
cx q[14],q[19];
cx q[15],q[19];
rz(1.0) q[19];
cx q[15],q[19];
cx q[15],q[18];
rz(1.0) q[18];
cx q[15],q[18];
cx q[12],q[18];
rz(1.0) q[18];
cx q[12],q[18];
cx q[12],q[16];
rz(1.0) q[16];
cx q[12],q[16];
cx q[12],q[17];
rz(1.0) q[17];
cx q[12],q[17];
cx q[13],q[17];
rz(1.0) q[17];
cx q[13],q[17];
cx q[13],q[22];
rz(1.0) q[22];
cx q[13],q[22];
cx q[10],q[22];
rz(1.0) q[22];
cx q[10],q[22];
cx q[10],q[23];
rz(1.0) q[23];
cx q[10],q[23];
cx q[9],q[23];
rz(1.0) q[23];
cx q[9],q[23];
cx q[9],q[20];
rz(1.0) q[20];
cx q[9],q[20];
cx q[10],q[20];
rz(1.0) q[20];
cx q[10],q[20];
cx q[10],q[21];
rz(1.0) q[21];
cx q[10],q[21];
cx q[11],q[21];
rz(1.0) q[21];
cx q[11],q[21];
cx q[11],q[22];
rz(1.0) q[22];
cx q[11],q[22];
cx q[8],q[22];
rz(1.0) q[22];
cx q[8],q[22];
cx q[8],q[23];
rz(1.0) q[23];
cx q[8],q[23];
cx q[26],q[32];
rz(1.0) q[32];
cx q[26],q[32];
cx q[26],q[33];
rz(1.0) q[33];
cx q[26],q[33];
cx q[27],q[33];
rz(1.0) q[33];
cx q[27],q[33];
cx q[27],q[32];
rz(1.0) q[32];
cx q[27],q[32];
cx q[27],q[39];
rz(1.0) q[39];
cx q[27],q[39];
cx q[26],q[39];
rz(1.0) q[39];
cx q[26],q[39];
cx q[26],q[38];
rz(1.0) q[38];
cx q[26],q[38];
cx q[29],q[38];
rz(1.0) q[38];
cx q[29],q[38];
cx q[29],q[33];
rz(1.0) q[33];
cx q[29],q[33];
cx q[28],q[33];
rz(1.0) q[33];
cx q[28],q[33];
cx q[28],q[32];
rz(1.0) q[32];
cx q[28],q[32];
cx q[29],q[32];
rz(1.0) q[32];
cx q[29],q[32];
cx q[29],q[35];
rz(1.0) q[35];
cx q[29],q[35];
cx q[31],q[35];
rz(1.0) q[35];
cx q[31],q[35];
cx q[31],q[37];
rz(1.0) q[37];
cx q[31],q[37];
cx q[30],q[37];
rz(1.0) q[37];
cx q[30],q[37];
cx q[30],q[36];
rz(1.0) q[36];
cx q[30],q[36];
cx q[31],q[36];
rz(1.0) q[36];
cx q[31],q[36];
cx q[24],q[36];
rz(1.0) q[36];
cx q[24],q[36];
cx q[24],q[37];
rz(1.0) q[37];
cx q[24],q[37];
cx q[25],q[37];
rz(1.0) q[37];
cx q[25],q[37];
cx q[25],q[34];
rz(1.0) q[34];
cx q[25],q[34];
cx q[30],q[34];
rz(1.0) q[34];
cx q[30],q[34];
cx q[30],q[35];
rz(1.0) q[35];
cx q[30],q[35];
cx q[30],q[32];
rz(1.0) q[32];
cx q[30],q[32];
cx q[30],q[33];
rz(1.0) q[33];
cx q[30],q[33];
cx q[31],q[33];
rz(1.0) q[33];
cx q[31],q[33];
cx q[31],q[34];
rz(1.0) q[34];
cx q[31],q[34];
cx q[28],q[34];
rz(1.0) q[34];
cx q[28],q[34];
cx q[28],q[35];
rz(1.0) q[35];
cx q[28],q[35];
cx q[4],q[56];
rz(1.0) q[56];
cx q[4],q[56];
cx q[4],q[57];
rz(1.0) q[57];
cx q[4],q[57];
cx q[5],q[57];
rz(1.0) q[57];
cx q[5],q[57];
cx q[5],q[56];
rz(1.0) q[56];
cx q[5],q[56];
cx q[5],q[59];
rz(1.0) q[59];
cx q[5],q[59];
cx q[7],q[59];
rz(1.0) q[59];
cx q[7],q[59];
cx q[6],q[59];
rz(1.0) q[59];
cx q[6],q[59];
cx q[6],q[58];
rz(1.0) q[58];
cx q[6],q[58];
cx q[7],q[58];
rz(1.0) q[58];
cx q[7],q[58];
cx q[4],q[58];
rz(1.0) q[58];
cx q[4],q[58];
cx q[4],q[59];
rz(1.0) q[59];
cx q[4],q[59];
cx q[44],q[48];
rz(1.0) q[48];
cx q[44],q[48];
cx q[44],q[49];
rz(1.0) q[49];
cx q[44],q[49];
cx q[45],q[49];
rz(1.0) q[49];
cx q[45],q[49];
cx q[45],q[48];
rz(1.0) q[48];
cx q[45],q[48];
cx q[45],q[51];
rz(1.0) q[51];
cx q[45],q[51];
cx q[44],q[51];
rz(1.0) q[51];
cx q[44],q[51];
cx q[44],q[50];
rz(1.0) q[50];
cx q[44],q[50];
cx q[47],q[50];
rz(1.0) q[50];
cx q[47],q[50];
cx q[47],q[49];
rz(1.0) q[49];
cx q[47],q[49];
cx q[46],q[49];
rz(1.0) q[49];
cx q[46],q[49];
cx q[46],q[48];
rz(1.0) q[48];
cx q[46],q[48];
cx q[47],q[48];
rz(1.0) q[48];
cx q[47],q[48];
cx q[13],q[16];
rz(1.0) q[16];
cx q[13],q[16];
cx q[13],q[19];
rz(1.0) q[19];
cx q[13],q[19];
cx q[12],q[19];
rz(1.0) q[19];
cx q[12],q[19];
cx q[25],q[36];
rz(1.0) q[36];
cx q[25],q[36];
cx q[25],q[39];
rz(1.0) q[39];
cx q[25],q[39];
cx q[24],q[39];
rz(1.0) q[39];
cx q[24],q[39];
cx q[24],q[38];
rz(1.0) q[38];
cx q[24],q[38];
cx q[27],q[38];
rz(1.0) q[38];
cx q[27],q[38];
cx q[27],q[37];
rz(1.0) q[37];
cx q[27],q[37];
cx q[26],q[37];
rz(1.0) q[37];
cx q[26],q[37];
cx q[26],q[36];
rz(1.0) q[36];
cx q[26],q[36];
cx q[27],q[36];
rz(1.0) q[36];
cx q[27],q[36];
cx q[6],q[56];
rz(1.0) q[56];
cx q[6],q[56];
cx q[6],q[57];
rz(1.0) q[57];
cx q[6],q[57];
cx q[7],q[57];
rz(1.0) q[57];
cx q[7],q[57];
cx q[7],q[56];
rz(1.0) q[56];
cx q[7],q[56];
cx q[14],q[16];
rz(1.0) q[16];
cx q[14],q[16];
cx q[14],q[17];
rz(1.0) q[17];
cx q[14],q[17];
cx q[15],q[17];
rz(1.0) q[17];
cx q[15],q[17];
cx q[15],q[16];
rz(1.0) q[16];
cx q[15],q[16];
cx q[0],q[63];
rz(1.0) q[63];
cx q[0],q[63];
cx q[1],q[62];
rz(1.0) q[62];
cx q[1],q[62];
cx q[5],q[58];
rz(1.0) q[58];
cx q[5],q[58];
cx q[9],q[22];
rz(1.0) q[22];
cx q[9],q[22];
cx q[11],q[20];
rz(1.0) q[20];
cx q[11],q[20];
cx q[13],q[18];
rz(1.0) q[18];
cx q[13],q[18];
cx q[25],q[38];
rz(1.0) q[38];
cx q[25],q[38];
cx q[29],q[34];
rz(1.0) q[34];
cx q[29],q[34];
cx q[31],q[32];
rz(1.0) q[32];
cx q[31],q[32];
cx q[40],q[55];
rz(1.0) q[55];
cx q[40],q[55];
cx q[43],q[52];
rz(1.0) q[52];
cx q[43],q[52];
cx q[45],q[50];
rz(1.0) q[50];
cx q[45],q[50];
