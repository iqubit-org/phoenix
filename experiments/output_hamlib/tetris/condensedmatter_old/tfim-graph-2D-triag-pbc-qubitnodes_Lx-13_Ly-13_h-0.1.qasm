OPENQASM 2.0;
include "qelib1.inc";
qreg q[91];
cx q[0],q[1];
rz(1.0) q[1];
cx q[0],q[1];
u(pi/2,0,pi) q[0];
rz(1.0) q[0];
u(pi/2,0,pi) q[0];
cx q[0],q[6];
rz(1.0) q[6];
cx q[0],q[6];
u(pi/2,0,pi) q[6];
rz(1.0) q[6];
u(pi/2,0,pi) q[6];
cx q[6],q[7];
rz(1.0) q[7];
cx q[6],q[7];
u(pi/2,0,pi) q[7];
rz(1.0) q[7];
u(pi/2,0,pi) q[7];
cx q[7],q[8];
rz(1.0) q[8];
cx q[7],q[8];
u(pi/2,0,pi) q[8];
rz(1.0) q[8];
u(pi/2,0,pi) q[8];
cx q[8],q[9];
rz(1.0) q[9];
cx q[8],q[9];
u(pi/2,0,pi) q[9];
rz(1.0) q[9];
u(pi/2,0,pi) q[9];
cx q[9],q[10];
rz(1.0) q[10];
cx q[9],q[10];
u(pi/2,0,pi) q[10];
rz(1.0) q[10];
u(pi/2,0,pi) q[10];
cx q[10],q[11];
rz(1.0) q[11];
cx q[10],q[11];
u(pi/2,0,pi) q[11];
rz(1.0) q[11];
u(pi/2,0,pi) q[11];
cx q[11],q[12];
rz(1.0) q[12];
cx q[11],q[12];
u(pi/2,0,pi) q[12];
rz(1.0) q[12];
u(pi/2,0,pi) q[12];
cx q[12],q[13];
rz(1.0) q[13];
cx q[12],q[13];
u(pi/2,0,pi) q[13];
rz(1.0) q[13];
u(pi/2,0,pi) q[13];
cx q[13],q[14];
rz(1.0) q[14];
cx q[13],q[14];
u(pi/2,0,pi) q[14];
rz(1.0) q[14];
u(pi/2,0,pi) q[14];
cx q[14],q[15];
rz(1.0) q[15];
cx q[14],q[15];
u(pi/2,0,pi) q[15];
rz(1.0) q[15];
u(pi/2,0,pi) q[15];
cx q[15],q[16];
rz(1.0) q[16];
cx q[15],q[16];
u(pi/2,0,pi) q[16];
rz(1.0) q[16];
u(pi/2,0,pi) q[16];
cx q[16],q[17];
rz(1.0) q[17];
cx q[16],q[17];
u(pi/2,0,pi) q[17];
rz(1.0) q[17];
u(pi/2,0,pi) q[17];
cx q[17],q[18];
rz(1.0) q[18];
cx q[17],q[18];
u(pi/2,0,pi) q[18];
rz(1.0) q[18];
u(pi/2,0,pi) q[18];
cx q[18],q[19];
rz(1.0) q[19];
cx q[18],q[19];
u(pi/2,0,pi) q[19];
rz(1.0) q[19];
u(pi/2,0,pi) q[19];
cx q[19],q[20];
rz(1.0) q[20];
cx q[19],q[20];
u(pi/2,0,pi) q[20];
rz(1.0) q[20];
u(pi/2,0,pi) q[20];
cx q[20],q[21];
rz(1.0) q[21];
cx q[20],q[21];
u(pi/2,0,pi) q[21];
rz(1.0) q[21];
u(pi/2,0,pi) q[21];
cx q[21],q[22];
rz(1.0) q[22];
cx q[21],q[22];
u(pi/2,0,pi) q[22];
rz(1.0) q[22];
u(pi/2,0,pi) q[22];
cx q[22],q[23];
rz(1.0) q[23];
cx q[22],q[23];
u(pi/2,0,pi) q[23];
rz(1.0) q[23];
u(pi/2,0,pi) q[23];
cx q[23],q[24];
rz(1.0) q[24];
cx q[23],q[24];
u(pi/2,0,pi) q[24];
rz(1.0) q[24];
u(pi/2,0,pi) q[24];
cx q[24],q[25];
rz(1.0) q[25];
cx q[24],q[25];
u(pi/2,0,pi) q[25];
rz(1.0) q[25];
u(pi/2,0,pi) q[25];
cx q[25],q[26];
rz(1.0) q[26];
cx q[25],q[26];
u(pi/2,0,pi) q[26];
rz(1.0) q[26];
u(pi/2,0,pi) q[26];
cx q[26],q[27];
rz(1.0) q[27];
cx q[26],q[27];
u(pi/2,0,pi) q[27];
rz(1.0) q[27];
u(pi/2,0,pi) q[27];
cx q[27],q[28];
rz(1.0) q[28];
cx q[27],q[28];
u(pi/2,0,pi) q[28];
rz(1.0) q[28];
u(pi/2,0,pi) q[28];
cx q[28],q[29];
rz(1.0) q[29];
cx q[28],q[29];
u(pi/2,0,pi) q[29];
rz(1.0) q[29];
u(pi/2,0,pi) q[29];
cx q[29],q[30];
rz(1.0) q[30];
cx q[29],q[30];
u(pi/2,0,pi) q[30];
rz(1.0) q[30];
u(pi/2,0,pi) q[30];
cx q[30],q[31];
rz(1.0) q[31];
cx q[30],q[31];
u(pi/2,0,pi) q[31];
rz(1.0) q[31];
u(pi/2,0,pi) q[31];
cx q[31],q[32];
rz(1.0) q[32];
cx q[31],q[32];
u(pi/2,0,pi) q[32];
rz(1.0) q[32];
u(pi/2,0,pi) q[32];
cx q[32],q[33];
rz(1.0) q[33];
cx q[32],q[33];
u(pi/2,0,pi) q[33];
rz(1.0) q[33];
u(pi/2,0,pi) q[33];
cx q[33],q[34];
rz(1.0) q[34];
cx q[33],q[34];
u(pi/2,0,pi) q[34];
rz(1.0) q[34];
u(pi/2,0,pi) q[34];
cx q[34],q[35];
rz(1.0) q[35];
cx q[34],q[35];
u(pi/2,0,pi) q[35];
rz(1.0) q[35];
u(pi/2,0,pi) q[35];
cx q[35],q[36];
rz(1.0) q[36];
cx q[35],q[36];
u(pi/2,0,pi) q[36];
rz(1.0) q[36];
u(pi/2,0,pi) q[36];
cx q[36],q[37];
rz(1.0) q[37];
cx q[36],q[37];
u(pi/2,0,pi) q[37];
rz(1.0) q[37];
u(pi/2,0,pi) q[37];
cx q[37],q[38];
rz(1.0) q[38];
cx q[37],q[38];
u(pi/2,0,pi) q[38];
rz(1.0) q[38];
u(pi/2,0,pi) q[38];
cx q[38],q[39];
rz(1.0) q[39];
cx q[38],q[39];
u(pi/2,0,pi) q[39];
rz(1.0) q[39];
u(pi/2,0,pi) q[39];
cx q[39],q[40];
rz(1.0) q[40];
cx q[39],q[40];
u(pi/2,0,pi) q[40];
rz(1.0) q[40];
u(pi/2,0,pi) q[40];
cx q[40],q[41];
rz(1.0) q[41];
cx q[40],q[41];
u(pi/2,0,pi) q[41];
rz(1.0) q[41];
u(pi/2,0,pi) q[41];
cx q[41],q[42];
rz(1.0) q[42];
cx q[41],q[42];
u(pi/2,0,pi) q[42];
rz(1.0) q[42];
u(pi/2,0,pi) q[42];
cx q[42],q[43];
rz(1.0) q[43];
cx q[42],q[43];
u(pi/2,0,pi) q[43];
rz(1.0) q[43];
u(pi/2,0,pi) q[43];
cx q[43],q[44];
rz(1.0) q[44];
cx q[43],q[44];
u(pi/2,0,pi) q[44];
rz(1.0) q[44];
u(pi/2,0,pi) q[44];
cx q[44],q[45];
rz(1.0) q[45];
cx q[44],q[45];
u(pi/2,0,pi) q[45];
rz(1.0) q[45];
u(pi/2,0,pi) q[45];
cx q[45],q[46];
rz(1.0) q[46];
cx q[45],q[46];
u(pi/2,0,pi) q[46];
rz(1.0) q[46];
u(pi/2,0,pi) q[46];
cx q[46],q[47];
rz(1.0) q[47];
cx q[46],q[47];
u(pi/2,0,pi) q[47];
rz(1.0) q[47];
u(pi/2,0,pi) q[47];
cx q[47],q[48];
rz(1.0) q[48];
cx q[47],q[48];
u(pi/2,0,pi) q[48];
rz(1.0) q[48];
u(pi/2,0,pi) q[48];
cx q[48],q[49];
rz(1.0) q[49];
cx q[48],q[49];
u(pi/2,0,pi) q[49];
rz(1.0) q[49];
u(pi/2,0,pi) q[49];
cx q[49],q[50];
rz(1.0) q[50];
cx q[49],q[50];
u(pi/2,0,pi) q[50];
rz(1.0) q[50];
u(pi/2,0,pi) q[50];
cx q[50],q[51];
rz(1.0) q[51];
cx q[50],q[51];
u(pi/2,0,pi) q[51];
rz(1.0) q[51];
u(pi/2,0,pi) q[51];
cx q[51],q[52];
rz(1.0) q[52];
cx q[51],q[52];
u(pi/2,0,pi) q[52];
rz(1.0) q[52];
u(pi/2,0,pi) q[52];
cx q[52],q[53];
rz(1.0) q[53];
cx q[52],q[53];
u(pi/2,0,pi) q[53];
rz(1.0) q[53];
u(pi/2,0,pi) q[53];
cx q[53],q[54];
rz(1.0) q[54];
cx q[53],q[54];
u(pi/2,0,pi) q[54];
rz(1.0) q[54];
u(pi/2,0,pi) q[54];
cx q[54],q[55];
rz(1.0) q[55];
cx q[54],q[55];
u(pi/2,0,pi) q[55];
rz(1.0) q[55];
u(pi/2,0,pi) q[55];
cx q[55],q[56];
rz(1.0) q[56];
cx q[55],q[56];
u(pi/2,0,pi) q[56];
rz(1.0) q[56];
u(pi/2,0,pi) q[56];
cx q[56],q[57];
rz(1.0) q[57];
cx q[56],q[57];
u(pi/2,0,pi) q[57];
rz(1.0) q[57];
u(pi/2,0,pi) q[57];
cx q[57],q[58];
rz(1.0) q[58];
cx q[57],q[58];
u(pi/2,0,pi) q[58];
rz(1.0) q[58];
u(pi/2,0,pi) q[58];
cx q[58],q[59];
rz(1.0) q[59];
cx q[58],q[59];
u(pi/2,0,pi) q[59];
rz(1.0) q[59];
u(pi/2,0,pi) q[59];
cx q[59],q[60];
rz(1.0) q[60];
cx q[59],q[60];
u(pi/2,0,pi) q[60];
rz(1.0) q[60];
u(pi/2,0,pi) q[60];
cx q[60],q[61];
rz(1.0) q[61];
cx q[60],q[61];
u(pi/2,0,pi) q[61];
rz(1.0) q[61];
u(pi/2,0,pi) q[61];
cx q[61],q[62];
rz(1.0) q[62];
cx q[61],q[62];
u(pi/2,0,pi) q[62];
rz(1.0) q[62];
u(pi/2,0,pi) q[62];
cx q[62],q[63];
rz(1.0) q[63];
cx q[62],q[63];
u(pi/2,0,pi) q[63];
rz(1.0) q[63];
u(pi/2,0,pi) q[63];
cx q[63],q[64];
rz(1.0) q[64];
cx q[63],q[64];
u(pi/2,0,pi) q[64];
rz(1.0) q[64];
u(pi/2,0,pi) q[64];
cx q[64],q[65];
rz(1.0) q[65];
cx q[64],q[65];
u(pi/2,0,pi) q[65];
rz(1.0) q[65];
u(pi/2,0,pi) q[65];
cx q[65],q[66];
rz(1.0) q[66];
cx q[65],q[66];
u(pi/2,0,pi) q[66];
rz(1.0) q[66];
u(pi/2,0,pi) q[66];
cx q[66],q[67];
rz(1.0) q[67];
cx q[66],q[67];
u(pi/2,0,pi) q[67];
rz(1.0) q[67];
u(pi/2,0,pi) q[67];
cx q[67],q[68];
rz(1.0) q[68];
cx q[67],q[68];
u(pi/2,0,pi) q[68];
rz(1.0) q[68];
u(pi/2,0,pi) q[68];
cx q[68],q[69];
rz(1.0) q[69];
cx q[68],q[69];
u(pi/2,0,pi) q[69];
rz(1.0) q[69];
u(pi/2,0,pi) q[69];
cx q[69],q[70];
rz(1.0) q[70];
cx q[69],q[70];
u(pi/2,0,pi) q[70];
rz(1.0) q[70];
u(pi/2,0,pi) q[70];
cx q[70],q[71];
rz(1.0) q[71];
cx q[70],q[71];
u(pi/2,0,pi) q[71];
rz(1.0) q[71];
u(pi/2,0,pi) q[71];
cx q[71],q[72];
rz(1.0) q[72];
cx q[71],q[72];
u(pi/2,0,pi) q[72];
rz(1.0) q[72];
u(pi/2,0,pi) q[72];
cx q[72],q[73];
rz(1.0) q[73];
cx q[72],q[73];
u(pi/2,0,pi) q[73];
rz(1.0) q[73];
u(pi/2,0,pi) q[73];
cx q[73],q[74];
rz(1.0) q[74];
cx q[73],q[74];
u(pi/2,0,pi) q[74];
rz(1.0) q[74];
u(pi/2,0,pi) q[74];
cx q[74],q[75];
rz(1.0) q[75];
cx q[74],q[75];
u(pi/2,0,pi) q[75];
rz(1.0) q[75];
u(pi/2,0,pi) q[75];
cx q[75],q[76];
rz(1.0) q[76];
cx q[75],q[76];
u(pi/2,0,pi) q[76];
rz(1.0) q[76];
u(pi/2,0,pi) q[76];
cx q[76],q[77];
rz(1.0) q[77];
cx q[76],q[77];
u(pi/2,0,pi) q[77];
rz(1.0) q[77];
u(pi/2,0,pi) q[77];
cx q[77],q[78];
rz(1.0) q[78];
cx q[77],q[78];
u(pi/2,0,pi) q[78];
rz(1.0) q[78];
u(pi/2,0,pi) q[78];
cx q[78],q[79];
rz(1.0) q[79];
cx q[78],q[79];
u(pi/2,0,pi) q[79];
rz(1.0) q[79];
u(pi/2,0,pi) q[79];
cx q[79],q[80];
rz(1.0) q[80];
cx q[79],q[80];
u(pi/2,0,pi) q[80];
rz(1.0) q[80];
u(pi/2,0,pi) q[80];
cx q[80],q[81];
rz(1.0) q[81];
cx q[80],q[81];
u(pi/2,0,pi) q[81];
rz(1.0) q[81];
u(pi/2,0,pi) q[81];
cx q[81],q[82];
rz(1.0) q[82];
cx q[81],q[82];
u(pi/2,0,pi) q[82];
rz(1.0) q[82];
u(pi/2,0,pi) q[82];
cx q[82],q[83];
rz(1.0) q[83];
cx q[82],q[83];
u(pi/2,0,pi) q[83];
rz(1.0) q[83];
u(pi/2,0,pi) q[83];
cx q[83],q[84];
rz(1.0) q[84];
cx q[83],q[84];
u(pi/2,0,pi) q[84];
rz(1.0) q[84];
u(pi/2,0,pi) q[84];
cx q[84],q[85];
rz(1.0) q[85];
cx q[84],q[85];
u(pi/2,0,pi) q[85];
rz(1.0) q[85];
u(pi/2,0,pi) q[85];
cx q[85],q[86];
rz(1.0) q[86];
cx q[85],q[86];
u(pi/2,0,pi) q[86];
rz(1.0) q[86];
u(pi/2,0,pi) q[86];
cx q[86],q[87];
rz(1.0) q[87];
cx q[86],q[87];
u(pi/2,0,pi) q[87];
rz(1.0) q[87];
u(pi/2,0,pi) q[87];
cx q[87],q[88];
rz(1.0) q[88];
cx q[87],q[88];
u(pi/2,0,pi) q[88];
rz(1.0) q[88];
u(pi/2,0,pi) q[88];
cx q[88],q[89];
rz(1.0) q[89];
cx q[88],q[89];
u(pi/2,0,pi) q[89];
rz(1.0) q[89];
u(pi/2,0,pi) q[89];
cx q[89],q[90];
rz(1.0) q[90];
cx q[89],q[90];
u(pi/2,0,pi) q[90];
rz(1.0) q[90];
u(pi/2,0,pi) q[90];
cx q[5],q[90];
rz(1.0) q[90];
cx q[5],q[90];
u(pi/2,0,pi) q[5];
rz(1.0) q[5];
u(pi/2,0,pi) q[5];
cx q[4],q[5];
rz(1.0) q[5];
cx q[4],q[5];
u(pi/2,0,pi) q[4];
rz(1.0) q[4];
u(pi/2,0,pi) q[4];
cx q[3],q[4];
rz(1.0) q[4];
cx q[3],q[4];
u(pi/2,0,pi) q[3];
rz(1.0) q[3];
u(pi/2,0,pi) q[3];
cx q[2],q[3];
rz(1.0) q[3];
cx q[2],q[3];
u(pi/2,0,pi) q[2];
rz(1.0) q[2];
u(pi/2,0,pi) q[2];
cx q[1],q[2];
rz(1.0) q[2];
cx q[1],q[2];
u(pi/2,0,pi) q[1];
rz(1.0) q[1];
u(pi/2,0,pi) q[1];
cx q[1],q[12];
rz(1.0) q[12];
cx q[1],q[12];
cx q[1],q[86];
rz(1.0) q[86];
cx q[1],q[86];
cx q[81],q[86];
rz(1.0) q[86];
cx q[81],q[86];
cx q[72],q[81];
rz(1.0) q[81];
cx q[72],q[81];
cx q[67],q[72];
rz(1.0) q[72];
cx q[67],q[72];
cx q[58],q[67];
rz(1.0) q[67];
cx q[58],q[67];
cx q[53],q[58];
rz(1.0) q[58];
cx q[53],q[58];
cx q[44],q[53];
rz(1.0) q[53];
cx q[44],q[53];
cx q[39],q[44];
rz(1.0) q[44];
cx q[39],q[44];
cx q[30],q[39];
rz(1.0) q[39];
cx q[30],q[39];
cx q[26],q[30];
rz(1.0) q[30];
cx q[26],q[30];
cx q[26],q[29];
rz(1.0) q[29];
cx q[26],q[29];
cx q[27],q[29];
rz(1.0) q[29];
cx q[27],q[29];
cx q[21],q[27];
rz(1.0) q[27];
cx q[21],q[27];
cx q[21],q[28];
rz(1.0) q[28];
cx q[21],q[28];
cx q[28],q[34];
rz(1.0) q[34];
cx q[28],q[34];
cx q[21],q[34];
rz(1.0) q[34];
cx q[21],q[34];
cx q[34],q[36];
rz(1.0) q[36];
cx q[34],q[36];
cx q[36],q[47];
rz(1.0) q[47];
cx q[36],q[47];
cx q[33],q[36];
rz(1.0) q[36];
cx q[33],q[36];
cx q[22],q[33];
rz(1.0) q[33];
cx q[22],q[33];
cx q[19],q[22];
rz(1.0) q[22];
cx q[19],q[22];
cx q[8],q[19];
rz(1.0) q[19];
cx q[8],q[19];
cx q[5],q[8];
rz(1.0) q[8];
cx q[5],q[8];
cx q[5],q[6];
rz(1.0) q[6];
cx q[5],q[6];
cx q[6],q[84];
rz(1.0) q[84];
cx q[6],q[84];
cx q[84],q[90];
rz(1.0) q[90];
cx q[84],q[90];
cx q[77],q[90];
rz(1.0) q[90];
cx q[77],q[90];
cx q[77],q[83];
rz(1.0) q[83];
cx q[77],q[83];
cx q[83],q[85];
rz(1.0) q[85];
cx q[83],q[85];
cx q[82],q[85];
rz(1.0) q[85];
cx q[82],q[85];
cx q[71],q[82];
rz(1.0) q[82];
cx q[71],q[82];
cx q[68],q[71];
rz(1.0) q[71];
cx q[68],q[71];
cx q[69],q[71];
rz(1.0) q[71];
cx q[69],q[71];
cx q[63],q[69];
rz(1.0) q[69];
cx q[63],q[69];
cx q[63],q[70];
rz(1.0) q[70];
cx q[63],q[70];
cx q[70],q[76];
rz(1.0) q[76];
cx q[70],q[76];
cx q[63],q[76];
rz(1.0) q[76];
cx q[63],q[76];
cx q[76],q[78];
rz(1.0) q[78];
cx q[76],q[78];
cx q[78],q[89];
rz(1.0) q[89];
cx q[78],q[89];
cx q[75],q[78];
rz(1.0) q[78];
cx q[75],q[78];
cx q[64],q[75];
rz(1.0) q[75];
cx q[64],q[75];
cx q[61],q[64];
rz(1.0) q[64];
cx q[61],q[64];
cx q[50],q[61];
rz(1.0) q[61];
cx q[50],q[61];
cx q[47],q[50];
rz(1.0) q[50];
cx q[47],q[50];
cx q[47],q[51];
rz(1.0) q[51];
cx q[47],q[51];
cx q[46],q[51];
rz(1.0) q[51];
cx q[46],q[51];
cx q[37],q[46];
rz(1.0) q[46];
cx q[37],q[46];
cx q[32],q[37];
rz(1.0) q[37];
cx q[32],q[37];
cx q[23],q[32];
rz(1.0) q[32];
cx q[23],q[32];
cx q[18],q[23];
rz(1.0) q[23];
cx q[18],q[23];
cx q[9],q[18];
rz(1.0) q[18];
cx q[9],q[18];
cx q[4],q[9];
rz(1.0) q[9];
cx q[4],q[9];
cx q[4],q[10];
rz(1.0) q[10];
cx q[4],q[10];
cx q[3],q[10];
rz(1.0) q[10];
cx q[3],q[10];
cx q[3],q[88];
rz(1.0) q[88];
cx q[3],q[88];
cx q[79],q[88];
rz(1.0) q[88];
cx q[79],q[88];
cx q[74],q[79];
rz(1.0) q[79];
cx q[74],q[79];
cx q[65],q[74];
rz(1.0) q[74];
cx q[65],q[74];
cx q[60],q[65];
rz(1.0) q[65];
cx q[60],q[65];
cx q[51],q[60];
rz(1.0) q[60];
cx q[51],q[60];
cx q[60],q[66];
rz(1.0) q[66];
cx q[60],q[66];
cx q[59],q[66];
rz(1.0) q[66];
cx q[59],q[66];
cx q[52],q[59];
rz(1.0) q[59];
cx q[52],q[59];
cx q[45],q[52];
rz(1.0) q[52];
cx q[45],q[52];
cx q[38],q[45];
rz(1.0) q[45];
cx q[38],q[45];
cx q[31],q[38];
rz(1.0) q[38];
cx q[31],q[38];
cx q[24],q[31];
rz(1.0) q[31];
cx q[24],q[31];
cx q[17],q[24];
rz(1.0) q[24];
cx q[17],q[24];
cx q[17],q[25];
rz(1.0) q[25];
cx q[17],q[25];
cx q[16],q[25];
rz(1.0) q[25];
cx q[16],q[25];
cx q[11],q[16];
rz(1.0) q[16];
cx q[11],q[16];
cx q[2],q[11];
rz(1.0) q[11];
cx q[2],q[11];
cx q[2],q[87];
rz(1.0) q[87];
cx q[2],q[87];
cx q[80],q[87];
rz(1.0) q[87];
cx q[80],q[87];
cx q[73],q[80];
rz(1.0) q[80];
cx q[73],q[80];
cx q[66],q[73];
rz(1.0) q[73];
cx q[66],q[73];
cx q[73],q[81];
rz(1.0) q[81];
cx q[73],q[81];
cx q[67],q[73];
rz(1.0) q[73];
cx q[67],q[73];
cx q[59],q[67];
rz(1.0) q[67];
cx q[59],q[67];
cx q[53],q[59];
rz(1.0) q[59];
cx q[53],q[59];
cx q[45],q[53];
rz(1.0) q[53];
cx q[45],q[53];
cx q[39],q[45];
rz(1.0) q[45];
cx q[39],q[45];
cx q[31],q[39];
rz(1.0) q[39];
cx q[31],q[39];
cx q[25],q[31];
rz(1.0) q[31];
cx q[25],q[31];
cx q[25],q[30];
rz(1.0) q[30];
cx q[25],q[30];
cx q[30],q[40];
rz(1.0) q[40];
cx q[30],q[40];
cx q[40],q[43];
rz(1.0) q[43];
cx q[40],q[43];
cx q[41],q[43];
rz(1.0) q[43];
cx q[41],q[43];
cx q[35],q[41];
rz(1.0) q[41];
cx q[35],q[41];
cx q[35],q[42];
rz(1.0) q[42];
cx q[35],q[42];
cx q[42],q[48];
rz(1.0) q[48];
cx q[42],q[48];
cx q[35],q[48];
rz(1.0) q[48];
cx q[35],q[48];
cx q[48],q[50];
rz(1.0) q[50];
cx q[48],q[50];
cx q[36],q[48];
rz(1.0) q[48];
cx q[36],q[48];
cx q[0],q[7];
rz(1.0) q[7];
cx q[0],q[7];
cx q[0],q[13];
rz(1.0) q[13];
cx q[0],q[13];
cx q[13],q[15];
rz(1.0) q[15];
cx q[13],q[15];
cx q[12],q[15];
rz(1.0) q[15];
cx q[12],q[15];
cx q[2],q[12];
rz(1.0) q[12];
cx q[2],q[12];
cx q[12],q[16];
rz(1.0) q[16];
cx q[12],q[16];
cx q[16],q[26];
rz(1.0) q[26];
cx q[16],q[26];
cx q[15],q[26];
rz(1.0) q[26];
cx q[15],q[26];
cx q[15],q[27];
rz(1.0) q[27];
cx q[15],q[27];
cx q[14],q[27];
rz(1.0) q[27];
cx q[14],q[27];
cx q[14],q[20];
rz(1.0) q[20];
cx q[14],q[20];
cx q[7],q[14];
rz(1.0) q[14];
cx q[7],q[14];
cx q[7],q[20];
rz(1.0) q[20];
cx q[7],q[20];
cx q[20],q[22];
rz(1.0) q[22];
cx q[20],q[22];
cx q[8],q[20];
rz(1.0) q[20];
cx q[8],q[20];
cx q[6],q[8];
rz(1.0) q[8];
cx q[6],q[8];
cx q[6],q[90];
rz(1.0) q[90];
cx q[6],q[90];
cx q[78],q[90];
rz(1.0) q[90];
cx q[78],q[90];
cx q[49],q[55];
rz(1.0) q[55];
cx q[49],q[55];
cx q[49],q[56];
rz(1.0) q[56];
cx q[49],q[56];
cx q[56],q[62];
rz(1.0) q[62];
cx q[56],q[62];
cx q[49],q[62];
rz(1.0) q[62];
cx q[49],q[62];
cx q[62],q[64];
rz(1.0) q[64];
cx q[62],q[64];
cx q[50],q[62];
rz(1.0) q[62];
cx q[50],q[62];
cx q[54],q[57];
rz(1.0) q[57];
cx q[54],q[57];
cx q[55],q[57];
rz(1.0) q[57];
cx q[55],q[57];
cx q[57],q[68];
rz(1.0) q[68];
cx q[57],q[68];
cx q[68],q[72];
rz(1.0) q[72];
cx q[68],q[72];
cx q[58],q[68];
rz(1.0) q[68];
cx q[58],q[68];
cx q[54],q[58];
rz(1.0) q[58];
cx q[54],q[58];
cx q[43],q[54];
rz(1.0) q[54];
cx q[43],q[54];
cx q[43],q[55];
rz(1.0) q[55];
cx q[43],q[55];
cx q[42],q[55];
rz(1.0) q[55];
cx q[42],q[55];
cx q[42],q[49];
rz(1.0) q[49];
cx q[42],q[49];
cx q[4],q[89];
rz(1.0) q[89];
cx q[4],q[89];
cx q[4],q[88];
rz(1.0) q[88];
cx q[4],q[88];
cx q[80],q[88];
rz(1.0) q[88];
cx q[80],q[88];
cx q[74],q[80];
rz(1.0) q[80];
cx q[74],q[80];
cx q[66],q[74];
rz(1.0) q[74];
cx q[66],q[74];
cx q[29],q[40];
rz(1.0) q[40];
cx q[29],q[40];
cx q[40],q[44];
rz(1.0) q[44];
cx q[40],q[44];
cx q[44],q[54];
rz(1.0) q[54];
cx q[44],q[54];
cx q[0],q[84];
rz(1.0) q[84];
cx q[0],q[84];
cx q[0],q[85];
rz(1.0) q[85];
cx q[0],q[85];
cx q[1],q[85];
rz(1.0) q[85];
cx q[1],q[85];
cx q[1],q[13];
rz(1.0) q[13];
cx q[1],q[13];
cx q[7],q[13];
rz(1.0) q[13];
cx q[7],q[13];
cx q[3],q[11];
rz(1.0) q[11];
cx q[3],q[11];
cx q[3],q[87];
rz(1.0) q[87];
cx q[3],q[87];
cx q[81],q[87];
rz(1.0) q[87];
cx q[81],q[87];
cx q[5],q[9];
rz(1.0) q[9];
cx q[5],q[9];
cx q[5],q[89];
rz(1.0) q[89];
cx q[5],q[89];
cx q[79],q[89];
rz(1.0) q[89];
cx q[79],q[89];
cx q[75],q[79];
rz(1.0) q[79];
cx q[75],q[79];
cx q[65],q[75];
rz(1.0) q[75];
cx q[65],q[75];
cx q[61],q[65];
rz(1.0) q[65];
cx q[61],q[65];
cx q[51],q[61];
rz(1.0) q[61];
cx q[51],q[61];
cx q[10],q[17];
rz(1.0) q[17];
cx q[10],q[17];
cx q[10],q[18];
rz(1.0) q[18];
cx q[10],q[18];
cx q[18],q[24];
rz(1.0) q[24];
cx q[18],q[24];
cx q[24],q[32];
rz(1.0) q[32];
cx q[24],q[32];
cx q[32],q[38];
rz(1.0) q[38];
cx q[32],q[38];
cx q[38],q[46];
rz(1.0) q[46];
cx q[38],q[46];
cx q[46],q[52];
rz(1.0) q[52];
cx q[46],q[52];
cx q[52],q[60];
rz(1.0) q[60];
cx q[52],q[60];
cx q[19],q[23];
rz(1.0) q[23];
cx q[19],q[23];
cx q[9],q[19];
rz(1.0) q[19];
cx q[9],q[19];
cx q[28],q[41];
rz(1.0) q[41];
cx q[28],q[41];
cx q[28],q[35];
rz(1.0) q[35];
cx q[28],q[35];
cx q[33],q[37];
rz(1.0) q[37];
cx q[33],q[37];
cx q[23],q[33];
rz(1.0) q[33];
cx q[23],q[33];
cx q[56],q[69];
rz(1.0) q[69];
cx q[56],q[69];
cx q[56],q[63];
rz(1.0) q[63];
cx q[56],q[63];
cx q[70],q[77];
rz(1.0) q[77];
cx q[70],q[77];
cx q[70],q[83];
rz(1.0) q[83];
cx q[70],q[83];
cx q[71],q[83];
rz(1.0) q[83];
cx q[71],q[83];
cx q[82],q[86];
rz(1.0) q[86];
cx q[82],q[86];
cx q[2],q[86];
rz(1.0) q[86];
cx q[2],q[86];
cx q[11],q[17];
rz(1.0) q[17];
cx q[11],q[17];
cx q[14],q[21];
rz(1.0) q[21];
cx q[14],q[21];
cx q[22],q[34];
rz(1.0) q[34];
cx q[22],q[34];
cx q[29],q[41];
rz(1.0) q[41];
cx q[29],q[41];
cx q[37],q[47];
rz(1.0) q[47];
cx q[37],q[47];
cx q[57],q[69];
rz(1.0) q[69];
cx q[57],q[69];
cx q[64],q[76];
rz(1.0) q[76];
cx q[64],q[76];
cx q[72],q[82];
rz(1.0) q[82];
cx q[72],q[82];
cx q[77],q[84];
rz(1.0) q[84];
cx q[77],q[84];
