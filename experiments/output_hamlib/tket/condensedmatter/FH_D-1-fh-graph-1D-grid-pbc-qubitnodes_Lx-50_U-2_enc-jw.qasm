OPENQASM 2.0;
include "qelib1.inc";

qreg q[100];
u3(0.5*pi,-0.5*pi,0.5*pi) q[0];
u3(0.5*pi,0.0*pi,0.5*pi) q[1];
u3(0.5*pi,0.0*pi,1.0*pi) q[2];
u3(0.5*pi,0.0*pi,0.5*pi) q[3];
u3(0.5*pi,0.0*pi,1.0*pi) q[4];
u3(0.5*pi,0.0*pi,0.5*pi) q[5];
u3(0.5*pi,0.0*pi,1.0*pi) q[6];
u3(0.5*pi,0.0*pi,0.5*pi) q[7];
u3(0.5*pi,0.0*pi,1.0*pi) q[8];
u3(0.5*pi,0.0*pi,0.5*pi) q[9];
u3(0.5*pi,0.0*pi,1.0*pi) q[10];
u3(0.5*pi,0.0*pi,0.5*pi) q[11];
u3(0.5*pi,0.0*pi,1.0*pi) q[12];
u3(0.5*pi,0.0*pi,0.5*pi) q[13];
u3(0.5*pi,0.0*pi,1.0*pi) q[14];
u3(0.5*pi,0.0*pi,0.5*pi) q[15];
u3(0.5*pi,0.0*pi,1.0*pi) q[16];
u3(0.5*pi,0.0*pi,0.5*pi) q[17];
u3(0.5*pi,0.0*pi,1.0*pi) q[18];
u3(0.5*pi,0.0*pi,0.5*pi) q[19];
u3(0.5*pi,0.0*pi,1.0*pi) q[20];
u3(0.5*pi,0.0*pi,0.5*pi) q[21];
u3(0.5*pi,0.0*pi,1.0*pi) q[22];
u3(0.5*pi,0.0*pi,0.5*pi) q[23];
u3(0.5*pi,0.0*pi,1.0*pi) q[24];
u3(0.5*pi,0.0*pi,0.5*pi) q[25];
u3(0.5*pi,0.0*pi,1.0*pi) q[26];
u3(0.5*pi,0.0*pi,0.5*pi) q[27];
u3(0.5*pi,0.0*pi,1.0*pi) q[28];
u3(0.5*pi,0.0*pi,0.5*pi) q[29];
u3(0.5*pi,0.0*pi,1.0*pi) q[30];
u3(0.5*pi,0.0*pi,0.5*pi) q[31];
u3(0.5*pi,0.0*pi,1.0*pi) q[32];
u3(0.5*pi,0.0*pi,0.5*pi) q[33];
u3(0.5*pi,0.0*pi,1.0*pi) q[34];
u3(0.5*pi,0.0*pi,0.5*pi) q[35];
u3(0.5*pi,0.0*pi,1.0*pi) q[36];
u3(0.5*pi,0.0*pi,0.5*pi) q[37];
u3(0.5*pi,0.0*pi,1.0*pi) q[38];
u3(0.5*pi,0.0*pi,0.5*pi) q[39];
u3(0.5*pi,0.0*pi,1.0*pi) q[40];
u3(0.5*pi,0.0*pi,0.5*pi) q[41];
u3(0.5*pi,0.0*pi,1.0*pi) q[42];
u3(0.5*pi,0.0*pi,0.5*pi) q[43];
u3(0.5*pi,0.0*pi,1.0*pi) q[44];
u3(0.5*pi,0.0*pi,0.5*pi) q[45];
u3(0.5*pi,0.0*pi,1.0*pi) q[46];
u3(0.5*pi,0.0*pi,0.5*pi) q[47];
u3(1.5*pi,-0.5*pi,4.0*pi) q[48];
u3(1.5*pi,-0.5*pi,0.5*pi) q[49];
u3(1.0*pi,-0.5*pi,3.5*pi) q[51];
u3(1.0*pi,-0.5*pi,4.0*pi) q[52];
u3(1.0*pi,-0.5*pi,4.0*pi) q[53];
u3(1.0*pi,-0.5*pi,4.0*pi) q[54];
u3(1.0*pi,-0.5*pi,4.0*pi) q[55];
u3(1.0*pi,-0.5*pi,4.0*pi) q[56];
u3(1.0*pi,-0.5*pi,4.0*pi) q[57];
u3(1.0*pi,-0.5*pi,4.0*pi) q[58];
u3(1.0*pi,-0.5*pi,4.0*pi) q[59];
u3(1.0*pi,-0.5*pi,4.0*pi) q[60];
u3(1.0*pi,-0.5*pi,4.0*pi) q[61];
u3(1.0*pi,-0.5*pi,4.0*pi) q[62];
u3(1.0*pi,-0.5*pi,4.0*pi) q[63];
u3(1.0*pi,-0.5*pi,4.0*pi) q[64];
u3(1.0*pi,-0.5*pi,4.0*pi) q[65];
u3(1.0*pi,-0.5*pi,4.0*pi) q[66];
u3(1.0*pi,-0.5*pi,4.0*pi) q[67];
u3(1.0*pi,-0.5*pi,4.0*pi) q[68];
u3(1.0*pi,-0.5*pi,4.0*pi) q[69];
u3(1.0*pi,-0.5*pi,4.0*pi) q[70];
u3(1.0*pi,-0.5*pi,4.0*pi) q[71];
u3(1.0*pi,-0.5*pi,4.0*pi) q[72];
u3(1.0*pi,-0.5*pi,4.0*pi) q[73];
u3(1.0*pi,-0.5*pi,4.0*pi) q[74];
u3(1.0*pi,-0.5*pi,4.0*pi) q[75];
u3(1.0*pi,-0.5*pi,4.0*pi) q[76];
u3(1.0*pi,-0.5*pi,4.0*pi) q[77];
u3(1.0*pi,-0.5*pi,4.0*pi) q[78];
u3(1.0*pi,-0.5*pi,4.0*pi) q[79];
u3(1.0*pi,-0.5*pi,4.0*pi) q[80];
u3(1.0*pi,-0.5*pi,4.0*pi) q[81];
u3(1.0*pi,-0.5*pi,4.0*pi) q[82];
u3(1.0*pi,-0.5*pi,4.0*pi) q[83];
u3(1.0*pi,-0.5*pi,4.0*pi) q[84];
u3(1.0*pi,-0.5*pi,4.0*pi) q[85];
u3(1.0*pi,-0.5*pi,4.0*pi) q[86];
u3(1.0*pi,-0.5*pi,4.0*pi) q[87];
u3(1.0*pi,-0.5*pi,4.0*pi) q[88];
u3(1.0*pi,-0.5*pi,4.0*pi) q[89];
u3(1.0*pi,-0.5*pi,4.0*pi) q[90];
u3(1.0*pi,-0.5*pi,4.0*pi) q[91];
u3(1.0*pi,-0.5*pi,4.0*pi) q[92];
u3(1.0*pi,-0.5*pi,4.0*pi) q[93];
u3(1.0*pi,-0.5*pi,4.0*pi) q[94];
u3(1.0*pi,-0.5*pi,4.0*pi) q[95];
u3(1.0*pi,-0.5*pi,4.0*pi) q[96];
u3(1.0*pi,-0.5*pi,4.0*pi) q[97];
u3(1.0*pi,-0.5*pi,1.0*pi) q[98];
u3(0.0*pi,-0.5*pi,1.0*pi) q[99];
cx q[1],q[0];
u3(0.0*pi,-0.5*pi,4.1816901138162095*pi) q[0];
cx q[1],q[0];
u3(0.5*pi,-0.5*pi,0.5*pi) q[1];
cx q[2],q[1];
u3(0.0*pi,-0.5*pi,1.1816901138162093*pi) q[1];
cx q[2],q[1];
u3(0.5*pi,0.0*pi,1.0*pi) q[1];
u3(0.5*pi,0.0*pi,0.5*pi) q[2];
cx q[3],q[2];
u3(3.5*pi,-0.5*pi,4.1816901138162095*pi) q[2];
cx q[3],q[2];
u3(0.5*pi,-0.5*pi,0.5*pi) q[3];
cx q[4],q[3];
u3(0.0*pi,-0.5*pi,1.1816901138162093*pi) q[3];
cx q[4],q[3];
u3(0.5*pi,0.0*pi,1.0*pi) q[3];
u3(0.5*pi,0.0*pi,0.5*pi) q[4];
cx q[5],q[4];
u3(3.5*pi,-0.5*pi,4.1816901138162095*pi) q[4];
cx q[5],q[4];
u3(0.5*pi,-0.5*pi,0.5*pi) q[5];
cx q[6],q[5];
u3(0.0*pi,-0.5*pi,1.1816901138162093*pi) q[5];
cx q[6],q[5];
u3(0.5*pi,0.0*pi,1.0*pi) q[5];
u3(0.5*pi,0.0*pi,0.5*pi) q[6];
cx q[7],q[6];
u3(3.5*pi,-0.5*pi,4.1816901138162095*pi) q[6];
cx q[7],q[6];
u3(0.5*pi,-0.5*pi,0.5*pi) q[7];
cx q[8],q[7];
u3(0.0*pi,-0.5*pi,1.1816901138162093*pi) q[7];
cx q[8],q[7];
u3(0.5*pi,0.0*pi,1.0*pi) q[7];
u3(0.5*pi,0.0*pi,0.5*pi) q[8];
cx q[9],q[8];
u3(3.5*pi,-0.5*pi,4.1816901138162095*pi) q[8];
cx q[9],q[8];
u3(0.5*pi,-0.5*pi,0.5*pi) q[9];
cx q[10],q[9];
u3(0.0*pi,-0.5*pi,1.1816901138162093*pi) q[9];
cx q[10],q[9];
u3(0.5*pi,0.0*pi,1.0*pi) q[9];
u3(0.5*pi,0.0*pi,0.5*pi) q[10];
cx q[11],q[10];
u3(3.5*pi,-0.5*pi,4.1816901138162095*pi) q[10];
cx q[11],q[10];
u3(0.5*pi,-0.5*pi,0.5*pi) q[11];
cx q[12],q[11];
u3(0.0*pi,-0.5*pi,1.1816901138162093*pi) q[11];
cx q[12],q[11];
u3(0.5*pi,0.0*pi,1.0*pi) q[11];
u3(0.5*pi,0.0*pi,0.5*pi) q[12];
cx q[13],q[12];
u3(3.5*pi,-0.5*pi,4.1816901138162095*pi) q[12];
cx q[13],q[12];
u3(0.5*pi,-0.5*pi,0.5*pi) q[13];
cx q[14],q[13];
u3(0.0*pi,-0.5*pi,1.1816901138162093*pi) q[13];
cx q[14],q[13];
u3(0.5*pi,0.0*pi,1.0*pi) q[13];
u3(0.5*pi,0.0*pi,0.5*pi) q[14];
cx q[15],q[14];
u3(3.5*pi,-0.5*pi,4.1816901138162095*pi) q[14];
cx q[15],q[14];
u3(0.5*pi,-0.5*pi,0.5*pi) q[15];
cx q[16],q[15];
u3(0.0*pi,-0.5*pi,1.1816901138162093*pi) q[15];
cx q[16],q[15];
u3(0.5*pi,0.0*pi,1.0*pi) q[15];
u3(0.5*pi,0.0*pi,0.5*pi) q[16];
cx q[17],q[16];
u3(3.5*pi,-0.5*pi,4.1816901138162095*pi) q[16];
cx q[17],q[16];
u3(0.5*pi,-0.5*pi,0.5*pi) q[17];
cx q[18],q[17];
u3(0.0*pi,-0.5*pi,1.1816901138162093*pi) q[17];
cx q[18],q[17];
u3(0.5*pi,0.0*pi,1.0*pi) q[17];
u3(0.5*pi,0.0*pi,0.5*pi) q[18];
cx q[19],q[18];
u3(3.5*pi,-0.5*pi,4.1816901138162095*pi) q[18];
cx q[19],q[18];
u3(0.5*pi,-0.5*pi,0.5*pi) q[19];
cx q[20],q[19];
u3(0.0*pi,-0.5*pi,1.1816901138162093*pi) q[19];
cx q[20],q[19];
u3(0.5*pi,0.0*pi,1.0*pi) q[19];
u3(0.5*pi,0.0*pi,0.5*pi) q[20];
cx q[21],q[20];
u3(3.5*pi,-0.5*pi,4.1816901138162095*pi) q[20];
cx q[21],q[20];
u3(0.5*pi,-0.5*pi,0.5*pi) q[21];
cx q[22],q[21];
u3(0.0*pi,-0.5*pi,1.1816901138162093*pi) q[21];
cx q[22],q[21];
u3(0.5*pi,0.0*pi,1.0*pi) q[21];
u3(0.5*pi,0.0*pi,0.5*pi) q[22];
cx q[23],q[22];
u3(3.5*pi,-0.5*pi,4.1816901138162095*pi) q[22];
cx q[23],q[22];
u3(0.5*pi,-0.5*pi,0.5*pi) q[23];
cx q[24],q[23];
u3(0.0*pi,-0.5*pi,1.1816901138162093*pi) q[23];
cx q[24],q[23];
u3(0.5*pi,0.0*pi,1.0*pi) q[23];
u3(0.5*pi,0.0*pi,0.5*pi) q[24];
cx q[25],q[24];
u3(3.5*pi,-0.5*pi,4.1816901138162095*pi) q[24];
cx q[25],q[24];
u3(0.5*pi,-0.5*pi,0.5*pi) q[25];
cx q[26],q[25];
u3(0.0*pi,-0.5*pi,1.1816901138162093*pi) q[25];
cx q[26],q[25];
u3(0.5*pi,0.0*pi,1.0*pi) q[25];
u3(0.5*pi,0.0*pi,0.5*pi) q[26];
cx q[27],q[26];
u3(3.5*pi,-0.5*pi,4.1816901138162095*pi) q[26];
cx q[27],q[26];
u3(0.5*pi,-0.5*pi,0.5*pi) q[27];
cx q[28],q[27];
u3(0.0*pi,-0.5*pi,1.1816901138162093*pi) q[27];
cx q[28],q[27];
u3(0.5*pi,0.0*pi,1.0*pi) q[27];
u3(0.5*pi,0.0*pi,0.5*pi) q[28];
cx q[29],q[28];
u3(3.5*pi,-0.5*pi,4.1816901138162095*pi) q[28];
cx q[29],q[28];
u3(0.5*pi,-0.5*pi,0.5*pi) q[29];
cx q[30],q[29];
u3(0.0*pi,-0.5*pi,1.1816901138162093*pi) q[29];
cx q[30],q[29];
u3(0.5*pi,0.0*pi,1.0*pi) q[29];
u3(0.5*pi,0.0*pi,0.5*pi) q[30];
cx q[31],q[30];
u3(3.5*pi,-0.5*pi,4.1816901138162095*pi) q[30];
cx q[31],q[30];
u3(0.5*pi,-0.5*pi,0.5*pi) q[31];
cx q[32],q[31];
u3(0.0*pi,-0.5*pi,1.1816901138162093*pi) q[31];
cx q[32],q[31];
u3(0.5*pi,0.0*pi,1.0*pi) q[31];
u3(0.5*pi,0.0*pi,0.5*pi) q[32];
cx q[33],q[32];
u3(3.5*pi,-0.5*pi,4.1816901138162095*pi) q[32];
cx q[33],q[32];
u3(0.5*pi,-0.5*pi,0.5*pi) q[33];
cx q[34],q[33];
u3(0.0*pi,-0.5*pi,1.1816901138162093*pi) q[33];
cx q[34],q[33];
u3(0.5*pi,0.0*pi,1.0*pi) q[33];
u3(0.5*pi,0.0*pi,0.5*pi) q[34];
cx q[35],q[34];
u3(3.5*pi,-0.5*pi,4.1816901138162095*pi) q[34];
cx q[35],q[34];
u3(0.5*pi,-0.5*pi,0.5*pi) q[35];
cx q[36],q[35];
u3(0.0*pi,-0.5*pi,1.1816901138162093*pi) q[35];
cx q[36],q[35];
u3(0.5*pi,0.0*pi,1.0*pi) q[35];
u3(0.5*pi,0.0*pi,0.5*pi) q[36];
cx q[37],q[36];
u3(3.5*pi,-0.5*pi,4.1816901138162095*pi) q[36];
cx q[37],q[36];
u3(0.5*pi,-0.5*pi,0.5*pi) q[37];
cx q[38],q[37];
u3(0.0*pi,-0.5*pi,1.1816901138162093*pi) q[37];
cx q[38],q[37];
u3(0.5*pi,0.0*pi,1.0*pi) q[37];
u3(0.5*pi,0.0*pi,0.5*pi) q[38];
cx q[39],q[38];
u3(3.5*pi,-0.5*pi,4.1816901138162095*pi) q[38];
cx q[39],q[38];
u3(0.5*pi,-0.5*pi,0.5*pi) q[39];
cx q[40],q[39];
u3(0.0*pi,-0.5*pi,1.1816901138162093*pi) q[39];
cx q[40],q[39];
u3(0.5*pi,0.0*pi,1.0*pi) q[39];
u3(0.5*pi,0.0*pi,0.5*pi) q[40];
cx q[41],q[40];
u3(3.5*pi,-0.5*pi,4.1816901138162095*pi) q[40];
cx q[41],q[40];
u3(0.5*pi,-0.5*pi,0.5*pi) q[41];
cx q[42],q[41];
u3(0.0*pi,-0.5*pi,1.1816901138162093*pi) q[41];
cx q[42],q[41];
u3(0.5*pi,0.0*pi,1.0*pi) q[41];
u3(0.5*pi,0.0*pi,0.5*pi) q[42];
cx q[43],q[42];
u3(3.5*pi,-0.5*pi,4.1816901138162095*pi) q[42];
cx q[43],q[42];
u3(0.5*pi,-0.5*pi,0.5*pi) q[43];
cx q[44],q[43];
u3(0.0*pi,-0.5*pi,1.1816901138162093*pi) q[43];
cx q[44],q[43];
u3(0.5*pi,0.0*pi,1.0*pi) q[43];
u3(0.5*pi,0.0*pi,0.5*pi) q[44];
cx q[45],q[44];
u3(3.5*pi,-0.5*pi,4.1816901138162095*pi) q[44];
cx q[45],q[44];
u3(0.5*pi,-0.5*pi,0.5*pi) q[45];
cx q[46],q[45];
u3(0.0*pi,-0.5*pi,1.1816901138162093*pi) q[45];
cx q[46],q[45];
u3(0.5*pi,0.0*pi,1.0*pi) q[45];
u3(0.5*pi,0.0*pi,0.5*pi) q[46];
cx q[47],q[46];
u3(3.5*pi,-0.5*pi,4.1816901138162095*pi) q[46];
cx q[47],q[46];
u3(0.5*pi,-0.5*pi,0.5*pi) q[47];
cx q[48],q[47];
u3(0.0*pi,-0.5*pi,1.1816901138162093*pi) q[47];
u3(0.5*pi,0.0*pi,0.5*pi) q[48];
cx q[49],q[48];
u3(0.0*pi,-0.5*pi,1.0*pi) q[48];
u3(0.5*pi,0.0*pi,0.5*pi) q[49];
cx q[48],q[46];
cx q[46],q[45];
cx q[45],q[44];
cx q[44],q[43];
cx q[43],q[42];
cx q[42],q[41];
cx q[41],q[40];
cx q[40],q[39];
cx q[39],q[38];
cx q[38],q[37];
cx q[37],q[36];
cx q[36],q[35];
cx q[35],q[34];
cx q[34],q[33];
cx q[33],q[32];
cx q[32],q[31];
cx q[31],q[30];
cx q[30],q[29];
cx q[29],q[28];
cx q[28],q[27];
cx q[27],q[26];
cx q[26],q[25];
cx q[25],q[24];
cx q[24],q[23];
cx q[23],q[22];
cx q[22],q[21];
cx q[21],q[20];
cx q[20],q[19];
cx q[19],q[18];
cx q[18],q[17];
cx q[17],q[16];
cx q[16],q[15];
cx q[15],q[14];
cx q[14],q[13];
cx q[13],q[12];
cx q[12],q[11];
cx q[11],q[10];
cx q[10],q[9];
cx q[9],q[8];
cx q[8],q[7];
cx q[7],q[6];
cx q[6],q[5];
cx q[5],q[4];
cx q[4],q[3];
cx q[3],q[2];
cx q[2],q[1];
cx q[1],q[0];
u3(2.5*pi,-0.5*pi,3.6816901138162095*pi) q[0];
u3(0.5*pi,1.3183098861837907*pi,4.0*pi) q[1];
cx q[0],q[1];
u3(0.5*pi,0.0*pi,1.0*pi) q[1];
cx q[2],q[1];
cx q[3],q[2];
cx q[4],q[3];
cx q[5],q[4];
cx q[6],q[5];
cx q[7],q[6];
cx q[8],q[7];
cx q[9],q[8];
cx q[10],q[9];
cx q[11],q[10];
cx q[12],q[11];
cx q[13],q[12];
cx q[14],q[13];
cx q[15],q[14];
cx q[16],q[15];
cx q[17],q[16];
cx q[18],q[17];
cx q[19],q[18];
cx q[20],q[19];
cx q[21],q[20];
cx q[22],q[21];
cx q[23],q[22];
cx q[24],q[23];
cx q[25],q[24];
cx q[26],q[25];
cx q[27],q[26];
cx q[28],q[27];
cx q[29],q[28];
cx q[30],q[29];
cx q[31],q[30];
cx q[32],q[31];
cx q[33],q[32];
cx q[34],q[33];
cx q[35],q[34];
cx q[36],q[35];
cx q[37],q[36];
cx q[38],q[37];
cx q[39],q[38];
cx q[40],q[39];
cx q[41],q[40];
cx q[42],q[41];
cx q[43],q[42];
cx q[44],q[43];
cx q[45],q[44];
cx q[46],q[45];
cx q[48],q[46];
u3(0.5*pi,-0.5*pi,0.5*pi) q[48];
cx q[48],q[47];
u3(0.5*pi,0.0*pi,1.0*pi) q[47];
u3(2.5*pi,-0.3183098861837905*pi,4.0*pi) q[48];
cx q[49],q[48];
cx q[48],q[47];
u3(0.5*pi,0.0*pi,0.5*pi) q[49];
cx q[47],q[46];
cx q[46],q[45];
cx q[45],q[44];
cx q[44],q[43];
cx q[43],q[42];
cx q[42],q[41];
cx q[41],q[40];
cx q[40],q[39];
cx q[39],q[38];
cx q[38],q[37];
cx q[37],q[36];
cx q[36],q[35];
cx q[35],q[34];
cx q[34],q[33];
cx q[33],q[32];
cx q[32],q[31];
cx q[31],q[30];
cx q[30],q[29];
cx q[29],q[28];
cx q[28],q[27];
cx q[27],q[26];
cx q[26],q[25];
cx q[25],q[24];
cx q[24],q[23];
cx q[23],q[22];
cx q[22],q[21];
cx q[21],q[20];
cx q[20],q[19];
cx q[19],q[18];
cx q[18],q[17];
cx q[17],q[16];
cx q[16],q[15];
cx q[15],q[14];
cx q[14],q[13];
cx q[13],q[12];
cx q[12],q[11];
cx q[11],q[10];
cx q[10],q[9];
cx q[9],q[8];
cx q[8],q[7];
cx q[7],q[6];
cx q[6],q[5];
cx q[5],q[4];
cx q[4],q[3];
cx q[3],q[2];
cx q[2],q[1];
cx q[1],q[0];
u3(1.0*pi,-0.5*pi,4.1816901138162095*pi) q[0];
cx q[1],q[0];
u3(3.5*pi,0.6816901138162093*pi,4.0*pi) q[0];
u3(0.5*pi,-0.5*pi,0.5*pi) q[1];
cx q[50],q[0];
cx q[2],q[1];
u3(0.0*pi,-0.5*pi,0.8183098861837907*pi) q[0];
u3(0.5*pi,-0.5*pi,0.5*pi) q[2];
cx q[50],q[0];
cx q[3],q[2];
u3(0.0*pi,-0.5*pi,1.0*pi) q[2];
cx q[4],q[3];
u3(0.5*pi,0.0*pi,0.5*pi) q[50];
cx q[2],q[1];
u3(0.5*pi,0.0*pi,1.0*pi) q[3];
u3(0.5*pi,-0.5*pi,0.5*pi) q[4];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[1];
cx q[5],q[4];
cx q[2],q[1];
u3(0.0*pi,-0.5*pi,1.0*pi) q[4];
cx q[6],q[5];
cx q[51],q[1];
u3(0.5*pi,-0.5*pi,0.5*pi) q[2];
u3(0.5*pi,0.0*pi,1.0*pi) q[5];
u3(0.5*pi,-0.5*pi,0.5*pi) q[6];
u3(1.0*pi,-0.5*pi,0.8183098861837907*pi) q[1];
cx q[3],q[2];
cx q[7],q[6];
cx q[51],q[1];
u3(0.0*pi,-0.5*pi,4.1816901138162095*pi) q[2];
u3(0.0*pi,-0.5*pi,1.0*pi) q[6];
cx q[8],q[7];
u3(0.0*pi,-0.5*pi,2.1816901138162095*pi) q[1];
cx q[3],q[2];
u3(0.5*pi,0.0*pi,1.0*pi) q[7];
u3(0.5*pi,-0.5*pi,0.5*pi) q[8];
u3(0.5*pi,-0.5*pi,0.5*pi) q[51];
u3(2.5*pi,0.3183098861837902*pi,4.0*pi) q[2];
u3(0.5*pi,0.0*pi,0.5*pi) q[3];
cx q[9],q[8];
cx q[50],q[51];
cx q[52],q[2];
cx q[4],q[3];
u3(0.0*pi,-0.5*pi,1.0*pi) q[8];
cx q[10],q[9];
u3(1.681690113816209*pi,0.0*pi,1.0*pi) q[50];
u3(0.5*pi,-0.5*pi,1.181690113816209*pi) q[51];
u3(1.0*pi,-0.5*pi,0.8183098861837907*pi) q[2];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[3];
u3(0.5*pi,0.0*pi,1.0*pi) q[9];
u3(0.5*pi,-0.5*pi,0.5*pi) q[10];
cx q[50],q[51];
cx q[52],q[2];
cx q[4],q[3];
cx q[11],q[10];
u3(0.5*pi,0.0*pi,0.5*pi) q[50];
u3(0.5*pi,-0.5*pi,1.0*pi) q[51];
cx q[53],q[3];
u3(0.5*pi,-0.5*pi,0.5*pi) q[4];
u3(0.0*pi,-0.5*pi,1.0*pi) q[10];
cx q[12],q[11];
u3(0.5*pi,-0.5*pi,0.5*pi) q[52];
u3(1.0*pi,-0.5*pi,0.8183098861837907*pi) q[3];
cx q[5],q[4];
u3(0.5*pi,0.0*pi,1.0*pi) q[11];
u3(0.5*pi,-0.5*pi,0.5*pi) q[12];
cx q[52],q[51];
cx q[53],q[3];
u3(0.0*pi,-0.5*pi,4.1816901138162095*pi) q[4];
cx q[13],q[12];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[51];
u3(1.68169011381621*pi,-0.5*pi,0.5*pi) q[52];
u3(0.0*pi,-0.5*pi,2.1816901138162095*pi) q[3];
cx q[5],q[4];
u3(0.0*pi,-0.5*pi,1.0*pi) q[12];
cx q[14],q[13];
cx q[52],q[51];
u3(0.5*pi,-0.5*pi,0.5*pi) q[53];
u3(2.5*pi,0.3183098861837902*pi,4.0*pi) q[4];
u3(0.5*pi,0.0*pi,0.5*pi) q[5];
u3(0.5*pi,0.0*pi,1.0*pi) q[13];
u3(0.5*pi,-0.5*pi,0.5*pi) q[14];
u3(0.0*pi,-0.5*pi,1.6816901138162093*pi) q[51];
cx q[53],q[52];
cx q[54],q[4];
cx q[6],q[5];
cx q[15],q[14];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[52];
u3(1.68169011381621*pi,-0.5*pi,0.5*pi) q[53];
u3(1.0*pi,-0.5*pi,0.8183098861837907*pi) q[4];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[5];
u3(0.0*pi,-0.5*pi,1.0*pi) q[14];
cx q[16],q[15];
cx q[53],q[52];
cx q[54],q[4];
cx q[6],q[5];
u3(0.5*pi,0.0*pi,1.0*pi) q[15];
u3(0.5*pi,-0.5*pi,0.5*pi) q[16];
u3(0.0*pi,-0.5*pi,1.6816901138162093*pi) q[52];
cx q[55],q[5];
u3(0.5*pi,-0.5*pi,0.5*pi) q[6];
cx q[17],q[16];
u3(0.5*pi,-0.5*pi,0.5*pi) q[54];
u3(1.0*pi,-0.5*pi,0.8183098861837907*pi) q[5];
cx q[7],q[6];
u3(0.0*pi,-0.5*pi,1.0*pi) q[16];
cx q[18],q[17];
cx q[54],q[53];
cx q[55],q[5];
u3(0.0*pi,-0.5*pi,4.1816901138162095*pi) q[6];
u3(0.5*pi,0.0*pi,1.0*pi) q[17];
u3(0.5*pi,-0.5*pi,0.5*pi) q[18];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[53];
u3(1.68169011381621*pi,-0.5*pi,0.5*pi) q[54];
u3(0.0*pi,-0.5*pi,2.1816901138162095*pi) q[5];
cx q[7],q[6];
cx q[19],q[18];
cx q[54],q[53];
u3(0.5*pi,-0.5*pi,0.5*pi) q[55];
u3(2.5*pi,0.3183098861837902*pi,4.0*pi) q[6];
u3(0.5*pi,0.0*pi,0.5*pi) q[7];
u3(0.0*pi,-0.5*pi,1.0*pi) q[18];
cx q[20],q[19];
u3(0.0*pi,-0.5*pi,1.6816901138162093*pi) q[53];
cx q[55],q[54];
cx q[56],q[6];
cx q[8],q[7];
u3(0.5*pi,0.0*pi,1.0*pi) q[19];
u3(0.5*pi,-0.5*pi,0.5*pi) q[20];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[54];
u3(1.68169011381621*pi,-0.5*pi,0.5*pi) q[55];
u3(1.0*pi,-0.5*pi,0.8183098861837907*pi) q[6];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[7];
cx q[21],q[20];
cx q[55],q[54];
cx q[56],q[6];
cx q[8],q[7];
u3(0.0*pi,-0.5*pi,1.0*pi) q[20];
cx q[22],q[21];
u3(0.0*pi,-0.5*pi,1.6816901138162093*pi) q[54];
cx q[57],q[7];
u3(0.5*pi,-0.5*pi,0.5*pi) q[8];
u3(0.5*pi,0.0*pi,1.0*pi) q[21];
u3(0.5*pi,-0.5*pi,0.5*pi) q[22];
u3(0.5*pi,-0.5*pi,0.5*pi) q[56];
u3(1.0*pi,-0.5*pi,0.8183098861837907*pi) q[7];
cx q[9],q[8];
cx q[23],q[22];
cx q[56],q[55];
cx q[57],q[7];
u3(0.0*pi,-0.5*pi,4.1816901138162095*pi) q[8];
u3(0.0*pi,-0.5*pi,1.0*pi) q[22];
cx q[24],q[23];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[55];
u3(1.68169011381621*pi,-0.5*pi,0.5*pi) q[56];
u3(0.0*pi,-0.5*pi,2.1816901138162095*pi) q[7];
cx q[9],q[8];
u3(0.5*pi,0.0*pi,1.0*pi) q[23];
u3(0.5*pi,-0.5*pi,0.5*pi) q[24];
cx q[56],q[55];
u3(0.5*pi,-0.5*pi,0.5*pi) q[57];
u3(2.5*pi,0.3183098861837902*pi,4.0*pi) q[8];
u3(0.5*pi,0.0*pi,0.5*pi) q[9];
cx q[25],q[24];
u3(0.0*pi,-0.5*pi,1.6816901138162093*pi) q[55];
cx q[57],q[56];
cx q[58],q[8];
cx q[10],q[9];
u3(0.0*pi,-0.5*pi,1.0*pi) q[24];
cx q[26],q[25];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[56];
u3(1.68169011381621*pi,-0.5*pi,0.5*pi) q[57];
u3(1.0*pi,-0.5*pi,0.8183098861837907*pi) q[8];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[9];
u3(0.5*pi,0.0*pi,1.0*pi) q[25];
u3(0.5*pi,-0.5*pi,0.5*pi) q[26];
cx q[57],q[56];
cx q[58],q[8];
cx q[10],q[9];
cx q[27],q[26];
u3(0.0*pi,-0.5*pi,1.6816901138162093*pi) q[56];
cx q[59],q[9];
u3(0.5*pi,-0.5*pi,0.5*pi) q[10];
u3(0.0*pi,-0.5*pi,1.0*pi) q[26];
cx q[28],q[27];
u3(0.5*pi,-0.5*pi,0.5*pi) q[58];
u3(1.0*pi,-0.5*pi,0.8183098861837907*pi) q[9];
cx q[11],q[10];
u3(0.5*pi,0.0*pi,1.0*pi) q[27];
u3(0.5*pi,-0.5*pi,0.5*pi) q[28];
cx q[58],q[57];
cx q[59],q[9];
u3(0.0*pi,-0.5*pi,4.1816901138162095*pi) q[10];
cx q[29],q[28];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[57];
u3(1.68169011381621*pi,-0.5*pi,0.5*pi) q[58];
u3(0.0*pi,-0.5*pi,2.1816901138162095*pi) q[9];
cx q[11],q[10];
u3(0.0*pi,-0.5*pi,1.0*pi) q[28];
cx q[30],q[29];
cx q[58],q[57];
u3(0.5*pi,-0.5*pi,0.5*pi) q[59];
u3(2.5*pi,0.3183098861837902*pi,4.0*pi) q[10];
u3(0.5*pi,0.0*pi,0.5*pi) q[11];
u3(0.5*pi,0.0*pi,1.0*pi) q[29];
u3(0.5*pi,-0.5*pi,0.5*pi) q[30];
u3(0.0*pi,-0.5*pi,1.6816901138162093*pi) q[57];
cx q[59],q[58];
cx q[60],q[10];
cx q[12],q[11];
cx q[31],q[30];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[58];
u3(1.68169011381621*pi,-0.5*pi,0.5*pi) q[59];
u3(1.0*pi,-0.5*pi,0.8183098861837907*pi) q[10];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[11];
u3(0.0*pi,-0.5*pi,1.0*pi) q[30];
cx q[32],q[31];
cx q[59],q[58];
cx q[60],q[10];
cx q[12],q[11];
u3(0.5*pi,0.0*pi,1.0*pi) q[31];
u3(0.5*pi,-0.5*pi,0.5*pi) q[32];
u3(0.0*pi,-0.5*pi,1.6816901138162093*pi) q[58];
cx q[61],q[11];
u3(0.5*pi,-0.5*pi,0.5*pi) q[12];
cx q[33],q[32];
u3(0.5*pi,-0.5*pi,0.5*pi) q[60];
u3(1.0*pi,-0.5*pi,0.8183098861837907*pi) q[11];
cx q[13],q[12];
u3(0.0*pi,-0.5*pi,1.0*pi) q[32];
cx q[34],q[33];
cx q[60],q[59];
cx q[61],q[11];
u3(0.0*pi,-0.5*pi,4.1816901138162095*pi) q[12];
u3(0.5*pi,0.0*pi,1.0*pi) q[33];
u3(0.5*pi,-0.5*pi,0.5*pi) q[34];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[59];
u3(1.68169011381621*pi,-0.5*pi,0.5*pi) q[60];
u3(0.0*pi,-0.5*pi,2.1816901138162095*pi) q[11];
cx q[13],q[12];
cx q[35],q[34];
cx q[60],q[59];
u3(0.5*pi,-0.5*pi,0.5*pi) q[61];
u3(2.5*pi,0.3183098861837902*pi,4.0*pi) q[12];
u3(0.5*pi,0.0*pi,0.5*pi) q[13];
u3(0.0*pi,-0.5*pi,1.0*pi) q[34];
cx q[36],q[35];
u3(0.0*pi,-0.5*pi,1.6816901138162093*pi) q[59];
cx q[61],q[60];
cx q[62],q[12];
cx q[14],q[13];
u3(0.5*pi,0.0*pi,1.0*pi) q[35];
u3(0.5*pi,-0.5*pi,0.5*pi) q[36];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[60];
u3(1.68169011381621*pi,-0.5*pi,0.5*pi) q[61];
u3(1.0*pi,-0.5*pi,0.8183098861837907*pi) q[12];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[13];
cx q[37],q[36];
cx q[61],q[60];
cx q[62],q[12];
cx q[14],q[13];
u3(0.0*pi,-0.5*pi,1.0*pi) q[36];
cx q[38],q[37];
u3(0.0*pi,-0.5*pi,1.6816901138162093*pi) q[60];
cx q[63],q[13];
u3(0.5*pi,-0.5*pi,0.5*pi) q[14];
u3(0.5*pi,0.0*pi,1.0*pi) q[37];
u3(0.5*pi,-0.5*pi,0.5*pi) q[38];
u3(0.5*pi,-0.5*pi,0.5*pi) q[62];
u3(1.0*pi,-0.5*pi,0.8183098861837907*pi) q[13];
cx q[15],q[14];
cx q[39],q[38];
cx q[62],q[61];
cx q[63],q[13];
u3(0.0*pi,-0.5*pi,4.1816901138162095*pi) q[14];
u3(0.0*pi,-0.5*pi,1.0*pi) q[38];
cx q[40],q[39];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[61];
u3(1.68169011381621*pi,-0.5*pi,0.5*pi) q[62];
u3(0.0*pi,-0.5*pi,2.1816901138162095*pi) q[13];
cx q[15],q[14];
u3(0.5*pi,0.0*pi,1.0*pi) q[39];
u3(0.5*pi,-0.5*pi,0.5*pi) q[40];
cx q[62],q[61];
u3(0.5*pi,-0.5*pi,0.5*pi) q[63];
u3(2.5*pi,0.3183098861837902*pi,4.0*pi) q[14];
u3(0.5*pi,0.0*pi,0.5*pi) q[15];
cx q[41],q[40];
u3(0.0*pi,-0.5*pi,1.6816901138162093*pi) q[61];
cx q[63],q[62];
cx q[64],q[14];
cx q[16],q[15];
u3(0.0*pi,-0.5*pi,1.0*pi) q[40];
cx q[42],q[41];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[62];
u3(1.68169011381621*pi,-0.5*pi,0.5*pi) q[63];
u3(1.0*pi,-0.5*pi,0.8183098861837907*pi) q[14];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[15];
u3(0.5*pi,0.0*pi,1.0*pi) q[41];
u3(0.5*pi,-0.5*pi,0.5*pi) q[42];
cx q[63],q[62];
cx q[64],q[14];
cx q[16],q[15];
cx q[43],q[42];
u3(0.0*pi,-0.5*pi,1.6816901138162093*pi) q[62];
cx q[65],q[15];
u3(0.5*pi,-0.5*pi,0.5*pi) q[16];
u3(0.0*pi,-0.5*pi,1.0*pi) q[42];
cx q[44],q[43];
u3(0.5*pi,-0.5*pi,0.5*pi) q[64];
u3(1.0*pi,-0.5*pi,0.8183098861837907*pi) q[15];
cx q[17],q[16];
u3(0.5*pi,0.0*pi,1.0*pi) q[43];
u3(0.5*pi,-0.5*pi,0.5*pi) q[44];
cx q[64],q[63];
cx q[65],q[15];
u3(0.0*pi,-0.5*pi,4.1816901138162095*pi) q[16];
cx q[45],q[44];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[63];
u3(1.68169011381621*pi,-0.5*pi,0.5*pi) q[64];
u3(0.0*pi,-0.5*pi,2.1816901138162095*pi) q[15];
cx q[17],q[16];
u3(0.0*pi,-0.5*pi,1.0*pi) q[44];
cx q[46],q[45];
cx q[64],q[63];
u3(0.5*pi,-0.5*pi,0.5*pi) q[65];
u3(2.5*pi,0.3183098861837902*pi,4.0*pi) q[16];
u3(0.5*pi,0.0*pi,0.5*pi) q[17];
u3(0.5*pi,0.0*pi,1.0*pi) q[45];
u3(0.5*pi,-0.5*pi,0.5*pi) q[46];
u3(0.0*pi,-0.5*pi,1.6816901138162093*pi) q[63];
cx q[65],q[64];
cx q[66],q[16];
cx q[18],q[17];
cx q[47],q[46];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[64];
u3(1.68169011381621*pi,-0.5*pi,0.5*pi) q[65];
u3(1.0*pi,-0.5*pi,0.8183098861837907*pi) q[16];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[17];
u3(0.0*pi,-0.5*pi,1.0*pi) q[46];
cx q[48],q[47];
cx q[65],q[64];
cx q[66],q[16];
cx q[18],q[17];
u3(0.5*pi,0.0*pi,1.0*pi) q[47];
u3(0.5*pi,-0.5*pi,0.5*pi) q[48];
u3(0.0*pi,-0.5*pi,1.6816901138162093*pi) q[64];
cx q[67],q[17];
u3(0.5*pi,-0.5*pi,0.5*pi) q[18];
cx q[49],q[48];
u3(0.5*pi,-0.5*pi,0.5*pi) q[66];
u3(1.0*pi,-0.5*pi,0.8183098861837907*pi) q[17];
cx q[19],q[18];
u3(0.0*pi,-0.5*pi,1.0*pi) q[48];
cx q[66],q[65];
cx q[67],q[17];
u3(0.0*pi,-0.5*pi,4.1816901138162095*pi) q[18];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[65];
u3(1.68169011381621*pi,-0.5*pi,0.5*pi) q[66];
u3(0.0*pi,-0.5*pi,2.1816901138162095*pi) q[17];
cx q[19],q[18];
cx q[66],q[65];
u3(0.5*pi,-0.5*pi,0.5*pi) q[67];
u3(2.5*pi,0.3183098861837902*pi,4.0*pi) q[18];
u3(0.5*pi,0.0*pi,0.5*pi) q[19];
u3(0.0*pi,-0.5*pi,1.6816901138162093*pi) q[65];
cx q[67],q[66];
cx q[68],q[18];
cx q[20],q[19];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[66];
u3(1.68169011381621*pi,-0.5*pi,0.5*pi) q[67];
u3(1.0*pi,-0.5*pi,0.8183098861837907*pi) q[18];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[19];
cx q[67],q[66];
cx q[68],q[18];
cx q[20],q[19];
u3(0.0*pi,-0.5*pi,1.6816901138162093*pi) q[66];
cx q[69],q[19];
u3(0.5*pi,-0.5*pi,0.5*pi) q[20];
u3(0.5*pi,-0.5*pi,0.5*pi) q[68];
u3(1.0*pi,-0.5*pi,0.8183098861837907*pi) q[19];
cx q[21],q[20];
cx q[68],q[67];
cx q[69],q[19];
u3(0.0*pi,-0.5*pi,4.1816901138162095*pi) q[20];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[67];
u3(1.68169011381621*pi,-0.5*pi,0.5*pi) q[68];
u3(0.0*pi,-0.5*pi,2.1816901138162095*pi) q[19];
cx q[21],q[20];
cx q[68],q[67];
u3(0.5*pi,-0.5*pi,0.5*pi) q[69];
u3(2.5*pi,0.3183098861837902*pi,4.0*pi) q[20];
u3(0.5*pi,0.0*pi,0.5*pi) q[21];
u3(0.0*pi,-0.5*pi,1.6816901138162093*pi) q[67];
cx q[69],q[68];
cx q[70],q[20];
cx q[22],q[21];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[68];
u3(1.68169011381621*pi,-0.5*pi,0.5*pi) q[69];
u3(1.0*pi,-0.5*pi,0.8183098861837907*pi) q[20];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[21];
cx q[69],q[68];
cx q[70],q[20];
cx q[22],q[21];
u3(0.0*pi,-0.5*pi,1.6816901138162093*pi) q[68];
cx q[71],q[21];
u3(0.5*pi,-0.5*pi,0.5*pi) q[22];
u3(0.5*pi,-0.5*pi,0.5*pi) q[70];
u3(1.0*pi,-0.5*pi,0.8183098861837907*pi) q[21];
cx q[23],q[22];
cx q[70],q[69];
cx q[71],q[21];
u3(0.0*pi,-0.5*pi,4.1816901138162095*pi) q[22];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[69];
u3(1.68169011381621*pi,-0.5*pi,0.5*pi) q[70];
u3(0.0*pi,-0.5*pi,2.1816901138162095*pi) q[21];
cx q[23],q[22];
cx q[70],q[69];
u3(0.5*pi,-0.5*pi,0.5*pi) q[71];
u3(2.5*pi,0.3183098861837902*pi,4.0*pi) q[22];
u3(0.5*pi,0.0*pi,0.5*pi) q[23];
u3(0.0*pi,-0.5*pi,1.6816901138162093*pi) q[69];
cx q[71],q[70];
cx q[72],q[22];
cx q[24],q[23];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[70];
u3(1.68169011381621*pi,-0.5*pi,0.5*pi) q[71];
u3(1.0*pi,-0.5*pi,0.8183098861837907*pi) q[22];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[23];
cx q[71],q[70];
cx q[72],q[22];
cx q[24],q[23];
u3(0.0*pi,-0.5*pi,1.6816901138162093*pi) q[70];
cx q[73],q[23];
u3(0.5*pi,-0.5*pi,0.5*pi) q[24];
u3(0.5*pi,-0.5*pi,0.5*pi) q[72];
u3(1.0*pi,-0.5*pi,0.8183098861837907*pi) q[23];
cx q[25],q[24];
cx q[72],q[71];
cx q[73],q[23];
u3(0.0*pi,-0.5*pi,4.1816901138162095*pi) q[24];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[71];
u3(1.68169011381621*pi,-0.5*pi,0.5*pi) q[72];
u3(0.0*pi,-0.5*pi,2.1816901138162095*pi) q[23];
cx q[25],q[24];
cx q[72],q[71];
u3(0.5*pi,-0.5*pi,0.5*pi) q[73];
u3(2.5*pi,0.3183098861837902*pi,4.0*pi) q[24];
u3(0.5*pi,0.0*pi,0.5*pi) q[25];
u3(0.0*pi,-0.5*pi,1.6816901138162093*pi) q[71];
cx q[73],q[72];
cx q[74],q[24];
cx q[26],q[25];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[72];
u3(1.68169011381621*pi,-0.5*pi,0.5*pi) q[73];
u3(1.0*pi,-0.5*pi,0.8183098861837907*pi) q[24];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[25];
cx q[73],q[72];
cx q[74],q[24];
cx q[26],q[25];
u3(0.0*pi,-0.5*pi,1.6816901138162093*pi) q[72];
cx q[75],q[25];
u3(0.5*pi,-0.5*pi,0.5*pi) q[26];
u3(0.5*pi,-0.5*pi,0.5*pi) q[74];
u3(1.0*pi,-0.5*pi,0.8183098861837907*pi) q[25];
cx q[27],q[26];
cx q[74],q[73];
cx q[75],q[25];
u3(0.0*pi,-0.5*pi,4.1816901138162095*pi) q[26];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[73];
u3(1.68169011381621*pi,-0.5*pi,0.5*pi) q[74];
u3(0.0*pi,-0.5*pi,2.1816901138162095*pi) q[25];
cx q[27],q[26];
cx q[74],q[73];
u3(0.5*pi,-0.5*pi,0.5*pi) q[75];
u3(2.5*pi,0.3183098861837902*pi,4.0*pi) q[26];
u3(0.5*pi,0.0*pi,0.5*pi) q[27];
u3(0.0*pi,-0.5*pi,1.6816901138162093*pi) q[73];
cx q[75],q[74];
cx q[76],q[26];
cx q[28],q[27];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[74];
u3(1.68169011381621*pi,-0.5*pi,0.5*pi) q[75];
u3(1.0*pi,-0.5*pi,0.8183098861837907*pi) q[26];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[27];
cx q[75],q[74];
cx q[76],q[26];
cx q[28],q[27];
u3(0.0*pi,-0.5*pi,1.6816901138162093*pi) q[74];
cx q[77],q[27];
u3(0.5*pi,-0.5*pi,0.5*pi) q[28];
u3(0.5*pi,-0.5*pi,0.5*pi) q[76];
u3(1.0*pi,-0.5*pi,0.8183098861837907*pi) q[27];
cx q[29],q[28];
cx q[76],q[75];
cx q[77],q[27];
u3(0.0*pi,-0.5*pi,4.1816901138162095*pi) q[28];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[75];
u3(1.68169011381621*pi,-0.5*pi,0.5*pi) q[76];
u3(0.0*pi,-0.5*pi,2.1816901138162095*pi) q[27];
cx q[29],q[28];
cx q[76],q[75];
u3(0.5*pi,-0.5*pi,0.5*pi) q[77];
u3(2.5*pi,0.3183098861837902*pi,4.0*pi) q[28];
u3(0.5*pi,0.0*pi,0.5*pi) q[29];
u3(0.0*pi,-0.5*pi,1.6816901138162093*pi) q[75];
cx q[77],q[76];
cx q[78],q[28];
cx q[30],q[29];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[76];
u3(1.68169011381621*pi,-0.5*pi,0.5*pi) q[77];
u3(1.0*pi,-0.5*pi,0.8183098861837907*pi) q[28];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[29];
cx q[77],q[76];
cx q[78],q[28];
cx q[30],q[29];
u3(0.0*pi,-0.5*pi,1.6816901138162093*pi) q[76];
cx q[79],q[29];
u3(0.5*pi,-0.5*pi,0.5*pi) q[30];
u3(0.5*pi,-0.5*pi,0.5*pi) q[78];
u3(1.0*pi,-0.5*pi,0.8183098861837907*pi) q[29];
cx q[31],q[30];
cx q[78],q[77];
cx q[79],q[29];
u3(0.0*pi,-0.5*pi,4.1816901138162095*pi) q[30];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[77];
u3(1.68169011381621*pi,-0.5*pi,0.5*pi) q[78];
u3(0.0*pi,-0.5*pi,2.1816901138162095*pi) q[29];
cx q[31],q[30];
cx q[78],q[77];
u3(0.5*pi,-0.5*pi,0.5*pi) q[79];
u3(2.5*pi,0.3183098861837902*pi,4.0*pi) q[30];
u3(0.5*pi,0.0*pi,0.5*pi) q[31];
u3(0.0*pi,-0.5*pi,1.6816901138162093*pi) q[77];
cx q[79],q[78];
cx q[80],q[30];
cx q[32],q[31];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[78];
u3(1.68169011381621*pi,-0.5*pi,0.5*pi) q[79];
u3(1.0*pi,-0.5*pi,0.8183098861837907*pi) q[30];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[31];
cx q[79],q[78];
cx q[80],q[30];
cx q[32],q[31];
u3(0.0*pi,-0.5*pi,1.6816901138162093*pi) q[78];
cx q[81],q[31];
u3(0.5*pi,-0.5*pi,0.5*pi) q[32];
u3(0.5*pi,-0.5*pi,0.5*pi) q[80];
u3(1.0*pi,-0.5*pi,0.8183098861837907*pi) q[31];
cx q[33],q[32];
cx q[80],q[79];
cx q[81],q[31];
u3(0.0*pi,-0.5*pi,4.1816901138162095*pi) q[32];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[79];
u3(1.68169011381621*pi,-0.5*pi,0.5*pi) q[80];
u3(0.0*pi,-0.5*pi,2.1816901138162095*pi) q[31];
cx q[33],q[32];
cx q[80],q[79];
u3(0.5*pi,-0.5*pi,0.5*pi) q[81];
u3(2.5*pi,0.3183098861837902*pi,4.0*pi) q[32];
u3(0.5*pi,0.0*pi,0.5*pi) q[33];
u3(0.0*pi,-0.5*pi,1.6816901138162093*pi) q[79];
cx q[81],q[80];
cx q[82],q[32];
cx q[34],q[33];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[80];
u3(1.68169011381621*pi,-0.5*pi,0.5*pi) q[81];
u3(1.0*pi,-0.5*pi,0.8183098861837907*pi) q[32];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[33];
cx q[81],q[80];
cx q[82],q[32];
cx q[34],q[33];
u3(0.0*pi,-0.5*pi,1.6816901138162093*pi) q[80];
cx q[83],q[33];
u3(0.5*pi,-0.5*pi,0.5*pi) q[34];
u3(0.5*pi,-0.5*pi,0.5*pi) q[82];
u3(1.0*pi,-0.5*pi,0.8183098861837907*pi) q[33];
cx q[35],q[34];
cx q[82],q[81];
cx q[83],q[33];
u3(0.0*pi,-0.5*pi,4.1816901138162095*pi) q[34];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[81];
u3(1.68169011381621*pi,-0.5*pi,0.5*pi) q[82];
u3(0.0*pi,-0.5*pi,2.1816901138162095*pi) q[33];
cx q[35],q[34];
cx q[82],q[81];
u3(0.5*pi,-0.5*pi,0.5*pi) q[83];
u3(2.5*pi,0.3183098861837902*pi,4.0*pi) q[34];
u3(0.5*pi,0.0*pi,0.5*pi) q[35];
u3(0.0*pi,-0.5*pi,1.6816901138162093*pi) q[81];
cx q[83],q[82];
cx q[84],q[34];
cx q[36],q[35];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[82];
u3(1.68169011381621*pi,-0.5*pi,0.5*pi) q[83];
u3(1.0*pi,-0.5*pi,0.8183098861837907*pi) q[34];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[35];
cx q[83],q[82];
cx q[84],q[34];
cx q[36],q[35];
u3(0.0*pi,-0.5*pi,1.6816901138162093*pi) q[82];
cx q[85],q[35];
u3(0.5*pi,-0.5*pi,0.5*pi) q[36];
u3(0.5*pi,-0.5*pi,0.5*pi) q[84];
u3(1.0*pi,-0.5*pi,0.8183098861837907*pi) q[35];
cx q[37],q[36];
cx q[84],q[83];
cx q[85],q[35];
u3(0.0*pi,-0.5*pi,4.1816901138162095*pi) q[36];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[83];
u3(1.68169011381621*pi,-0.5*pi,0.5*pi) q[84];
u3(0.0*pi,-0.5*pi,2.1816901138162095*pi) q[35];
cx q[37],q[36];
cx q[84],q[83];
u3(0.5*pi,-0.5*pi,0.5*pi) q[85];
u3(2.5*pi,0.3183098861837902*pi,4.0*pi) q[36];
u3(0.5*pi,0.0*pi,0.5*pi) q[37];
u3(0.0*pi,-0.5*pi,1.6816901138162093*pi) q[83];
cx q[85],q[84];
cx q[86],q[36];
cx q[38],q[37];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[84];
u3(1.68169011381621*pi,-0.5*pi,0.5*pi) q[85];
u3(1.0*pi,-0.5*pi,0.8183098861837907*pi) q[36];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[37];
cx q[85],q[84];
cx q[86],q[36];
cx q[38],q[37];
u3(0.0*pi,-0.5*pi,1.6816901138162093*pi) q[84];
cx q[87],q[37];
u3(0.5*pi,-0.5*pi,0.5*pi) q[38];
u3(0.5*pi,-0.5*pi,0.5*pi) q[86];
u3(1.0*pi,-0.5*pi,0.8183098861837907*pi) q[37];
cx q[39],q[38];
cx q[86],q[85];
cx q[87],q[37];
u3(0.0*pi,-0.5*pi,4.1816901138162095*pi) q[38];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[85];
u3(1.68169011381621*pi,-0.5*pi,0.5*pi) q[86];
u3(0.0*pi,-0.5*pi,2.1816901138162095*pi) q[37];
cx q[39],q[38];
cx q[86],q[85];
u3(0.5*pi,-0.5*pi,0.5*pi) q[87];
u3(2.5*pi,0.3183098861837902*pi,4.0*pi) q[38];
u3(0.5*pi,0.0*pi,0.5*pi) q[39];
u3(0.0*pi,-0.5*pi,1.6816901138162093*pi) q[85];
cx q[87],q[86];
cx q[88],q[38];
cx q[40],q[39];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[86];
u3(1.68169011381621*pi,-0.5*pi,0.5*pi) q[87];
u3(1.0*pi,-0.5*pi,0.8183098861837907*pi) q[38];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[39];
cx q[87],q[86];
cx q[88],q[38];
cx q[40],q[39];
u3(0.0*pi,-0.5*pi,1.6816901138162093*pi) q[86];
cx q[89],q[39];
u3(0.5*pi,-0.5*pi,0.5*pi) q[40];
u3(0.5*pi,-0.5*pi,0.5*pi) q[88];
u3(1.0*pi,-0.5*pi,0.8183098861837907*pi) q[39];
cx q[41],q[40];
cx q[88],q[87];
cx q[89],q[39];
u3(0.0*pi,-0.5*pi,4.1816901138162095*pi) q[40];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[87];
u3(1.68169011381621*pi,-0.5*pi,0.5*pi) q[88];
u3(0.0*pi,-0.5*pi,2.1816901138162095*pi) q[39];
cx q[41],q[40];
cx q[88],q[87];
u3(0.5*pi,-0.5*pi,0.5*pi) q[89];
u3(2.5*pi,0.3183098861837902*pi,4.0*pi) q[40];
u3(0.5*pi,0.0*pi,0.5*pi) q[41];
u3(0.0*pi,-0.5*pi,1.6816901138162093*pi) q[87];
cx q[89],q[88];
cx q[90],q[40];
cx q[42],q[41];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[88];
u3(1.68169011381621*pi,-0.5*pi,0.5*pi) q[89];
u3(1.0*pi,-0.5*pi,0.8183098861837907*pi) q[40];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[41];
cx q[89],q[88];
cx q[90],q[40];
cx q[42],q[41];
u3(0.0*pi,-0.5*pi,1.6816901138162093*pi) q[88];
cx q[91],q[41];
u3(0.5*pi,-0.5*pi,0.5*pi) q[42];
u3(0.5*pi,-0.5*pi,0.5*pi) q[90];
u3(1.0*pi,-0.5*pi,0.8183098861837907*pi) q[41];
cx q[43],q[42];
cx q[90],q[89];
cx q[91],q[41];
u3(0.0*pi,-0.5*pi,4.1816901138162095*pi) q[42];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[89];
u3(1.68169011381621*pi,-0.5*pi,0.5*pi) q[90];
u3(0.0*pi,-0.5*pi,2.1816901138162095*pi) q[41];
cx q[43],q[42];
cx q[90],q[89];
u3(0.5*pi,-0.5*pi,0.5*pi) q[91];
u3(2.5*pi,0.3183098861837902*pi,4.0*pi) q[42];
u3(0.5*pi,0.0*pi,0.5*pi) q[43];
u3(0.0*pi,-0.5*pi,1.6816901138162093*pi) q[89];
cx q[91],q[90];
cx q[92],q[42];
cx q[44],q[43];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[90];
u3(1.68169011381621*pi,-0.5*pi,0.5*pi) q[91];
u3(1.0*pi,-0.5*pi,0.8183098861837907*pi) q[42];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[43];
cx q[91],q[90];
cx q[92],q[42];
cx q[44],q[43];
u3(0.0*pi,-0.5*pi,1.6816901138162093*pi) q[90];
cx q[93],q[43];
u3(0.5*pi,-0.5*pi,0.5*pi) q[44];
u3(0.5*pi,-0.5*pi,0.5*pi) q[92];
u3(1.0*pi,-0.5*pi,0.8183098861837907*pi) q[43];
cx q[45],q[44];
cx q[92],q[91];
cx q[93],q[43];
u3(0.0*pi,-0.5*pi,4.1816901138162095*pi) q[44];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[91];
u3(1.68169011381621*pi,-0.5*pi,0.5*pi) q[92];
u3(0.0*pi,-0.5*pi,2.1816901138162095*pi) q[43];
cx q[45],q[44];
cx q[92],q[91];
u3(0.5*pi,-0.5*pi,0.5*pi) q[93];
u3(2.5*pi,0.3183098861837902*pi,4.0*pi) q[44];
u3(0.5*pi,0.0*pi,0.5*pi) q[45];
u3(0.0*pi,-0.5*pi,1.6816901138162093*pi) q[91];
cx q[93],q[92];
cx q[94],q[44];
cx q[46],q[45];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[92];
u3(1.68169011381621*pi,-0.5*pi,0.5*pi) q[93];
u3(1.0*pi,-0.5*pi,0.8183098861837907*pi) q[44];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[45];
cx q[93],q[92];
cx q[94],q[44];
cx q[46],q[45];
u3(0.0*pi,-0.5*pi,1.6816901138162093*pi) q[92];
cx q[95],q[45];
u3(0.5*pi,-0.5*pi,0.5*pi) q[46];
u3(0.5*pi,-0.5*pi,0.5*pi) q[94];
u3(1.0*pi,-0.5*pi,0.8183098861837907*pi) q[45];
cx q[47],q[46];
cx q[94],q[93];
cx q[95],q[45];
u3(0.0*pi,-0.5*pi,4.1816901138162095*pi) q[46];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[93];
u3(1.68169011381621*pi,-0.5*pi,0.5*pi) q[94];
u3(0.0*pi,-0.5*pi,2.1816901138162095*pi) q[45];
cx q[47],q[46];
cx q[94],q[93];
u3(0.5*pi,-0.5*pi,0.5*pi) q[95];
u3(2.5*pi,0.3183098861837902*pi,4.0*pi) q[46];
u3(0.5*pi,0.0*pi,0.5*pi) q[47];
u3(0.0*pi,-0.5*pi,1.6816901138162093*pi) q[93];
cx q[95],q[94];
cx q[96],q[46];
cx q[48],q[47];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[94];
u3(1.68169011381621*pi,-0.5*pi,0.5*pi) q[95];
u3(1.0*pi,-0.5*pi,0.8183098861837907*pi) q[46];
u3(3.5*pi,-0.5*pi,4.1816901138162095*pi) q[47];
cx q[95],q[94];
cx q[96],q[46];
cx q[48],q[47];
u3(0.0*pi,-0.5*pi,1.6816901138162093*pi) q[94];
u3(1.0*pi,-0.5*pi,4.1816901138162095*pi) q[47];
u3(0.5*pi,-0.5*pi,0.5*pi) q[48];
u3(0.5*pi,-0.5*pi,0.5*pi) q[96];
cx q[97],q[47];
cx q[49],q[48];
cx q[96],q[95];
u3(1.0*pi,-0.5*pi,0.8183098861837907*pi) q[47];
u3(0.0*pi,-0.5*pi,4.1816901138162095*pi) q[48];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[95];
u3(1.68169011381621*pi,-0.5*pi,0.5*pi) q[96];
cx q[97],q[47];
cx q[49],q[48];
cx q[96],q[95];
u3(2.5*pi,0.3183098861837902*pi,4.0*pi) q[48];
u3(3.5*pi,0.6816901138162093*pi,4.0*pi) q[49];
u3(0.0*pi,-0.5*pi,1.6816901138162093*pi) q[95];
u3(0.5*pi,-0.5*pi,0.5*pi) q[97];
cx q[99],q[49];
cx q[97],q[96];
u3(0.0*pi,-0.5*pi,0.8183098861837907*pi) q[49];
u3(1.5*pi,-0.5*pi,4.1816901138162095*pi) q[96];
u3(0.6816901138162099*pi,-0.5*pi,0.5*pi) q[97];
cx q[99],q[49];
cx q[97],q[96];
u3(0.0*pi,-0.5*pi,1.6816901138162093*pi) q[96];
u3(0.5*pi,0.0*pi,0.5*pi) q[97];
u3(0.5*pi,0.0*pi,0.5*pi) q[99];
cx q[98],q[97];
cx q[98],q[48];
cx q[97],q[96];
u3(1.0*pi,-0.5*pi,0.8183098861837907*pi) q[48];
cx q[96],q[95];
cx q[98],q[48];
cx q[95],q[94];
cx q[94],q[93];
u3(0.5*pi,0.0*pi,0.5*pi) q[98];
cx q[93],q[92];
cx q[92],q[91];
cx q[91],q[90];
cx q[90],q[89];
cx q[89],q[88];
cx q[88],q[87];
cx q[87],q[86];
cx q[86],q[85];
cx q[85],q[84];
cx q[84],q[83];
cx q[83],q[82];
cx q[82],q[81];
cx q[81],q[80];
cx q[80],q[79];
cx q[79],q[78];
cx q[78],q[77];
cx q[77],q[76];
cx q[76],q[75];
cx q[75],q[74];
cx q[74],q[73];
cx q[73],q[72];
cx q[72],q[71];
cx q[71],q[70];
cx q[70],q[69];
cx q[69],q[68];
cx q[68],q[67];
cx q[67],q[66];
cx q[66],q[65];
cx q[65],q[64];
cx q[64],q[63];
cx q[63],q[62];
cx q[62],q[61];
cx q[61],q[60];
cx q[60],q[59];
cx q[59],q[58];
cx q[58],q[57];
cx q[57],q[56];
cx q[56],q[55];
cx q[55],q[54];
cx q[54],q[53];
cx q[53],q[52];
cx q[52],q[51];
cx q[51],q[99];
u3(0.0*pi,-0.5*pi,1.0*pi) q[99];
cx q[99],q[50];
u3(0.5*pi,-0.5*pi,4.1816901138162095*pi) q[50];
u3(1.68169011381621*pi,0.0*pi,0.5*pi) q[99];
cx q[99],q[50];
u3(0.0*pi,-0.5*pi,0.6816901138162095*pi) q[50];
u3(0.5*pi,-0.5*pi,0.5*pi) q[99];
cx q[51],q[99];
cx q[52],q[51];
cx q[53],q[52];
cx q[54],q[53];
cx q[55],q[54];
cx q[56],q[55];
cx q[57],q[56];
cx q[58],q[57];
cx q[59],q[58];
cx q[60],q[59];
cx q[61],q[60];
cx q[62],q[61];
cx q[63],q[62];
cx q[64],q[63];
cx q[65],q[64];
cx q[66],q[65];
cx q[67],q[66];
cx q[68],q[67];
cx q[69],q[68];
cx q[70],q[69];
cx q[71],q[70];
cx q[72],q[71];
cx q[73],q[72];
cx q[74],q[73];
cx q[75],q[74];
cx q[76],q[75];
cx q[77],q[76];
cx q[78],q[77];
cx q[79],q[78];
cx q[80],q[79];
cx q[81],q[80];
cx q[82],q[81];
cx q[83],q[82];
cx q[84],q[83];
cx q[85],q[84];
cx q[86],q[85];
cx q[87],q[86];
cx q[88],q[87];
cx q[89],q[88];
cx q[90],q[89];
cx q[91],q[90];
cx q[92],q[91];
cx q[93],q[92];
cx q[94],q[93];
cx q[95],q[94];
cx q[96],q[95];
cx q[97],q[96];
cx q[98],q[97];
u3(2.5*pi,0.0*pi,1.3183098861837907*pi) q[97];
u3(2.5*pi,-0.5*pi,3.6816901138162095*pi) q[98];
cx q[97],q[98];
u3(1.5*pi,0.6816901138162099*pi,0.5*pi) q[97];
u3(0.0*pi,-0.5*pi,1.0*pi) q[98];
cx q[99],q[98];
u3(3.681690113816209*pi,0.0*pi,4.0*pi) q[98];
u3(1.6816901138162093*pi,-0.5*pi,1.0*pi) q[99];
cx q[99],q[98];
u3(0.0*pi,-0.5*pi,2.1816901138162095*pi) q[98];
u3(1.5*pi,3.18169011381621*pi,0.5*pi) q[99];
