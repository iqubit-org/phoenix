OPENQASM 2.0;
include "qelib1.inc";
qreg q[112];
cx q[0],q[4];
rz(1.0) q[4];
cx q[0],q[4];
rz(1.0) q[0];
cx q[0],q[8];
rz(1.0) q[8];
cx q[0],q[8];
rz(1.0) q[8];
cx q[8],q[12];
rz(1.0) q[12];
cx q[8],q[12];
rz(1.0) q[12];
cx q[4],q[12];
rz(1.0) q[12];
cx q[4],q[12];
rz(1.0) q[4];
cx q[4],q[32];
rz(1.0) q[32];
cx q[4],q[32];
rz(1.0) q[32];
cx q[32],q[60];
rz(1.0) q[60];
cx q[32],q[60];
rz(1.0) q[60];
cx q[60],q[84];
rz(1.0) q[84];
cx q[60],q[84];
rz(1.0) q[84];
cx q[84],q[104];
rz(1.0) q[104];
cx q[84],q[104];
rz(1.0) q[104];
cx q[104],q[108];
rz(1.0) q[108];
cx q[104],q[108];
rz(1.0) q[108];
cx q[68],q[108];
rz(1.0) q[108];
cx q[68],q[108];
rz(1.0) q[68];
cx q[64],q[68];
rz(1.0) q[68];
cx q[64],q[68];
rz(1.0) q[64];
cx q[64],q[72];
rz(1.0) q[72];
cx q[64],q[72];
rz(1.0) q[72];
cx q[52],q[72];
rz(1.0) q[72];
cx q[52],q[72];
rz(1.0) q[52];
cx q[24],q[52];
rz(1.0) q[52];
cx q[24],q[52];
rz(1.0) q[24];
cx q[24],q[28];
rz(1.0) q[28];
cx q[24],q[28];
rz(1.0) q[28];
cx q[28],q[44];
rz(1.0) q[44];
cx q[28],q[44];
rz(1.0) q[44];
cx q[36],q[44];
rz(1.0) q[44];
cx q[36],q[44];
rz(1.0) q[36];
cx q[16],q[36];
rz(1.0) q[36];
cx q[16],q[36];
rz(1.0) q[16];
cx q[16],q[20];
rz(1.0) q[20];
cx q[16],q[20];
rz(1.0) q[20];
cx q[20],q[76];
rz(1.0) q[76];
cx q[20],q[76];
rz(1.0) q[76];
cx q[32],q[76];
rz(1.0) q[76];
cx q[32],q[76];
cx q[40],q[76];
rz(1.0) q[76];
cx q[40],q[76];
rz(1.0) q[40];
cx q[40],q[56];
rz(1.0) q[56];
cx q[40],q[56];
rz(1.0) q[56];
cx q[48],q[56];
rz(1.0) q[56];
cx q[48],q[56];
rz(1.0) q[48];
cx q[48],q[100];
rz(1.0) q[100];
cx q[48],q[100];
rz(1.0) q[100];
cx q[80],q[100];
rz(1.0) q[100];
cx q[80],q[100];
rz(1.0) q[80];
cx q[80],q[88];
rz(1.0) q[88];
cx q[80],q[88];
rz(1.0) q[88];
cx q[88],q[96];
rz(1.0) q[96];
cx q[88],q[96];
rz(1.0) q[96];
cx q[92],q[96];
rz(1.0) q[96];
cx q[92],q[96];
rz(1.0) q[92];
cx q[40],q[92];
rz(1.0) q[92];
cx q[40],q[92];
cx q[80],q[92];
rz(1.0) q[92];
cx q[80],q[92];
cx q[92],q[108];
rz(1.0) q[108];
cx q[92],q[108];
cx q[44],q[92];
rz(1.0) q[92];
cx q[44],q[92];
cx q[44],q[52];
rz(1.0) q[52];
cx q[44],q[52];
cx q[12],q[44];
rz(1.0) q[44];
cx q[12],q[44];
cx q[0],q[12];
rz(1.0) q[12];
cx q[0],q[12];
cx q[0],q[28];
rz(1.0) q[28];
cx q[0],q[28];
cx q[0],q[16];
rz(1.0) q[16];
cx q[0],q[16];
cx q[16],q[64];
rz(1.0) q[64];
cx q[16],q[64];
cx q[64],q[84];
rz(1.0) q[84];
cx q[64],q[84];
cx q[32],q[84];
rz(1.0) q[84];
cx q[32],q[84];
cx q[12],q[32];
rz(1.0) q[32];
cx q[12],q[32];
cx q[12],q[60];
rz(1.0) q[60];
cx q[12],q[60];
cx q[60],q[100];
rz(1.0) q[100];
cx q[60],q[100];
cx q[52],q[60];
rz(1.0) q[60];
cx q[52],q[60];
cx q[52],q[56];
rz(1.0) q[56];
cx q[52],q[56];
cx q[56],q[88];
rz(1.0) q[88];
cx q[56],q[88];
cx q[88],q[104];
rz(1.0) q[104];
cx q[88],q[104];
cx q[68],q[104];
rz(1.0) q[104];
cx q[68],q[104];
cx q[68],q[96];
rz(1.0) q[96];
cx q[68],q[96];
cx q[44],q[96];
rz(1.0) q[96];
cx q[44],q[96];
cx q[96],q[104];
rz(1.0) q[104];
cx q[96],q[104];
cx q[1],q[5];
rz(1.0) q[5];
cx q[1],q[5];
rz(1.0) q[1];
cx q[1],q[9];
rz(1.0) q[9];
cx q[1],q[9];
rz(1.0) q[9];
cx q[9],q[13];
rz(1.0) q[13];
cx q[9],q[13];
rz(1.0) q[13];
cx q[5],q[13];
rz(1.0) q[13];
cx q[5],q[13];
rz(1.0) q[5];
cx q[5],q[33];
rz(1.0) q[33];
cx q[5],q[33];
rz(1.0) q[33];
cx q[33],q[61];
rz(1.0) q[61];
cx q[33],q[61];
rz(1.0) q[61];
cx q[61],q[85];
rz(1.0) q[85];
cx q[61],q[85];
rz(1.0) q[85];
cx q[85],q[105];
rz(1.0) q[105];
cx q[85],q[105];
rz(1.0) q[105];
cx q[105],q[109];
rz(1.0) q[109];
cx q[105],q[109];
rz(1.0) q[109];
cx q[69],q[109];
rz(1.0) q[109];
cx q[69],q[109];
rz(1.0) q[69];
cx q[65],q[69];
rz(1.0) q[69];
cx q[65],q[69];
rz(1.0) q[65];
cx q[65],q[73];
rz(1.0) q[73];
cx q[65],q[73];
rz(1.0) q[73];
cx q[53],q[73];
rz(1.0) q[73];
cx q[53],q[73];
rz(1.0) q[53];
cx q[25],q[53];
rz(1.0) q[53];
cx q[25],q[53];
rz(1.0) q[25];
cx q[25],q[29];
rz(1.0) q[29];
cx q[25],q[29];
rz(1.0) q[29];
cx q[29],q[45];
rz(1.0) q[45];
cx q[29],q[45];
rz(1.0) q[45];
cx q[37],q[45];
rz(1.0) q[45];
cx q[37],q[45];
rz(1.0) q[37];
cx q[17],q[37];
rz(1.0) q[37];
cx q[17],q[37];
rz(1.0) q[17];
cx q[17],q[21];
rz(1.0) q[21];
cx q[17],q[21];
rz(1.0) q[21];
cx q[21],q[77];
rz(1.0) q[77];
cx q[21],q[77];
rz(1.0) q[77];
cx q[33],q[77];
rz(1.0) q[77];
cx q[33],q[77];
cx q[41],q[77];
rz(1.0) q[77];
cx q[41],q[77];
rz(1.0) q[41];
cx q[41],q[57];
rz(1.0) q[57];
cx q[41],q[57];
rz(1.0) q[57];
cx q[49],q[57];
rz(1.0) q[57];
cx q[49],q[57];
rz(1.0) q[49];
cx q[49],q[101];
rz(1.0) q[101];
cx q[49],q[101];
rz(1.0) q[101];
cx q[81],q[101];
rz(1.0) q[101];
cx q[81],q[101];
rz(1.0) q[81];
cx q[81],q[89];
rz(1.0) q[89];
cx q[81],q[89];
rz(1.0) q[89];
cx q[89],q[97];
rz(1.0) q[97];
cx q[89],q[97];
rz(1.0) q[97];
cx q[93],q[97];
rz(1.0) q[97];
cx q[93],q[97];
rz(1.0) q[93];
cx q[41],q[93];
rz(1.0) q[93];
cx q[41],q[93];
cx q[81],q[93];
rz(1.0) q[93];
cx q[81],q[93];
cx q[93],q[109];
rz(1.0) q[109];
cx q[93],q[109];
cx q[45],q[93];
rz(1.0) q[93];
cx q[45],q[93];
cx q[45],q[53];
rz(1.0) q[53];
cx q[45],q[53];
cx q[13],q[45];
rz(1.0) q[45];
cx q[13],q[45];
cx q[1],q[13];
rz(1.0) q[13];
cx q[1],q[13];
cx q[1],q[29];
rz(1.0) q[29];
cx q[1],q[29];
cx q[1],q[17];
rz(1.0) q[17];
cx q[1],q[17];
cx q[17],q[65];
rz(1.0) q[65];
cx q[17],q[65];
cx q[65],q[85];
rz(1.0) q[85];
cx q[65],q[85];
cx q[33],q[85];
rz(1.0) q[85];
cx q[33],q[85];
cx q[13],q[33];
rz(1.0) q[33];
cx q[13],q[33];
cx q[13],q[61];
rz(1.0) q[61];
cx q[13],q[61];
cx q[61],q[101];
rz(1.0) q[101];
cx q[61],q[101];
cx q[53],q[61];
rz(1.0) q[61];
cx q[53],q[61];
cx q[53],q[57];
rz(1.0) q[57];
cx q[53],q[57];
cx q[57],q[89];
rz(1.0) q[89];
cx q[57],q[89];
cx q[89],q[105];
rz(1.0) q[105];
cx q[89],q[105];
cx q[69],q[105];
rz(1.0) q[105];
cx q[69],q[105];
cx q[69],q[97];
rz(1.0) q[97];
cx q[69],q[97];
cx q[45],q[97];
rz(1.0) q[97];
cx q[45],q[97];
cx q[97],q[105];
rz(1.0) q[105];
cx q[97],q[105];
cx q[2],q[6];
rz(1.0) q[6];
cx q[2],q[6];
rz(1.0) q[2];
cx q[2],q[10];
rz(1.0) q[10];
cx q[2],q[10];
rz(1.0) q[10];
cx q[10],q[14];
rz(1.0) q[14];
cx q[10],q[14];
rz(1.0) q[14];
cx q[6],q[14];
rz(1.0) q[14];
cx q[6],q[14];
rz(1.0) q[6];
cx q[6],q[34];
rz(1.0) q[34];
cx q[6],q[34];
rz(1.0) q[34];
cx q[34],q[62];
rz(1.0) q[62];
cx q[34],q[62];
rz(1.0) q[62];
cx q[62],q[86];
rz(1.0) q[86];
cx q[62],q[86];
rz(1.0) q[86];
cx q[86],q[106];
rz(1.0) q[106];
cx q[86],q[106];
rz(1.0) q[106];
cx q[106],q[110];
rz(1.0) q[110];
cx q[106],q[110];
rz(1.0) q[110];
cx q[70],q[110];
rz(1.0) q[110];
cx q[70],q[110];
rz(1.0) q[70];
cx q[66],q[70];
rz(1.0) q[70];
cx q[66],q[70];
rz(1.0) q[66];
cx q[66],q[74];
rz(1.0) q[74];
cx q[66],q[74];
rz(1.0) q[74];
cx q[54],q[74];
rz(1.0) q[74];
cx q[54],q[74];
rz(1.0) q[54];
cx q[26],q[54];
rz(1.0) q[54];
cx q[26],q[54];
rz(1.0) q[26];
cx q[26],q[30];
rz(1.0) q[30];
cx q[26],q[30];
rz(1.0) q[30];
cx q[30],q[46];
rz(1.0) q[46];
cx q[30],q[46];
rz(1.0) q[46];
cx q[38],q[46];
rz(1.0) q[46];
cx q[38],q[46];
rz(1.0) q[38];
cx q[18],q[38];
rz(1.0) q[38];
cx q[18],q[38];
rz(1.0) q[18];
cx q[18],q[22];
rz(1.0) q[22];
cx q[18],q[22];
rz(1.0) q[22];
cx q[22],q[78];
rz(1.0) q[78];
cx q[22],q[78];
rz(1.0) q[78];
cx q[34],q[78];
rz(1.0) q[78];
cx q[34],q[78];
cx q[42],q[78];
rz(1.0) q[78];
cx q[42],q[78];
rz(1.0) q[42];
cx q[42],q[58];
rz(1.0) q[58];
cx q[42],q[58];
rz(1.0) q[58];
cx q[50],q[58];
rz(1.0) q[58];
cx q[50],q[58];
rz(1.0) q[50];
cx q[50],q[102];
rz(1.0) q[102];
cx q[50],q[102];
rz(1.0) q[102];
cx q[82],q[102];
rz(1.0) q[102];
cx q[82],q[102];
rz(1.0) q[82];
cx q[82],q[90];
rz(1.0) q[90];
cx q[82],q[90];
rz(1.0) q[90];
cx q[90],q[98];
rz(1.0) q[98];
cx q[90],q[98];
rz(1.0) q[98];
cx q[94],q[98];
rz(1.0) q[98];
cx q[94],q[98];
rz(1.0) q[94];
cx q[42],q[94];
rz(1.0) q[94];
cx q[42],q[94];
cx q[82],q[94];
rz(1.0) q[94];
cx q[82],q[94];
cx q[94],q[110];
rz(1.0) q[110];
cx q[94],q[110];
cx q[46],q[94];
rz(1.0) q[94];
cx q[46],q[94];
cx q[46],q[54];
rz(1.0) q[54];
cx q[46],q[54];
cx q[14],q[46];
rz(1.0) q[46];
cx q[14],q[46];
cx q[2],q[14];
rz(1.0) q[14];
cx q[2],q[14];
cx q[2],q[30];
rz(1.0) q[30];
cx q[2],q[30];
cx q[2],q[18];
rz(1.0) q[18];
cx q[2],q[18];
cx q[18],q[66];
rz(1.0) q[66];
cx q[18],q[66];
cx q[66],q[86];
rz(1.0) q[86];
cx q[66],q[86];
cx q[34],q[86];
rz(1.0) q[86];
cx q[34],q[86];
cx q[14],q[34];
rz(1.0) q[34];
cx q[14],q[34];
cx q[14],q[62];
rz(1.0) q[62];
cx q[14],q[62];
cx q[62],q[102];
rz(1.0) q[102];
cx q[62],q[102];
cx q[54],q[62];
rz(1.0) q[62];
cx q[54],q[62];
cx q[54],q[58];
rz(1.0) q[58];
cx q[54],q[58];
cx q[58],q[90];
rz(1.0) q[90];
cx q[58],q[90];
cx q[90],q[106];
rz(1.0) q[106];
cx q[90],q[106];
cx q[70],q[106];
rz(1.0) q[106];
cx q[70],q[106];
cx q[70],q[98];
rz(1.0) q[98];
cx q[70],q[98];
cx q[46],q[98];
rz(1.0) q[98];
cx q[46],q[98];
cx q[98],q[106];
rz(1.0) q[106];
cx q[98],q[106];
cx q[3],q[7];
rz(1.0) q[7];
cx q[3],q[7];
rz(1.0) q[3];
cx q[3],q[11];
rz(1.0) q[11];
cx q[3],q[11];
rz(1.0) q[11];
cx q[11],q[15];
rz(1.0) q[15];
cx q[11],q[15];
rz(1.0) q[15];
cx q[7],q[15];
rz(1.0) q[15];
cx q[7],q[15];
rz(1.0) q[7];
cx q[7],q[35];
rz(1.0) q[35];
cx q[7],q[35];
rz(1.0) q[35];
cx q[35],q[63];
rz(1.0) q[63];
cx q[35],q[63];
rz(1.0) q[63];
cx q[63],q[87];
rz(1.0) q[87];
cx q[63],q[87];
rz(1.0) q[87];
cx q[87],q[107];
rz(1.0) q[107];
cx q[87],q[107];
rz(1.0) q[107];
cx q[107],q[111];
rz(1.0) q[111];
cx q[107],q[111];
rz(1.0) q[111];
cx q[71],q[111];
rz(1.0) q[111];
cx q[71],q[111];
rz(1.0) q[71];
cx q[67],q[71];
rz(1.0) q[71];
cx q[67],q[71];
rz(1.0) q[67];
cx q[67],q[75];
rz(1.0) q[75];
cx q[67],q[75];
rz(1.0) q[75];
cx q[55],q[75];
rz(1.0) q[75];
cx q[55],q[75];
rz(1.0) q[55];
cx q[27],q[55];
rz(1.0) q[55];
cx q[27],q[55];
rz(1.0) q[27];
cx q[27],q[31];
rz(1.0) q[31];
cx q[27],q[31];
rz(1.0) q[31];
cx q[31],q[47];
rz(1.0) q[47];
cx q[31],q[47];
rz(1.0) q[47];
cx q[39],q[47];
rz(1.0) q[47];
cx q[39],q[47];
rz(1.0) q[39];
cx q[19],q[39];
rz(1.0) q[39];
cx q[19],q[39];
rz(1.0) q[19];
cx q[19],q[23];
rz(1.0) q[23];
cx q[19],q[23];
rz(1.0) q[23];
cx q[23],q[79];
rz(1.0) q[79];
cx q[23],q[79];
rz(1.0) q[79];
cx q[35],q[79];
rz(1.0) q[79];
cx q[35],q[79];
cx q[43],q[79];
rz(1.0) q[79];
cx q[43],q[79];
rz(1.0) q[43];
cx q[43],q[59];
rz(1.0) q[59];
cx q[43],q[59];
rz(1.0) q[59];
cx q[51],q[59];
rz(1.0) q[59];
cx q[51],q[59];
rz(1.0) q[51];
cx q[51],q[103];
rz(1.0) q[103];
cx q[51],q[103];
rz(1.0) q[103];
cx q[83],q[103];
rz(1.0) q[103];
cx q[83],q[103];
rz(1.0) q[83];
cx q[83],q[91];
rz(1.0) q[91];
cx q[83],q[91];
rz(1.0) q[91];
cx q[91],q[99];
rz(1.0) q[99];
cx q[91],q[99];
rz(1.0) q[99];
cx q[95],q[99];
rz(1.0) q[99];
cx q[95],q[99];
rz(1.0) q[95];
cx q[43],q[95];
rz(1.0) q[95];
cx q[43],q[95];
cx q[83],q[95];
rz(1.0) q[95];
cx q[83],q[95];
cx q[95],q[111];
rz(1.0) q[111];
cx q[95],q[111];
cx q[47],q[95];
rz(1.0) q[95];
cx q[47],q[95];
cx q[47],q[55];
rz(1.0) q[55];
cx q[47],q[55];
cx q[15],q[47];
rz(1.0) q[47];
cx q[15],q[47];
cx q[3],q[15];
rz(1.0) q[15];
cx q[3],q[15];
cx q[3],q[31];
rz(1.0) q[31];
cx q[3],q[31];
cx q[3],q[19];
rz(1.0) q[19];
cx q[3],q[19];
cx q[19],q[67];
rz(1.0) q[67];
cx q[19],q[67];
cx q[67],q[87];
rz(1.0) q[87];
cx q[67],q[87];
cx q[35],q[87];
rz(1.0) q[87];
cx q[35],q[87];
cx q[15],q[35];
rz(1.0) q[35];
cx q[15],q[35];
cx q[15],q[63];
rz(1.0) q[63];
cx q[15],q[63];
cx q[63],q[103];
rz(1.0) q[103];
cx q[63],q[103];
cx q[55],q[63];
rz(1.0) q[63];
cx q[55],q[63];
cx q[55],q[59];
rz(1.0) q[59];
cx q[55],q[59];
cx q[59],q[91];
rz(1.0) q[91];
cx q[59],q[91];
cx q[91],q[107];
rz(1.0) q[107];
cx q[91],q[107];
cx q[71],q[107];
rz(1.0) q[107];
cx q[71],q[107];
cx q[71],q[99];
rz(1.0) q[99];
cx q[71],q[99];
cx q[47],q[99];
rz(1.0) q[99];
cx q[47],q[99];
cx q[99],q[107];
rz(1.0) q[107];
cx q[99],q[107];
cx q[8],q[36];
rz(1.0) q[36];
cx q[8],q[36];
cx q[4],q[36];
rz(1.0) q[36];
cx q[4],q[36];
cx q[36],q[48];
rz(1.0) q[48];
cx q[36],q[48];
cx q[20],q[48];
rz(1.0) q[48];
cx q[20],q[48];
cx q[20],q[72];
rz(1.0) q[72];
cx q[20],q[72];
cx q[24],q[72];
rz(1.0) q[72];
cx q[24],q[72];
cx q[24],q[80];
rz(1.0) q[80];
cx q[24],q[80];
cx q[64],q[80];
rz(1.0) q[80];
cx q[64],q[80];
cx q[64],q[88];
rz(1.0) q[88];
cx q[64],q[88];
cx q[9],q[37];
rz(1.0) q[37];
cx q[9],q[37];
cx q[5],q[37];
rz(1.0) q[37];
cx q[5],q[37];
cx q[37],q[49];
rz(1.0) q[49];
cx q[37],q[49];
cx q[21],q[49];
rz(1.0) q[49];
cx q[21],q[49];
cx q[21],q[73];
rz(1.0) q[73];
cx q[21],q[73];
cx q[25],q[73];
rz(1.0) q[73];
cx q[25],q[73];
cx q[25],q[81];
rz(1.0) q[81];
cx q[25],q[81];
cx q[65],q[81];
rz(1.0) q[81];
cx q[65],q[81];
cx q[65],q[89];
rz(1.0) q[89];
cx q[65],q[89];
cx q[10],q[38];
rz(1.0) q[38];
cx q[10],q[38];
cx q[6],q[38];
rz(1.0) q[38];
cx q[6],q[38];
cx q[38],q[50];
rz(1.0) q[50];
cx q[38],q[50];
cx q[22],q[50];
rz(1.0) q[50];
cx q[22],q[50];
cx q[22],q[74];
rz(1.0) q[74];
cx q[22],q[74];
cx q[26],q[74];
rz(1.0) q[74];
cx q[26],q[74];
cx q[26],q[82];
rz(1.0) q[82];
cx q[26],q[82];
cx q[66],q[82];
rz(1.0) q[82];
cx q[66],q[82];
cx q[66],q[90];
rz(1.0) q[90];
cx q[66],q[90];
cx q[11],q[39];
rz(1.0) q[39];
cx q[11],q[39];
cx q[7],q[39];
rz(1.0) q[39];
cx q[7],q[39];
cx q[39],q[51];
rz(1.0) q[51];
cx q[39],q[51];
cx q[23],q[51];
rz(1.0) q[51];
cx q[23],q[51];
cx q[23],q[75];
rz(1.0) q[75];
cx q[23],q[75];
cx q[27],q[75];
rz(1.0) q[75];
cx q[27],q[75];
cx q[27],q[83];
rz(1.0) q[83];
cx q[27],q[83];
cx q[67],q[83];
rz(1.0) q[83];
cx q[67],q[83];
cx q[67],q[91];
rz(1.0) q[91];
cx q[67],q[91];
cx q[8],q[40];
rz(1.0) q[40];
cx q[8],q[40];
cx q[8],q[48];
rz(1.0) q[48];
cx q[8],q[48];
cx q[8],q[44];
rz(1.0) q[44];
cx q[8],q[44];
cx q[44],q[48];
rz(1.0) q[48];
cx q[44],q[48];
cx q[44],q[56];
rz(1.0) q[56];
cx q[44],q[56];
cx q[56],q[76];
rz(1.0) q[76];
cx q[56],q[76];
cx q[12],q[56];
rz(1.0) q[56];
cx q[12],q[56];
cx q[12],q[36];
rz(1.0) q[36];
cx q[12],q[36];
cx q[12],q[52];
rz(1.0) q[52];
cx q[12],q[52];
cx q[52],q[84];
rz(1.0) q[84];
cx q[52],q[84];
cx q[9],q[41];
rz(1.0) q[41];
cx q[9],q[41];
cx q[9],q[49];
rz(1.0) q[49];
cx q[9],q[49];
cx q[9],q[45];
rz(1.0) q[45];
cx q[9],q[45];
cx q[45],q[49];
rz(1.0) q[49];
cx q[45],q[49];
cx q[45],q[57];
rz(1.0) q[57];
cx q[45],q[57];
cx q[57],q[77];
rz(1.0) q[77];
cx q[57],q[77];
cx q[13],q[57];
rz(1.0) q[57];
cx q[13],q[57];
cx q[13],q[37];
rz(1.0) q[37];
cx q[13],q[37];
cx q[13],q[53];
rz(1.0) q[53];
cx q[13],q[53];
cx q[53],q[85];
rz(1.0) q[85];
cx q[53],q[85];
cx q[10],q[42];
rz(1.0) q[42];
cx q[10],q[42];
cx q[10],q[50];
rz(1.0) q[50];
cx q[10],q[50];
cx q[10],q[46];
rz(1.0) q[46];
cx q[10],q[46];
cx q[46],q[50];
rz(1.0) q[50];
cx q[46],q[50];
cx q[46],q[58];
rz(1.0) q[58];
cx q[46],q[58];
cx q[58],q[78];
rz(1.0) q[78];
cx q[58],q[78];
cx q[14],q[58];
rz(1.0) q[58];
cx q[14],q[58];
cx q[14],q[38];
rz(1.0) q[38];
cx q[14],q[38];
cx q[14],q[54];
rz(1.0) q[54];
cx q[14],q[54];
cx q[54],q[86];
rz(1.0) q[86];
cx q[54],q[86];
cx q[11],q[43];
rz(1.0) q[43];
cx q[11],q[43];
cx q[11],q[51];
rz(1.0) q[51];
cx q[11],q[51];
cx q[11],q[47];
rz(1.0) q[47];
cx q[11],q[47];
cx q[47],q[51];
rz(1.0) q[51];
cx q[47],q[51];
cx q[47],q[59];
rz(1.0) q[59];
cx q[47],q[59];
cx q[59],q[79];
rz(1.0) q[79];
cx q[59],q[79];
cx q[15],q[59];
rz(1.0) q[59];
cx q[15],q[59];
cx q[15],q[39];
rz(1.0) q[39];
cx q[15],q[39];
cx q[15],q[55];
rz(1.0) q[55];
cx q[15],q[55];
cx q[55],q[87];
rz(1.0) q[87];
cx q[55],q[87];
cx q[28],q[72];
rz(1.0) q[72];
cx q[28],q[72];
cx q[29],q[73];
rz(1.0) q[73];
cx q[29],q[73];
cx q[30],q[74];
rz(1.0) q[74];
cx q[30],q[74];
cx q[31],q[75];
rz(1.0) q[75];
cx q[31],q[75];
cx q[16],q[68];
rz(1.0) q[68];
cx q[16],q[68];
cx q[17],q[69];
rz(1.0) q[69];
cx q[17],q[69];
cx q[18],q[70];
rz(1.0) q[70];
cx q[18],q[70];
cx q[19],q[71];
rz(1.0) q[71];
cx q[19],q[71];
cx q[0],q[20];
rz(1.0) q[20];
cx q[0],q[20];
cx q[0],q[24];
rz(1.0) q[24];
cx q[0],q[24];
cx q[1],q[21];
rz(1.0) q[21];
cx q[1],q[21];
cx q[1],q[25];
rz(1.0) q[25];
cx q[1],q[25];
cx q[2],q[22];
rz(1.0) q[22];
cx q[2],q[22];
cx q[2],q[26];
rz(1.0) q[26];
cx q[2],q[26];
cx q[3],q[23];
rz(1.0) q[23];
cx q[3],q[23];
cx q[3],q[27];
rz(1.0) q[27];
cx q[3],q[27];
