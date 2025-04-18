OPENQASM 2.0;
include "qelib1.inc";

qreg q[64];
u3(0.0*pi,-0.5*pi,0.6600887264517992*pi) q[0];
u3(0.0*pi,-0.5*pi,1.085597795724301*pi) q[1];
u3(0.0*pi,-0.5*pi,1.2715909155711453*pi) q[2];
u3(0.0*pi,-0.5*pi,1.0377867201978201*pi) q[3];
u3(0.0*pi,-0.5*pi,0.8265033026802702*pi) q[4];
u3(0.0*pi,-0.5*pi,0.6377406630184949*pi) q[5];
u3(0.0*pi,-0.5*pi,3.7222943300574247*pi) q[6];
u3(0.0*pi,-0.5*pi,3.7361227386406943*pi) q[7];
cx q[0],q[9];
cx q[1],q[8];
u3(0.0*pi,-0.5*pi,0.7802976777121895*pi) q[8];
u3(0.0*pi,-0.5*pi,0.7802976777121895*pi) q[9];
cx q[0],q[9];
cx q[1],q[8];
cx q[0],q[10];
cx q[2],q[8];
u3(0.0*pi,-0.5*pi,3.5521380696713476*pi) q[8];
u3(0.0*pi,-0.5*pi,3.5521380696713476*pi) q[10];
cx q[0],q[10];
cx q[2],q[8];
cx q[0],q[11];
cx q[1],q[10];
cx q[2],q[9];
cx q[3],q[8];
u3(0.0*pi,-0.5*pi,4.05636431373644*pi) q[8];
u3(0.0*pi,-0.5*pi,4.498605644846842*pi) q[9];
u3(0.0*pi,-0.5*pi,4.498605644846842*pi) q[10];
u3(0.0*pi,-0.5*pi,4.05636431373644*pi) q[11];
cx q[0],q[11];
cx q[1],q[10];
cx q[2],q[9];
cx q[3],q[8];
cx q[0],q[12];
cx q[1],q[11];
cx q[3],q[9];
cx q[4],q[8];
u3(0.0*pi,-0.5*pi,0.7169376000957258*pi) q[8];
u3(0.0*pi,-0.5*pi,0.9169126524569245*pi) q[9];
u3(0.0*pi,-0.5*pi,0.9169126524569245*pi) q[11];
u3(0.0*pi,-0.5*pi,0.7169376000957258*pi) q[12];
cx q[0],q[12];
cx q[1],q[11];
cx q[3],q[9];
cx q[4],q[8];
cx q[0],q[13];
cx q[1],q[12];
cx q[2],q[11];
cx q[3],q[10];
cx q[4],q[9];
cx q[5],q[8];
u3(0.0*pi,-0.5*pi,1.063422446713467*pi) q[8];
u3(0.0*pi,-0.5*pi,0.7817112233567335*pi) q[9];
u3(0.0*pi,-0.5*pi,3.605641708758775*pi) q[10];
u3(0.0*pi,-0.5*pi,3.605641708758775*pi) q[11];
u3(0.0*pi,-0.5*pi,0.7817112233567335*pi) q[12];
u3(0.0*pi,-0.5*pi,1.063422446713467*pi) q[13];
cx q[0],q[13];
cx q[1],q[12];
cx q[2],q[11];
cx q[3],q[10];
cx q[4],q[9];
cx q[5],q[8];
cx q[0],q[14];
cx q[1],q[13];
cx q[2],q[12];
cx q[4],q[10];
cx q[5],q[9];
cx q[6],q[8];
u3(0.0*pi,-0.5*pi,1.2098938599872382*pi) q[8];
u3(0.0*pi,-0.5*pi,1.383126688050416*pi) q[9];
u3(0.0*pi,-0.5*pi,1.4056474659061906*pi) q[10];
u3(0.0*pi,-0.5*pi,1.4056474659061906*pi) q[12];
u3(0.0*pi,-0.5*pi,1.383126688050416*pi) q[13];
u3(0.0*pi,-0.5*pi,1.2098938599872382*pi) q[14];
cx q[0],q[14];
cx q[1],q[13];
cx q[2],q[12];
cx q[4],q[10];
cx q[5],q[9];
cx q[6],q[8];
cx q[0],q[15];
cx q[1],q[14];
cx q[2],q[13];
cx q[3],q[12];
cx q[4],q[11];
cx q[5],q[10];
cx q[6],q[9];
cx q[7],q[8];
u3(0.0*pi,-0.5*pi,1.0409016688576922*pi) q[8];
u3(0.0*pi,-0.5*pi,1.3549517276164655*pi) q[9];
u3(0.0*pi,-0.5*pi,4.377482100717933*pi) q[10];
u3(0.0*pi,-0.5*pi,3.797193058726867*pi) q[11];
u3(0.0*pi,-0.5*pi,3.797193058726867*pi) q[12];
u3(0.0*pi,-0.5*pi,4.377482100717933*pi) q[13];
u3(0.0*pi,-0.5*pi,1.3549517276164655*pi) q[14];
u3(0.0*pi,-0.5*pi,1.0409016688576922*pi) q[15];
cx q[0],q[15];
cx q[1],q[14];
cx q[2],q[13];
cx q[3],q[12];
cx q[4],q[11];
cx q[5],q[10];
cx q[6],q[9];
cx q[7],q[8];
cx q[0],q[57];
cx q[1],q[15];
cx q[2],q[14];
cx q[3],q[13];
cx q[5],q[11];
cx q[6],q[10];
cx q[7],q[9];
u3(0.0*pi,-0.5*pi,0.6600887264517992*pi) q[8];
cx q[8],q[17];
u3(0.0*pi,-0.5*pi,4.4915954880982785*pi) q[9];
u3(0.0*pi,-0.5*pi,4.476065676499681*pi) q[10];
u3(0.0*pi,-0.5*pi,3.6028338079610727*pi) q[11];
u3(0.0*pi,-0.5*pi,3.6028338079610727*pi) q[13];
u3(0.0*pi,-0.5*pi,4.476065676499681*pi) q[14];
u3(0.0*pi,-0.5*pi,4.4915954880982785*pi) q[15];
u3(0.0*pi,-0.5*pi,0.7802976777121895*pi) q[57];
cx q[0],q[57];
cx q[1],q[15];
cx q[2],q[14];
cx q[3],q[13];
cx q[5],q[11];
cx q[6],q[10];
cx q[7],q[9];
u3(0.0*pi,-0.5*pi,0.7802976777121895*pi) q[17];
cx q[0],q[58];
cx q[1],q[56];
cx q[2],q[15];
cx q[3],q[14];
cx q[4],q[13];
cx q[5],q[12];
cx q[6],q[11];
cx q[7],q[10];
cx q[8],q[17];
u3(0.0*pi,-0.5*pi,1.085597795724301*pi) q[9];
cx q[8],q[18];
cx q[9],q[16];
u3(0.0*pi,-0.5*pi,1.198623875813658*pi) q[10];
u3(0.0*pi,-0.5*pi,4.466209237970645*pi) q[11];
u3(0.0*pi,-0.5*pi,3.584515286056159*pi) q[12];
u3(0.0*pi,-0.5*pi,3.584515286056159*pi) q[13];
u3(0.0*pi,-0.5*pi,4.466209237970645*pi) q[14];
u3(0.0*pi,-0.5*pi,1.198623875813658*pi) q[15];
u3(0.0*pi,-0.5*pi,0.7802976777121895*pi) q[56];
u3(0.0*pi,-0.5*pi,3.5521380696713476*pi) q[58];
cx q[0],q[58];
cx q[1],q[56];
cx q[2],q[15];
cx q[3],q[14];
cx q[4],q[13];
cx q[5],q[12];
cx q[6],q[11];
cx q[7],q[10];
u3(0.0*pi,-0.5*pi,0.7802976777121895*pi) q[16];
u3(0.0*pi,-0.5*pi,3.5521380696713476*pi) q[18];
cx q[0],q[59];
cx q[1],q[58];
cx q[2],q[56];
cx q[3],q[15];
cx q[4],q[14];
cx q[6],q[12];
cx q[7],q[11];
cx q[8],q[18];
cx q[9],q[16];
u3(0.0*pi,-0.5*pi,1.2715909155711453*pi) q[10];
cx q[8],q[19];
cx q[9],q[18];
cx q[10],q[16];
u3(0.0*pi,-0.5*pi,0.7859518602903652*pi) q[11];
u3(0.0*pi,-0.5*pi,1.361981074856414*pi) q[12];
u3(0.0*pi,-0.5*pi,1.361981074856414*pi) q[14];
u3(0.0*pi,-0.5*pi,0.7859518602903652*pi) q[15];
u3(0.0*pi,-0.5*pi,3.5521380696713476*pi) q[56];
u3(0.0*pi,-0.5*pi,4.498605644846842*pi) q[58];
u3(0.0*pi,-0.5*pi,4.05636431373644*pi) q[59];
cx q[0],q[59];
cx q[1],q[58];
cx q[2],q[56];
cx q[3],q[15];
cx q[4],q[14];
cx q[6],q[12];
cx q[7],q[11];
u3(0.0*pi,-0.5*pi,3.5521380696713476*pi) q[16];
u3(0.0*pi,-0.5*pi,4.498605644846842*pi) q[18];
u3(0.0*pi,-0.5*pi,4.05636431373644*pi) q[19];
cx q[0],q[60];
cx q[1],q[59];
cx q[2],q[57];
cx q[3],q[56];
cx q[4],q[15];
cx q[5],q[14];
cx q[6],q[13];
cx q[7],q[12];
cx q[8],q[19];
cx q[9],q[18];
cx q[10],q[16];
u3(0.0*pi,-0.5*pi,1.0377867201978201*pi) q[11];
cx q[8],q[20];
cx q[9],q[19];
cx q[10],q[17];
cx q[11],q[16];
u3(0.0*pi,-0.5*pi,0.6887626396617753*pi) q[12];
u3(0.0*pi,-0.5*pi,3.8816987495373327*pi) q[13];
u3(0.0*pi,-0.5*pi,3.8816987495373327*pi) q[14];
u3(0.0*pi,-0.5*pi,0.6887626396617753*pi) q[15];
u3(0.0*pi,-0.5*pi,4.05636431373644*pi) q[56];
u3(0.0*pi,-0.5*pi,4.498605644846842*pi) q[57];
u3(0.0*pi,-0.5*pi,0.9169126524569245*pi) q[59];
u3(0.0*pi,-0.5*pi,0.7169376000957258*pi) q[60];
cx q[0],q[60];
cx q[1],q[59];
cx q[2],q[57];
cx q[3],q[56];
cx q[4],q[15];
cx q[5],q[14];
cx q[6],q[13];
cx q[7],q[12];
u3(0.0*pi,-0.5*pi,4.05636431373644*pi) q[16];
u3(0.0*pi,-0.5*pi,4.498605644846842*pi) q[17];
u3(0.0*pi,-0.5*pi,0.9169126524569245*pi) q[19];
u3(0.0*pi,-0.5*pi,0.7169376000957258*pi) q[20];
cx q[0],q[61];
cx q[1],q[60];
cx q[2],q[59];
cx q[3],q[57];
cx q[4],q[56];
cx q[5],q[15];
cx q[7],q[13];
cx q[8],q[20];
cx q[9],q[19];
cx q[10],q[17];
cx q[11],q[16];
u3(0.0*pi,-0.5*pi,0.8265033026802702*pi) q[12];
cx q[8],q[21];
cx q[9],q[20];
cx q[10],q[19];
cx q[11],q[17];
cx q[12],q[16];
u3(0.0*pi,-0.5*pi,0.5380505894543723*pi) q[13];
u3(0.0*pi,-0.5*pi,0.5380505894543723*pi) q[15];
u3(0.0*pi,-0.5*pi,0.7169376000957258*pi) q[56];
u3(0.0*pi,-0.5*pi,0.9169126524569245*pi) q[57];
u3(0.0*pi,-0.5*pi,3.605641708758775*pi) q[59];
u3(0.0*pi,-0.5*pi,0.7817112233567335*pi) q[60];
u3(0.0*pi,-0.5*pi,1.063422446713467*pi) q[61];
cx q[0],q[61];
cx q[1],q[60];
cx q[2],q[59];
cx q[3],q[57];
cx q[4],q[56];
cx q[5],q[15];
cx q[7],q[13];
u3(0.0*pi,-0.5*pi,0.7169376000957258*pi) q[16];
u3(0.0*pi,-0.5*pi,0.9169126524569245*pi) q[17];
u3(0.0*pi,-0.5*pi,3.605641708758775*pi) q[19];
u3(0.0*pi,-0.5*pi,0.7817112233567335*pi) q[20];
u3(0.0*pi,-0.5*pi,1.063422446713467*pi) q[21];
cx q[0],q[62];
cx q[1],q[61];
cx q[2],q[60];
cx q[3],q[58];
cx q[4],q[57];
cx q[5],q[56];
cx q[6],q[15];
cx q[7],q[14];
cx q[8],q[21];
cx q[9],q[20];
cx q[10],q[19];
cx q[11],q[17];
cx q[12],q[16];
u3(0.0*pi,-0.5*pi,0.6377406630184949*pi) q[13];
cx q[8],q[22];
cx q[9],q[21];
cx q[10],q[20];
cx q[11],q[18];
cx q[12],q[17];
cx q[13],q[16];
u3(0.0*pi,-0.5*pi,1.138052508503511*pi) q[14];
u3(0.0*pi,-0.5*pi,1.138052508503511*pi) q[15];
u3(0.0*pi,-0.5*pi,1.063422446713467*pi) q[56];
u3(0.0*pi,-0.5*pi,0.7817112233567335*pi) q[57];
u3(0.0*pi,-0.5*pi,3.605641708758775*pi) q[58];
u3(0.0*pi,-0.5*pi,1.4056474659061906*pi) q[60];
u3(0.0*pi,-0.5*pi,1.383126688050416*pi) q[61];
u3(0.0*pi,-0.5*pi,1.2098938599872382*pi) q[62];
cx q[0],q[62];
cx q[1],q[61];
cx q[2],q[60];
cx q[3],q[58];
cx q[4],q[57];
cx q[5],q[56];
cx q[6],q[15];
cx q[7],q[14];
u3(0.0*pi,-0.5*pi,1.063422446713467*pi) q[16];
u3(0.0*pi,-0.5*pi,0.7817112233567335*pi) q[17];
u3(0.0*pi,-0.5*pi,3.605641708758775*pi) q[18];
u3(0.0*pi,-0.5*pi,1.4056474659061906*pi) q[20];
u3(0.0*pi,-0.5*pi,1.383126688050416*pi) q[21];
u3(0.0*pi,-0.5*pi,1.2098938599872382*pi) q[22];
cx q[0],q[63];
cx q[1],q[62];
cx q[2],q[61];
cx q[3],q[60];
cx q[4],q[58];
cx q[5],q[57];
cx q[6],q[56];
cx q[8],q[22];
cx q[9],q[21];
cx q[10],q[20];
cx q[11],q[18];
cx q[12],q[17];
cx q[13],q[16];
u3(0.0*pi,-0.5*pi,3.7222943300574247*pi) q[14];
u3(0.0*pi,-0.5*pi,3.7361227386406943*pi) q[15];
cx q[8],q[23];
cx q[9],q[22];
cx q[10],q[21];
cx q[11],q[20];
cx q[12],q[18];
cx q[13],q[17];
cx q[14],q[16];
u3(0.0*pi,-0.5*pi,1.2098938599872382*pi) q[56];
u3(0.0*pi,-0.5*pi,1.383126688050416*pi) q[57];
u3(0.0*pi,-0.5*pi,1.4056474659061906*pi) q[58];
u3(0.0*pi,-0.5*pi,3.797193058726867*pi) q[60];
u3(0.0*pi,-0.5*pi,4.377482100717933*pi) q[61];
u3(0.0*pi,-0.5*pi,1.3549517276164655*pi) q[62];
u3(0.0*pi,-0.5*pi,1.0409016688576922*pi) q[63];
cx q[0],q[63];
cx q[1],q[62];
cx q[2],q[61];
cx q[3],q[60];
cx q[4],q[58];
cx q[5],q[57];
cx q[6],q[56];
u3(0.0*pi,-0.5*pi,1.2098938599872382*pi) q[16];
u3(0.0*pi,-0.5*pi,1.383126688050416*pi) q[17];
u3(0.0*pi,-0.5*pi,1.4056474659061906*pi) q[18];
u3(0.0*pi,-0.5*pi,3.797193058726867*pi) q[20];
u3(0.0*pi,-0.5*pi,4.377482100717933*pi) q[21];
u3(0.0*pi,-0.5*pi,1.3549517276164655*pi) q[22];
u3(0.0*pi,-0.5*pi,1.0409016688576922*pi) q[23];
cx q[1],q[63];
cx q[2],q[62];
cx q[3],q[61];
cx q[4],q[59];
cx q[5],q[58];
cx q[6],q[57];
cx q[7],q[56];
cx q[8],q[23];
cx q[9],q[22];
cx q[10],q[21];
cx q[11],q[20];
cx q[12],q[18];
cx q[13],q[17];
cx q[14],q[16];
cx q[9],q[23];
cx q[10],q[22];
cx q[11],q[21];
cx q[12],q[19];
cx q[13],q[18];
cx q[14],q[17];
cx q[15],q[16];
u3(0.0*pi,-0.5*pi,1.0409016688576922*pi) q[56];
u3(0.0*pi,-0.5*pi,1.3549517276164655*pi) q[57];
u3(0.0*pi,-0.5*pi,4.377482100717933*pi) q[58];
u3(0.0*pi,-0.5*pi,3.797193058726867*pi) q[59];
u3(0.0*pi,-0.5*pi,3.6028338079610727*pi) q[61];
u3(0.0*pi,-0.5*pi,4.476065676499681*pi) q[62];
u3(0.0*pi,-0.5*pi,4.4915954880982785*pi) q[63];
cx q[1],q[63];
cx q[2],q[62];
cx q[3],q[61];
cx q[4],q[59];
cx q[5],q[58];
cx q[6],q[57];
cx q[7],q[56];
u3(0.0*pi,-0.5*pi,1.0409016688576922*pi) q[16];
u3(0.0*pi,-0.5*pi,1.3549517276164655*pi) q[17];
u3(0.0*pi,-0.5*pi,4.377482100717933*pi) q[18];
u3(0.0*pi,-0.5*pi,3.797193058726867*pi) q[19];
u3(0.0*pi,-0.5*pi,3.6028338079610727*pi) q[21];
u3(0.0*pi,-0.5*pi,4.476065676499681*pi) q[22];
u3(0.0*pi,-0.5*pi,4.4915954880982785*pi) q[23];
cx q[2],q[63];
cx q[3],q[62];
cx q[4],q[61];
cx q[5],q[59];
cx q[6],q[58];
cx q[7],q[57];
cx q[9],q[23];
cx q[10],q[22];
cx q[11],q[21];
cx q[12],q[19];
cx q[13],q[18];
cx q[14],q[17];
cx q[15],q[16];
cx q[10],q[23];
cx q[11],q[22];
cx q[12],q[21];
cx q[13],q[19];
cx q[14],q[18];
cx q[15],q[17];
u3(0.0*pi,-0.5*pi,0.6600887264517992*pi) q[16];
u3(0.0*pi,-0.5*pi,4.4915954880982785*pi) q[57];
u3(0.0*pi,-0.5*pi,4.476065676499681*pi) q[58];
u3(0.0*pi,-0.5*pi,3.6028338079610727*pi) q[59];
u3(0.0*pi,-0.5*pi,3.584515286056159*pi) q[61];
u3(0.0*pi,-0.5*pi,4.466209237970645*pi) q[62];
u3(0.0*pi,-0.5*pi,1.198623875813658*pi) q[63];
cx q[2],q[63];
cx q[3],q[62];
cx q[4],q[61];
cx q[5],q[59];
cx q[6],q[58];
cx q[7],q[57];
cx q[16],q[25];
u3(0.0*pi,-0.5*pi,4.4915954880982785*pi) q[17];
u3(0.0*pi,-0.5*pi,4.476065676499681*pi) q[18];
u3(0.0*pi,-0.5*pi,3.6028338079610727*pi) q[19];
u3(0.0*pi,-0.5*pi,3.584515286056159*pi) q[21];
u3(0.0*pi,-0.5*pi,4.466209237970645*pi) q[22];
u3(0.0*pi,-0.5*pi,1.198623875813658*pi) q[23];
cx q[3],q[63];
cx q[4],q[62];
cx q[5],q[60];
cx q[6],q[59];
cx q[7],q[58];
cx q[10],q[23];
cx q[11],q[22];
cx q[12],q[21];
cx q[13],q[19];
cx q[14],q[18];
cx q[15],q[17];
u3(0.0*pi,-0.5*pi,0.7802976777121895*pi) q[25];
cx q[11],q[23];
cx q[12],q[22];
cx q[13],q[20];
cx q[14],q[19];
cx q[15],q[18];
cx q[16],q[25];
u3(0.0*pi,-0.5*pi,1.085597795724301*pi) q[17];
u3(0.0*pi,-0.5*pi,1.198623875813658*pi) q[58];
u3(0.0*pi,-0.5*pi,4.466209237970645*pi) q[59];
u3(0.0*pi,-0.5*pi,3.584515286056159*pi) q[60];
u3(0.0*pi,-0.5*pi,1.361981074856414*pi) q[62];
u3(0.0*pi,-0.5*pi,0.7859518602903652*pi) q[63];
cx q[3],q[63];
cx q[4],q[62];
cx q[5],q[60];
cx q[6],q[59];
cx q[7],q[58];
cx q[16],q[26];
cx q[17],q[24];
u3(0.0*pi,-0.5*pi,1.198623875813658*pi) q[18];
u3(0.0*pi,-0.5*pi,4.466209237970645*pi) q[19];
u3(0.0*pi,-0.5*pi,3.584515286056159*pi) q[20];
u3(0.0*pi,-0.5*pi,1.361981074856414*pi) q[22];
u3(0.0*pi,-0.5*pi,0.7859518602903652*pi) q[23];
cx q[4],q[63];
cx q[5],q[62];
cx q[6],q[60];
cx q[7],q[59];
cx q[11],q[23];
cx q[12],q[22];
cx q[13],q[20];
cx q[14],q[19];
cx q[15],q[18];
u3(0.0*pi,-0.5*pi,0.7802976777121895*pi) q[24];
u3(0.0*pi,-0.5*pi,3.5521380696713476*pi) q[26];
cx q[12],q[23];
cx q[13],q[22];
cx q[14],q[20];
cx q[15],q[19];
cx q[16],q[26];
cx q[17],q[24];
u3(0.0*pi,-0.5*pi,1.2715909155711453*pi) q[18];
u3(0.0*pi,-0.5*pi,0.7859518602903652*pi) q[59];
u3(0.0*pi,-0.5*pi,1.361981074856414*pi) q[60];
u3(0.0*pi,-0.5*pi,3.8816987495373327*pi) q[62];
u3(0.0*pi,-0.5*pi,0.6887626396617753*pi) q[63];
cx q[4],q[63];
cx q[5],q[62];
cx q[6],q[60];
cx q[7],q[59];
cx q[16],q[27];
cx q[17],q[26];
cx q[18],q[24];
u3(0.0*pi,-0.5*pi,0.7859518602903652*pi) q[19];
u3(0.0*pi,-0.5*pi,1.361981074856414*pi) q[20];
u3(0.0*pi,-0.5*pi,3.8816987495373327*pi) q[22];
u3(0.0*pi,-0.5*pi,0.6887626396617753*pi) q[23];
cx q[5],q[63];
cx q[6],q[61];
cx q[7],q[60];
cx q[12],q[23];
cx q[13],q[22];
cx q[14],q[20];
cx q[15],q[19];
u3(0.0*pi,-0.5*pi,3.5521380696713476*pi) q[24];
u3(0.0*pi,-0.5*pi,4.498605644846842*pi) q[26];
u3(0.0*pi,-0.5*pi,4.05636431373644*pi) q[27];
cx q[13],q[23];
cx q[14],q[21];
cx q[15],q[20];
cx q[16],q[27];
cx q[17],q[26];
cx q[18],q[24];
u3(0.0*pi,-0.5*pi,1.0377867201978201*pi) q[19];
u3(0.0*pi,-0.5*pi,0.6887626396617753*pi) q[60];
u3(0.0*pi,-0.5*pi,3.8816987495373327*pi) q[61];
u3(0.0*pi,-0.5*pi,0.5380505894543723*pi) q[63];
cx q[5],q[63];
cx q[6],q[61];
cx q[7],q[60];
cx q[16],q[28];
cx q[17],q[27];
cx q[18],q[25];
cx q[19],q[24];
u3(0.0*pi,-0.5*pi,0.6887626396617753*pi) q[20];
u3(0.0*pi,-0.5*pi,3.8816987495373327*pi) q[21];
u3(0.0*pi,-0.5*pi,0.5380505894543723*pi) q[23];
cx q[6],q[63];
cx q[7],q[61];
cx q[13],q[23];
cx q[14],q[21];
cx q[15],q[20];
u3(0.0*pi,-0.5*pi,4.05636431373644*pi) q[24];
u3(0.0*pi,-0.5*pi,4.498605644846842*pi) q[25];
u3(0.0*pi,-0.5*pi,0.9169126524569245*pi) q[27];
u3(0.0*pi,-0.5*pi,0.7169376000957258*pi) q[28];
cx q[14],q[23];
cx q[15],q[21];
cx q[16],q[28];
cx q[17],q[27];
cx q[18],q[25];
cx q[19],q[24];
u3(0.0*pi,-0.5*pi,0.8265033026802702*pi) q[20];
u3(0.0*pi,-0.5*pi,0.5380505894543723*pi) q[61];
u3(0.0*pi,-0.5*pi,1.138052508503511*pi) q[63];
cx q[6],q[63];
cx q[7],q[61];
cx q[16],q[29];
cx q[17],q[28];
cx q[18],q[27];
cx q[19],q[25];
cx q[20],q[24];
u3(0.0*pi,-0.5*pi,0.5380505894543723*pi) q[21];
u3(0.0*pi,-0.5*pi,1.138052508503511*pi) q[23];
cx q[7],q[62];
cx q[14],q[23];
cx q[15],q[21];
u3(0.0*pi,-0.5*pi,0.7169376000957258*pi) q[24];
u3(0.0*pi,-0.5*pi,0.9169126524569245*pi) q[25];
u3(0.0*pi,-0.5*pi,3.605641708758775*pi) q[27];
u3(0.0*pi,-0.5*pi,0.7817112233567335*pi) q[28];
u3(0.0*pi,-0.5*pi,1.063422446713467*pi) q[29];
cx q[15],q[22];
cx q[16],q[29];
cx q[17],q[28];
cx q[18],q[27];
cx q[19],q[25];
cx q[20],q[24];
u3(0.0*pi,-0.5*pi,0.6377406630184949*pi) q[21];
u3(0.0*pi,-0.5*pi,3.7361227386406943*pi) q[23];
u3(0.0*pi,-0.5*pi,1.138052508503511*pi) q[62];
cx q[7],q[62];
cx q[16],q[30];
cx q[17],q[29];
cx q[18],q[28];
cx q[19],q[26];
cx q[20],q[25];
cx q[21],q[24];
u3(0.0*pi,-0.5*pi,1.138052508503511*pi) q[22];
cx q[15],q[22];
u3(0.0*pi,-0.5*pi,1.063422446713467*pi) q[24];
u3(0.0*pi,-0.5*pi,0.7817112233567335*pi) q[25];
u3(0.0*pi,-0.5*pi,3.605641708758775*pi) q[26];
u3(0.0*pi,-0.5*pi,1.4056474659061906*pi) q[28];
u3(0.0*pi,-0.5*pi,1.383126688050416*pi) q[29];
u3(0.0*pi,-0.5*pi,1.2098938599872382*pi) q[30];
cx q[16],q[30];
cx q[17],q[29];
cx q[18],q[28];
cx q[19],q[26];
cx q[20],q[25];
cx q[21],q[24];
u3(0.0*pi,-0.5*pi,3.7222943300574247*pi) q[22];
cx q[16],q[31];
cx q[17],q[30];
cx q[18],q[29];
cx q[19],q[28];
cx q[20],q[26];
cx q[21],q[25];
cx q[22],q[24];
u3(0.0*pi,-0.5*pi,1.2098938599872382*pi) q[24];
u3(0.0*pi,-0.5*pi,1.383126688050416*pi) q[25];
u3(0.0*pi,-0.5*pi,1.4056474659061906*pi) q[26];
u3(0.0*pi,-0.5*pi,3.797193058726867*pi) q[28];
u3(0.0*pi,-0.5*pi,4.377482100717933*pi) q[29];
u3(0.0*pi,-0.5*pi,1.3549517276164655*pi) q[30];
u3(0.0*pi,-0.5*pi,1.0409016688576922*pi) q[31];
cx q[16],q[31];
cx q[17],q[30];
cx q[18],q[29];
cx q[19],q[28];
cx q[20],q[26];
cx q[21],q[25];
cx q[22],q[24];
cx q[17],q[31];
cx q[18],q[30];
cx q[19],q[29];
cx q[20],q[27];
cx q[21],q[26];
cx q[22],q[25];
cx q[23],q[24];
u3(0.0*pi,-0.5*pi,1.0409016688576922*pi) q[24];
u3(0.0*pi,-0.5*pi,1.3549517276164655*pi) q[25];
u3(0.0*pi,-0.5*pi,4.377482100717933*pi) q[26];
u3(0.0*pi,-0.5*pi,3.797193058726867*pi) q[27];
u3(0.0*pi,-0.5*pi,3.6028338079610727*pi) q[29];
u3(0.0*pi,-0.5*pi,4.476065676499681*pi) q[30];
u3(0.0*pi,-0.5*pi,4.4915954880982785*pi) q[31];
cx q[17],q[31];
cx q[18],q[30];
cx q[19],q[29];
cx q[20],q[27];
cx q[21],q[26];
cx q[22],q[25];
cx q[23],q[24];
cx q[18],q[31];
cx q[19],q[30];
cx q[20],q[29];
cx q[21],q[27];
cx q[22],q[26];
cx q[23],q[25];
u3(0.0*pi,-0.5*pi,0.6600887264517992*pi) q[24];
cx q[24],q[33];
u3(0.0*pi,-0.5*pi,4.4915954880982785*pi) q[25];
u3(0.0*pi,-0.5*pi,4.476065676499681*pi) q[26];
u3(0.0*pi,-0.5*pi,3.6028338079610727*pi) q[27];
u3(0.0*pi,-0.5*pi,3.584515286056159*pi) q[29];
u3(0.0*pi,-0.5*pi,4.466209237970645*pi) q[30];
u3(0.0*pi,-0.5*pi,1.198623875813658*pi) q[31];
cx q[18],q[31];
cx q[19],q[30];
cx q[20],q[29];
cx q[21],q[27];
cx q[22],q[26];
cx q[23],q[25];
u3(0.0*pi,-0.5*pi,0.7802976777121895*pi) q[33];
cx q[19],q[31];
cx q[20],q[30];
cx q[21],q[28];
cx q[22],q[27];
cx q[23],q[26];
cx q[24],q[33];
u3(0.0*pi,-0.5*pi,1.085597795724301*pi) q[25];
cx q[24],q[34];
cx q[25],q[32];
u3(0.0*pi,-0.5*pi,1.198623875813658*pi) q[26];
u3(0.0*pi,-0.5*pi,4.466209237970645*pi) q[27];
u3(0.0*pi,-0.5*pi,3.584515286056159*pi) q[28];
u3(0.0*pi,-0.5*pi,1.361981074856414*pi) q[30];
u3(0.0*pi,-0.5*pi,0.7859518602903652*pi) q[31];
cx q[19],q[31];
cx q[20],q[30];
cx q[21],q[28];
cx q[22],q[27];
cx q[23],q[26];
u3(0.0*pi,-0.5*pi,0.7802976777121895*pi) q[32];
u3(0.0*pi,-0.5*pi,3.5521380696713476*pi) q[34];
cx q[20],q[31];
cx q[21],q[30];
cx q[22],q[28];
cx q[23],q[27];
cx q[24],q[34];
cx q[25],q[32];
u3(0.0*pi,-0.5*pi,1.2715909155711453*pi) q[26];
cx q[24],q[35];
cx q[25],q[34];
cx q[26],q[32];
u3(0.0*pi,-0.5*pi,0.7859518602903652*pi) q[27];
u3(0.0*pi,-0.5*pi,1.361981074856414*pi) q[28];
u3(0.0*pi,-0.5*pi,3.8816987495373327*pi) q[30];
u3(0.0*pi,-0.5*pi,0.6887626396617753*pi) q[31];
cx q[20],q[31];
cx q[21],q[30];
cx q[22],q[28];
cx q[23],q[27];
u3(0.0*pi,-0.5*pi,3.5521380696713476*pi) q[32];
u3(0.0*pi,-0.5*pi,4.498605644846842*pi) q[34];
u3(0.0*pi,-0.5*pi,4.05636431373644*pi) q[35];
cx q[21],q[31];
cx q[22],q[29];
cx q[23],q[28];
cx q[24],q[35];
cx q[25],q[34];
cx q[26],q[32];
u3(0.0*pi,-0.5*pi,1.0377867201978201*pi) q[27];
cx q[24],q[36];
cx q[25],q[35];
cx q[26],q[33];
cx q[27],q[32];
u3(0.0*pi,-0.5*pi,0.6887626396617753*pi) q[28];
u3(0.0*pi,-0.5*pi,3.8816987495373327*pi) q[29];
u3(0.0*pi,-0.5*pi,0.5380505894543723*pi) q[31];
cx q[21],q[31];
cx q[22],q[29];
cx q[23],q[28];
u3(0.0*pi,-0.5*pi,4.05636431373644*pi) q[32];
u3(0.0*pi,-0.5*pi,4.498605644846842*pi) q[33];
u3(0.0*pi,-0.5*pi,0.9169126524569245*pi) q[35];
u3(0.0*pi,-0.5*pi,0.7169376000957258*pi) q[36];
cx q[22],q[31];
cx q[23],q[29];
cx q[24],q[36];
cx q[25],q[35];
cx q[26],q[33];
cx q[27],q[32];
u3(0.0*pi,-0.5*pi,0.8265033026802702*pi) q[28];
cx q[24],q[37];
cx q[25],q[36];
cx q[26],q[35];
cx q[27],q[33];
cx q[28],q[32];
u3(0.0*pi,-0.5*pi,0.5380505894543723*pi) q[29];
u3(0.0*pi,-0.5*pi,1.138052508503511*pi) q[31];
cx q[22],q[31];
cx q[23],q[29];
u3(0.0*pi,-0.5*pi,0.7169376000957258*pi) q[32];
u3(0.0*pi,-0.5*pi,0.9169126524569245*pi) q[33];
u3(0.0*pi,-0.5*pi,3.605641708758775*pi) q[35];
u3(0.0*pi,-0.5*pi,0.7817112233567335*pi) q[36];
u3(0.0*pi,-0.5*pi,1.063422446713467*pi) q[37];
cx q[23],q[30];
cx q[24],q[37];
cx q[25],q[36];
cx q[26],q[35];
cx q[27],q[33];
cx q[28],q[32];
u3(0.0*pi,-0.5*pi,0.6377406630184949*pi) q[29];
u3(0.0*pi,-0.5*pi,3.7361227386406943*pi) q[31];
cx q[24],q[38];
cx q[25],q[37];
cx q[26],q[36];
cx q[27],q[34];
cx q[28],q[33];
cx q[29],q[32];
u3(0.0*pi,-0.5*pi,1.138052508503511*pi) q[30];
cx q[23],q[30];
u3(0.0*pi,-0.5*pi,1.063422446713467*pi) q[32];
u3(0.0*pi,-0.5*pi,0.7817112233567335*pi) q[33];
u3(0.0*pi,-0.5*pi,3.605641708758775*pi) q[34];
u3(0.0*pi,-0.5*pi,1.4056474659061906*pi) q[36];
u3(0.0*pi,-0.5*pi,1.383126688050416*pi) q[37];
u3(0.0*pi,-0.5*pi,1.2098938599872382*pi) q[38];
cx q[24],q[38];
cx q[25],q[37];
cx q[26],q[36];
cx q[27],q[34];
cx q[28],q[33];
cx q[29],q[32];
u3(0.0*pi,-0.5*pi,3.7222943300574247*pi) q[30];
cx q[24],q[39];
cx q[25],q[38];
cx q[26],q[37];
cx q[27],q[36];
cx q[28],q[34];
cx q[29],q[33];
cx q[30],q[32];
u3(0.0*pi,-0.5*pi,1.2098938599872382*pi) q[32];
u3(0.0*pi,-0.5*pi,1.383126688050416*pi) q[33];
u3(0.0*pi,-0.5*pi,1.4056474659061906*pi) q[34];
u3(0.0*pi,-0.5*pi,3.797193058726867*pi) q[36];
u3(0.0*pi,-0.5*pi,4.377482100717933*pi) q[37];
u3(0.0*pi,-0.5*pi,1.3549517276164655*pi) q[38];
u3(0.0*pi,-0.5*pi,1.0409016688576922*pi) q[39];
cx q[24],q[39];
cx q[25],q[38];
cx q[26],q[37];
cx q[27],q[36];
cx q[28],q[34];
cx q[29],q[33];
cx q[30],q[32];
cx q[25],q[39];
cx q[26],q[38];
cx q[27],q[37];
cx q[28],q[35];
cx q[29],q[34];
cx q[30],q[33];
cx q[31],q[32];
u3(0.0*pi,-0.5*pi,1.0409016688576922*pi) q[32];
u3(0.0*pi,-0.5*pi,1.3549517276164655*pi) q[33];
u3(0.0*pi,-0.5*pi,4.377482100717933*pi) q[34];
u3(0.0*pi,-0.5*pi,3.797193058726867*pi) q[35];
u3(0.0*pi,-0.5*pi,3.6028338079610727*pi) q[37];
u3(0.0*pi,-0.5*pi,4.476065676499681*pi) q[38];
u3(0.0*pi,-0.5*pi,4.4915954880982785*pi) q[39];
cx q[25],q[39];
cx q[26],q[38];
cx q[27],q[37];
cx q[28],q[35];
cx q[29],q[34];
cx q[30],q[33];
cx q[31],q[32];
cx q[26],q[39];
cx q[27],q[38];
cx q[28],q[37];
cx q[29],q[35];
cx q[30],q[34];
cx q[31],q[33];
u3(0.0*pi,-0.5*pi,0.6600887264517992*pi) q[32];
cx q[32],q[41];
u3(0.0*pi,-0.5*pi,4.4915954880982785*pi) q[33];
u3(0.0*pi,-0.5*pi,4.476065676499681*pi) q[34];
u3(0.0*pi,-0.5*pi,3.6028338079610727*pi) q[35];
u3(0.0*pi,-0.5*pi,3.584515286056159*pi) q[37];
u3(0.0*pi,-0.5*pi,4.466209237970645*pi) q[38];
u3(0.0*pi,-0.5*pi,1.198623875813658*pi) q[39];
cx q[26],q[39];
cx q[27],q[38];
cx q[28],q[37];
cx q[29],q[35];
cx q[30],q[34];
cx q[31],q[33];
u3(0.0*pi,-0.5*pi,0.7802976777121895*pi) q[41];
cx q[27],q[39];
cx q[28],q[38];
cx q[29],q[36];
cx q[30],q[35];
cx q[31],q[34];
cx q[32],q[41];
u3(0.0*pi,-0.5*pi,1.085597795724301*pi) q[33];
cx q[32],q[42];
cx q[33],q[40];
u3(0.0*pi,-0.5*pi,1.198623875813658*pi) q[34];
u3(0.0*pi,-0.5*pi,4.466209237970645*pi) q[35];
u3(0.0*pi,-0.5*pi,3.584515286056159*pi) q[36];
u3(0.0*pi,-0.5*pi,1.361981074856414*pi) q[38];
u3(0.0*pi,-0.5*pi,0.7859518602903652*pi) q[39];
cx q[27],q[39];
cx q[28],q[38];
cx q[29],q[36];
cx q[30],q[35];
cx q[31],q[34];
u3(0.0*pi,-0.5*pi,0.7802976777121895*pi) q[40];
u3(0.0*pi,-0.5*pi,3.5521380696713476*pi) q[42];
cx q[28],q[39];
cx q[29],q[38];
cx q[30],q[36];
cx q[31],q[35];
cx q[32],q[42];
cx q[33],q[40];
u3(0.0*pi,-0.5*pi,1.2715909155711453*pi) q[34];
cx q[32],q[43];
cx q[33],q[42];
cx q[34],q[40];
u3(0.0*pi,-0.5*pi,0.7859518602903652*pi) q[35];
u3(0.0*pi,-0.5*pi,1.361981074856414*pi) q[36];
u3(0.0*pi,-0.5*pi,3.8816987495373327*pi) q[38];
u3(0.0*pi,-0.5*pi,0.6887626396617753*pi) q[39];
cx q[28],q[39];
cx q[29],q[38];
cx q[30],q[36];
cx q[31],q[35];
u3(0.0*pi,-0.5*pi,3.5521380696713476*pi) q[40];
u3(0.0*pi,-0.5*pi,4.498605644846842*pi) q[42];
u3(0.0*pi,-0.5*pi,4.05636431373644*pi) q[43];
cx q[29],q[39];
cx q[30],q[37];
cx q[31],q[36];
cx q[32],q[43];
cx q[33],q[42];
cx q[34],q[40];
u3(0.0*pi,-0.5*pi,1.0377867201978201*pi) q[35];
cx q[32],q[44];
cx q[33],q[43];
cx q[34],q[41];
cx q[35],q[40];
u3(0.0*pi,-0.5*pi,0.6887626396617753*pi) q[36];
u3(0.0*pi,-0.5*pi,3.8816987495373327*pi) q[37];
u3(0.0*pi,-0.5*pi,0.5380505894543723*pi) q[39];
cx q[29],q[39];
cx q[30],q[37];
cx q[31],q[36];
u3(0.0*pi,-0.5*pi,4.05636431373644*pi) q[40];
u3(0.0*pi,-0.5*pi,4.498605644846842*pi) q[41];
u3(0.0*pi,-0.5*pi,0.9169126524569245*pi) q[43];
u3(0.0*pi,-0.5*pi,0.7169376000957258*pi) q[44];
cx q[30],q[39];
cx q[31],q[37];
cx q[32],q[44];
cx q[33],q[43];
cx q[34],q[41];
cx q[35],q[40];
u3(0.0*pi,-0.5*pi,0.8265033026802702*pi) q[36];
cx q[32],q[45];
cx q[33],q[44];
cx q[34],q[43];
cx q[35],q[41];
cx q[36],q[40];
u3(0.0*pi,-0.5*pi,0.5380505894543723*pi) q[37];
u3(0.0*pi,-0.5*pi,1.138052508503511*pi) q[39];
cx q[30],q[39];
cx q[31],q[37];
u3(0.0*pi,-0.5*pi,0.7169376000957258*pi) q[40];
u3(0.0*pi,-0.5*pi,0.9169126524569245*pi) q[41];
u3(0.0*pi,-0.5*pi,3.605641708758775*pi) q[43];
u3(0.0*pi,-0.5*pi,0.7817112233567335*pi) q[44];
u3(0.0*pi,-0.5*pi,1.063422446713467*pi) q[45];
cx q[31],q[38];
cx q[32],q[45];
cx q[33],q[44];
cx q[34],q[43];
cx q[35],q[41];
cx q[36],q[40];
u3(0.0*pi,-0.5*pi,0.6377406630184949*pi) q[37];
u3(0.0*pi,-0.5*pi,3.7361227386406943*pi) q[39];
cx q[32],q[46];
cx q[33],q[45];
cx q[34],q[44];
cx q[35],q[42];
cx q[36],q[41];
cx q[37],q[40];
u3(0.0*pi,-0.5*pi,1.138052508503511*pi) q[38];
cx q[31],q[38];
u3(0.0*pi,-0.5*pi,1.063422446713467*pi) q[40];
u3(0.0*pi,-0.5*pi,0.7817112233567335*pi) q[41];
u3(0.0*pi,-0.5*pi,3.605641708758775*pi) q[42];
u3(0.0*pi,-0.5*pi,1.4056474659061906*pi) q[44];
u3(0.0*pi,-0.5*pi,1.383126688050416*pi) q[45];
u3(0.0*pi,-0.5*pi,1.2098938599872382*pi) q[46];
cx q[32],q[46];
cx q[33],q[45];
cx q[34],q[44];
cx q[35],q[42];
cx q[36],q[41];
cx q[37],q[40];
u3(0.0*pi,-0.5*pi,3.7222943300574247*pi) q[38];
cx q[32],q[47];
cx q[33],q[46];
cx q[34],q[45];
cx q[35],q[44];
cx q[36],q[42];
cx q[37],q[41];
cx q[38],q[40];
u3(0.0*pi,-0.5*pi,1.2098938599872382*pi) q[40];
u3(0.0*pi,-0.5*pi,1.383126688050416*pi) q[41];
u3(0.0*pi,-0.5*pi,1.4056474659061906*pi) q[42];
u3(0.0*pi,-0.5*pi,3.797193058726867*pi) q[44];
u3(0.0*pi,-0.5*pi,4.377482100717933*pi) q[45];
u3(0.0*pi,-0.5*pi,1.3549517276164655*pi) q[46];
u3(0.0*pi,-0.5*pi,1.0409016688576922*pi) q[47];
cx q[32],q[47];
cx q[33],q[46];
cx q[34],q[45];
cx q[35],q[44];
cx q[36],q[42];
cx q[37],q[41];
cx q[38],q[40];
cx q[33],q[47];
cx q[34],q[46];
cx q[35],q[45];
cx q[36],q[43];
cx q[37],q[42];
cx q[38],q[41];
cx q[39],q[40];
u3(0.0*pi,-0.5*pi,1.0409016688576922*pi) q[40];
u3(0.0*pi,-0.5*pi,1.3549517276164655*pi) q[41];
u3(0.0*pi,-0.5*pi,4.377482100717933*pi) q[42];
u3(0.0*pi,-0.5*pi,3.797193058726867*pi) q[43];
u3(0.0*pi,-0.5*pi,3.6028338079610727*pi) q[45];
u3(0.0*pi,-0.5*pi,4.476065676499681*pi) q[46];
u3(0.0*pi,-0.5*pi,4.4915954880982785*pi) q[47];
cx q[33],q[47];
cx q[34],q[46];
cx q[35],q[45];
cx q[36],q[43];
cx q[37],q[42];
cx q[38],q[41];
cx q[39],q[40];
cx q[34],q[47];
cx q[35],q[46];
cx q[36],q[45];
cx q[37],q[43];
cx q[38],q[42];
cx q[39],q[41];
u3(0.0*pi,-0.5*pi,0.6600887264517992*pi) q[40];
cx q[40],q[49];
u3(0.0*pi,-0.5*pi,4.4915954880982785*pi) q[41];
u3(0.0*pi,-0.5*pi,4.476065676499681*pi) q[42];
u3(0.0*pi,-0.5*pi,3.6028338079610727*pi) q[43];
u3(0.0*pi,-0.5*pi,3.584515286056159*pi) q[45];
u3(0.0*pi,-0.5*pi,4.466209237970645*pi) q[46];
u3(0.0*pi,-0.5*pi,1.198623875813658*pi) q[47];
cx q[34],q[47];
cx q[35],q[46];
cx q[36],q[45];
cx q[37],q[43];
cx q[38],q[42];
cx q[39],q[41];
u3(0.0*pi,-0.5*pi,0.7802976777121895*pi) q[49];
cx q[35],q[47];
cx q[36],q[46];
cx q[37],q[44];
cx q[38],q[43];
cx q[39],q[42];
cx q[40],q[49];
u3(0.0*pi,-0.5*pi,1.085597795724301*pi) q[41];
cx q[40],q[50];
cx q[41],q[48];
u3(0.0*pi,-0.5*pi,1.198623875813658*pi) q[42];
u3(0.0*pi,-0.5*pi,4.466209237970645*pi) q[43];
u3(0.0*pi,-0.5*pi,3.584515286056159*pi) q[44];
u3(0.0*pi,-0.5*pi,1.361981074856414*pi) q[46];
u3(0.0*pi,-0.5*pi,0.7859518602903652*pi) q[47];
cx q[35],q[47];
cx q[36],q[46];
cx q[37],q[44];
cx q[38],q[43];
cx q[39],q[42];
u3(0.0*pi,-0.5*pi,0.7802976777121895*pi) q[48];
u3(0.0*pi,-0.5*pi,3.5521380696713476*pi) q[50];
cx q[36],q[47];
cx q[37],q[46];
cx q[38],q[44];
cx q[39],q[43];
cx q[40],q[50];
cx q[41],q[48];
u3(0.0*pi,-0.5*pi,1.2715909155711453*pi) q[42];
cx q[40],q[51];
cx q[41],q[50];
cx q[42],q[48];
u3(0.0*pi,-0.5*pi,0.7859518602903652*pi) q[43];
u3(0.0*pi,-0.5*pi,1.361981074856414*pi) q[44];
u3(0.0*pi,-0.5*pi,3.8816987495373327*pi) q[46];
u3(0.0*pi,-0.5*pi,0.6887626396617753*pi) q[47];
cx q[36],q[47];
cx q[37],q[46];
cx q[38],q[44];
cx q[39],q[43];
u3(0.0*pi,-0.5*pi,3.5521380696713476*pi) q[48];
u3(0.0*pi,-0.5*pi,4.498605644846842*pi) q[50];
u3(0.0*pi,-0.5*pi,4.05636431373644*pi) q[51];
cx q[37],q[47];
cx q[38],q[45];
cx q[39],q[44];
cx q[40],q[51];
cx q[41],q[50];
cx q[42],q[48];
u3(0.0*pi,-0.5*pi,1.0377867201978201*pi) q[43];
cx q[40],q[52];
cx q[41],q[51];
cx q[42],q[49];
cx q[43],q[48];
u3(0.0*pi,-0.5*pi,0.6887626396617753*pi) q[44];
u3(0.0*pi,-0.5*pi,3.8816987495373327*pi) q[45];
u3(0.0*pi,-0.5*pi,0.5380505894543723*pi) q[47];
cx q[37],q[47];
cx q[38],q[45];
cx q[39],q[44];
u3(0.0*pi,-0.5*pi,4.05636431373644*pi) q[48];
u3(0.0*pi,-0.5*pi,4.498605644846842*pi) q[49];
u3(0.0*pi,-0.5*pi,0.9169126524569245*pi) q[51];
u3(0.0*pi,-0.5*pi,0.7169376000957258*pi) q[52];
cx q[38],q[47];
cx q[39],q[45];
cx q[40],q[52];
cx q[41],q[51];
cx q[42],q[49];
cx q[43],q[48];
u3(0.0*pi,-0.5*pi,0.8265033026802702*pi) q[44];
cx q[40],q[53];
cx q[41],q[52];
cx q[42],q[51];
cx q[43],q[49];
cx q[44],q[48];
u3(0.0*pi,-0.5*pi,0.5380505894543723*pi) q[45];
u3(0.0*pi,-0.5*pi,1.138052508503511*pi) q[47];
cx q[38],q[47];
cx q[39],q[45];
u3(0.0*pi,-0.5*pi,0.7169376000957258*pi) q[48];
u3(0.0*pi,-0.5*pi,0.9169126524569245*pi) q[49];
u3(0.0*pi,-0.5*pi,3.605641708758775*pi) q[51];
u3(0.0*pi,-0.5*pi,0.7817112233567335*pi) q[52];
u3(0.0*pi,-0.5*pi,1.063422446713467*pi) q[53];
cx q[39],q[46];
cx q[40],q[53];
cx q[41],q[52];
cx q[42],q[51];
cx q[43],q[49];
cx q[44],q[48];
u3(0.0*pi,-0.5*pi,0.6377406630184949*pi) q[45];
u3(0.0*pi,-0.5*pi,3.7361227386406943*pi) q[47];
cx q[40],q[54];
cx q[41],q[53];
cx q[42],q[52];
cx q[43],q[50];
cx q[44],q[49];
cx q[45],q[48];
u3(0.0*pi,-0.5*pi,1.138052508503511*pi) q[46];
cx q[39],q[46];
u3(0.0*pi,-0.5*pi,1.063422446713467*pi) q[48];
u3(0.0*pi,-0.5*pi,0.7817112233567335*pi) q[49];
u3(0.0*pi,-0.5*pi,3.605641708758775*pi) q[50];
u3(0.0*pi,-0.5*pi,1.4056474659061906*pi) q[52];
u3(0.0*pi,-0.5*pi,1.383126688050416*pi) q[53];
u3(0.0*pi,-0.5*pi,1.2098938599872382*pi) q[54];
cx q[40],q[54];
cx q[41],q[53];
cx q[42],q[52];
cx q[43],q[50];
cx q[44],q[49];
cx q[45],q[48];
u3(0.0*pi,-0.5*pi,3.7222943300574247*pi) q[46];
cx q[40],q[55];
cx q[41],q[54];
cx q[42],q[53];
cx q[43],q[52];
cx q[44],q[50];
cx q[45],q[49];
cx q[46],q[48];
u3(0.0*pi,-0.5*pi,1.2098938599872382*pi) q[48];
u3(0.0*pi,-0.5*pi,1.383126688050416*pi) q[49];
u3(0.0*pi,-0.5*pi,1.4056474659061906*pi) q[50];
u3(0.0*pi,-0.5*pi,3.797193058726867*pi) q[52];
u3(0.0*pi,-0.5*pi,4.377482100717933*pi) q[53];
u3(0.0*pi,-0.5*pi,1.3549517276164655*pi) q[54];
u3(0.0*pi,-0.5*pi,1.0409016688576922*pi) q[55];
cx q[40],q[55];
cx q[41],q[54];
cx q[42],q[53];
cx q[43],q[52];
cx q[44],q[50];
cx q[45],q[49];
cx q[46],q[48];
cx q[41],q[55];
cx q[42],q[54];
cx q[43],q[53];
cx q[44],q[51];
cx q[45],q[50];
cx q[46],q[49];
cx q[47],q[48];
u3(0.0*pi,-0.5*pi,1.0409016688576922*pi) q[48];
u3(0.0*pi,-0.5*pi,1.3549517276164655*pi) q[49];
u3(0.0*pi,-0.5*pi,4.377482100717933*pi) q[50];
u3(0.0*pi,-0.5*pi,3.797193058726867*pi) q[51];
u3(0.0*pi,-0.5*pi,3.6028338079610727*pi) q[53];
u3(0.0*pi,-0.5*pi,4.476065676499681*pi) q[54];
u3(0.0*pi,-0.5*pi,4.4915954880982785*pi) q[55];
cx q[41],q[55];
cx q[42],q[54];
cx q[43],q[53];
cx q[44],q[51];
cx q[45],q[50];
cx q[46],q[49];
cx q[47],q[48];
cx q[42],q[55];
cx q[43],q[54];
cx q[44],q[53];
cx q[45],q[51];
cx q[46],q[50];
cx q[47],q[49];
u3(0.0*pi,-0.5*pi,0.6600887264517992*pi) q[48];
cx q[48],q[57];
u3(0.0*pi,-0.5*pi,4.4915954880982785*pi) q[49];
u3(0.0*pi,-0.5*pi,4.476065676499681*pi) q[50];
u3(0.0*pi,-0.5*pi,3.6028338079610727*pi) q[51];
u3(0.0*pi,-0.5*pi,3.584515286056159*pi) q[53];
u3(0.0*pi,-0.5*pi,4.466209237970645*pi) q[54];
u3(0.0*pi,-0.5*pi,1.198623875813658*pi) q[55];
cx q[42],q[55];
cx q[43],q[54];
cx q[44],q[53];
cx q[45],q[51];
cx q[46],q[50];
cx q[47],q[49];
u3(0.0*pi,-0.5*pi,0.7802976777121895*pi) q[57];
cx q[43],q[55];
cx q[44],q[54];
cx q[45],q[52];
cx q[46],q[51];
cx q[47],q[50];
cx q[48],q[57];
u3(0.0*pi,-0.5*pi,1.085597795724301*pi) q[49];
cx q[48],q[58];
cx q[49],q[56];
u3(0.0*pi,-0.5*pi,1.198623875813658*pi) q[50];
u3(0.0*pi,-0.5*pi,4.466209237970645*pi) q[51];
u3(0.0*pi,-0.5*pi,3.584515286056159*pi) q[52];
u3(0.0*pi,-0.5*pi,1.361981074856414*pi) q[54];
u3(0.0*pi,-0.5*pi,0.7859518602903652*pi) q[55];
cx q[43],q[55];
cx q[44],q[54];
cx q[45],q[52];
cx q[46],q[51];
cx q[47],q[50];
u3(0.0*pi,-0.5*pi,0.7802976777121895*pi) q[56];
u3(0.0*pi,-0.5*pi,3.5521380696713476*pi) q[58];
cx q[44],q[55];
cx q[45],q[54];
cx q[46],q[52];
cx q[47],q[51];
cx q[48],q[58];
cx q[49],q[56];
u3(0.0*pi,-0.5*pi,1.2715909155711453*pi) q[50];
cx q[48],q[59];
cx q[49],q[58];
cx q[50],q[56];
u3(0.0*pi,-0.5*pi,0.7859518602903652*pi) q[51];
u3(0.0*pi,-0.5*pi,1.361981074856414*pi) q[52];
u3(0.0*pi,-0.5*pi,3.8816987495373327*pi) q[54];
u3(0.0*pi,-0.5*pi,0.6887626396617753*pi) q[55];
cx q[44],q[55];
cx q[45],q[54];
cx q[46],q[52];
cx q[47],q[51];
u3(0.0*pi,-0.5*pi,3.5521380696713476*pi) q[56];
u3(0.0*pi,-0.5*pi,4.498605644846842*pi) q[58];
u3(0.0*pi,-0.5*pi,4.05636431373644*pi) q[59];
cx q[45],q[55];
cx q[46],q[53];
cx q[47],q[52];
cx q[48],q[59];
cx q[49],q[58];
cx q[50],q[56];
u3(0.0*pi,-0.5*pi,1.0377867201978201*pi) q[51];
cx q[48],q[60];
cx q[49],q[59];
cx q[50],q[57];
cx q[51],q[56];
u3(0.0*pi,-0.5*pi,0.6887626396617753*pi) q[52];
u3(0.0*pi,-0.5*pi,3.8816987495373327*pi) q[53];
u3(0.0*pi,-0.5*pi,0.5380505894543723*pi) q[55];
cx q[45],q[55];
cx q[46],q[53];
cx q[47],q[52];
u3(0.0*pi,-0.5*pi,4.05636431373644*pi) q[56];
u3(0.0*pi,-0.5*pi,4.498605644846842*pi) q[57];
u3(0.0*pi,-0.5*pi,0.9169126524569245*pi) q[59];
u3(0.0*pi,-0.5*pi,0.7169376000957258*pi) q[60];
cx q[46],q[55];
cx q[47],q[53];
cx q[48],q[60];
cx q[49],q[59];
cx q[50],q[57];
cx q[51],q[56];
u3(0.0*pi,-0.5*pi,0.8265033026802702*pi) q[52];
cx q[48],q[61];
cx q[49],q[60];
cx q[50],q[59];
cx q[51],q[57];
cx q[52],q[56];
u3(0.0*pi,-0.5*pi,0.5380505894543723*pi) q[53];
u3(0.0*pi,-0.5*pi,1.138052508503511*pi) q[55];
cx q[46],q[55];
cx q[47],q[53];
u3(0.0*pi,-0.5*pi,0.7169376000957258*pi) q[56];
u3(0.0*pi,-0.5*pi,0.9169126524569245*pi) q[57];
u3(0.0*pi,-0.5*pi,3.605641708758775*pi) q[59];
u3(0.0*pi,-0.5*pi,0.7817112233567335*pi) q[60];
u3(0.0*pi,-0.5*pi,1.063422446713467*pi) q[61];
cx q[47],q[54];
cx q[48],q[61];
cx q[49],q[60];
cx q[50],q[59];
cx q[51],q[57];
cx q[52],q[56];
u3(0.0*pi,-0.5*pi,0.6377406630184949*pi) q[53];
u3(0.0*pi,-0.5*pi,3.7361227386406943*pi) q[55];
cx q[48],q[62];
cx q[49],q[61];
cx q[50],q[60];
cx q[51],q[58];
cx q[52],q[57];
cx q[53],q[56];
u3(0.0*pi,-0.5*pi,1.138052508503511*pi) q[54];
cx q[47],q[54];
u3(0.0*pi,-0.5*pi,1.063422446713467*pi) q[56];
u3(0.0*pi,-0.5*pi,0.7817112233567335*pi) q[57];
u3(0.0*pi,-0.5*pi,3.605641708758775*pi) q[58];
u3(0.0*pi,-0.5*pi,1.4056474659061906*pi) q[60];
u3(0.0*pi,-0.5*pi,1.383126688050416*pi) q[61];
u3(0.0*pi,-0.5*pi,1.2098938599872382*pi) q[62];
cx q[48],q[62];
cx q[49],q[61];
cx q[50],q[60];
cx q[51],q[58];
cx q[52],q[57];
cx q[53],q[56];
u3(0.0*pi,-0.5*pi,3.7222943300574247*pi) q[54];
cx q[48],q[63];
cx q[49],q[62];
cx q[50],q[61];
cx q[51],q[60];
cx q[52],q[58];
cx q[53],q[57];
cx q[54],q[56];
u3(0.0*pi,-0.5*pi,1.2098938599872382*pi) q[56];
u3(0.0*pi,-0.5*pi,1.383126688050416*pi) q[57];
u3(0.0*pi,-0.5*pi,1.4056474659061906*pi) q[58];
u3(0.0*pi,-0.5*pi,3.797193058726867*pi) q[60];
u3(0.0*pi,-0.5*pi,4.377482100717933*pi) q[61];
u3(0.0*pi,-0.5*pi,1.3549517276164655*pi) q[62];
u3(0.0*pi,-0.5*pi,1.0409016688576922*pi) q[63];
cx q[48],q[63];
cx q[49],q[62];
cx q[50],q[61];
cx q[51],q[60];
cx q[52],q[58];
cx q[53],q[57];
cx q[54],q[56];
cx q[49],q[63];
cx q[50],q[62];
cx q[51],q[61];
cx q[52],q[59];
cx q[53],q[58];
cx q[54],q[57];
cx q[55],q[56];
u3(0.0*pi,-0.5*pi,1.0409016688576922*pi) q[56];
u3(0.0*pi,-0.5*pi,1.3549517276164655*pi) q[57];
u3(0.0*pi,-0.5*pi,4.377482100717933*pi) q[58];
u3(0.0*pi,-0.5*pi,3.797193058726867*pi) q[59];
u3(0.0*pi,-0.5*pi,3.6028338079610727*pi) q[61];
u3(0.0*pi,-0.5*pi,4.476065676499681*pi) q[62];
u3(0.0*pi,-0.5*pi,4.4915954880982785*pi) q[63];
cx q[49],q[63];
cx q[50],q[62];
cx q[51],q[61];
cx q[52],q[59];
cx q[53],q[58];
cx q[54],q[57];
cx q[55],q[56];
cx q[50],q[63];
cx q[51],q[62];
cx q[52],q[61];
cx q[53],q[59];
cx q[54],q[58];
cx q[55],q[57];
u3(0.0*pi,-0.5*pi,0.6600887264517992*pi) q[56];
u3(0.0*pi,-0.5*pi,4.4915954880982785*pi) q[57];
u3(0.0*pi,-0.5*pi,4.476065676499681*pi) q[58];
u3(0.0*pi,-0.5*pi,3.6028338079610727*pi) q[59];
u3(0.0*pi,-0.5*pi,3.584515286056159*pi) q[61];
u3(0.0*pi,-0.5*pi,4.466209237970645*pi) q[62];
u3(0.0*pi,-0.5*pi,1.198623875813658*pi) q[63];
cx q[50],q[63];
cx q[51],q[62];
cx q[52],q[61];
cx q[53],q[59];
cx q[54],q[58];
cx q[55],q[57];
cx q[51],q[63];
cx q[52],q[62];
cx q[53],q[60];
cx q[54],q[59];
cx q[55],q[58];
u3(0.0*pi,-0.5*pi,1.085597795724301*pi) q[57];
u3(0.0*pi,-0.5*pi,1.198623875813658*pi) q[58];
u3(0.0*pi,-0.5*pi,4.466209237970645*pi) q[59];
u3(0.0*pi,-0.5*pi,3.584515286056159*pi) q[60];
u3(0.0*pi,-0.5*pi,1.361981074856414*pi) q[62];
u3(0.0*pi,-0.5*pi,0.7859518602903652*pi) q[63];
cx q[51],q[63];
cx q[52],q[62];
cx q[53],q[60];
cx q[54],q[59];
cx q[55],q[58];
cx q[52],q[63];
cx q[53],q[62];
cx q[54],q[60];
cx q[55],q[59];
u3(0.0*pi,-0.5*pi,1.2715909155711453*pi) q[58];
u3(0.0*pi,-0.5*pi,0.7859518602903652*pi) q[59];
u3(0.0*pi,-0.5*pi,1.361981074856414*pi) q[60];
u3(0.0*pi,-0.5*pi,3.8816987495373327*pi) q[62];
u3(0.0*pi,-0.5*pi,0.6887626396617753*pi) q[63];
cx q[52],q[63];
cx q[53],q[62];
cx q[54],q[60];
cx q[55],q[59];
cx q[53],q[63];
cx q[54],q[61];
cx q[55],q[60];
u3(0.0*pi,-0.5*pi,1.0377867201978201*pi) q[59];
u3(0.0*pi,-0.5*pi,0.6887626396617753*pi) q[60];
u3(0.0*pi,-0.5*pi,3.8816987495373327*pi) q[61];
u3(0.0*pi,-0.5*pi,0.5380505894543723*pi) q[63];
cx q[53],q[63];
cx q[54],q[61];
cx q[55],q[60];
cx q[54],q[63];
cx q[55],q[61];
u3(0.0*pi,-0.5*pi,0.8265033026802702*pi) q[60];
u3(0.0*pi,-0.5*pi,0.5380505894543723*pi) q[61];
u3(0.0*pi,-0.5*pi,1.138052508503511*pi) q[63];
cx q[54],q[63];
cx q[55],q[61];
cx q[55],q[62];
u3(0.0*pi,-0.5*pi,0.6377406630184949*pi) q[61];
u3(0.0*pi,-0.5*pi,3.7361227386406943*pi) q[63];
u3(0.0*pi,-0.5*pi,1.138052508503511*pi) q[62];
cx q[55],q[62];
u3(0.0*pi,-0.5*pi,3.7222943300574247*pi) q[62];
