OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
u2(-pi,-pi) q[0];
u2(-pi,-pi) q[1];
u2(-pi,-pi) q[2];
u2(0,-pi) q[3];
u3(0.7274746214270381,-pi/2,pi/2) q[4];
u2(-pi,-pi) q[5];
u2(-2.5119253413580616,-pi) q[6];
u2(0,1.0754290375762228) q[7];
u3(0.10110569981616446,-pi/2,-pi) q[8];
cx q[7],q[8];
u3(pi,0.004205296856392593,1.5750016236512892) q[7];
u3(1.4996329033974247,1.4738626979022218,-2.837005671244258) q[8];
cx q[7],q[8];
u2(0.4953672892186729,-pi/2) q[7];
u3(1.5255836036689971,-3.0511285591156643,0.4622486115335107) q[8];
cx q[6],q[8];
u3(0.999999999999999,-pi,-0.6296673122317324) q[6];
u3(1.8151647370584045,1.9696283203748335,2.620420755356923) q[8];
u2(-pi/2,pi/2) q[9];
cx q[9],q[8];
u1(1.0) q[8];
cx q[6],q[8];
u2(0,-pi/2) q[6];
cx q[9],q[8];
cx q[6],q[8];
u2(-pi/2,-pi) q[9];
cx q[9],q[8];
u1(1.0) q[8];
cx q[6],q[8];
u2(-pi/2,-pi) q[6];
cx q[9],q[8];
u3(pi/2,-pi/2,pi/2) q[8];
cx q[6],q[8];
u2(0,-pi/2) q[9];
cx q[9],q[8];
u1(1.0) q[8];
cx q[6],q[8];
cx q[9],q[8];
u2(-pi/2,-pi) q[8];
cx q[6],q[8];
u2(-pi/2,-pi) q[9];
cx q[9],q[8];
u1(1.0) q[8];
cx q[9],q[8];
u2(0,-pi/2) q[9];
cx q[9],q[8];
u1(1.0) q[8];
cx q[6],q[8];
cx q[9],q[8];
u2(0,-pi/2) q[8];
cx q[6],q[8];
u2(-pi/2,-pi) q[9];
cx q[9],q[8];
u1(1.0) q[8];
cx q[6],q[8];
u2(0,-pi/2) q[6];
cx q[9],q[8];
u2(-pi/2,-pi) q[8];
cx q[6],q[8];
u2(0,-pi/2) q[9];
cx q[9],q[8];
u1(1.0) q[8];
cx q[6],q[8];
cx q[9],q[8];
u2(0,-pi/2) q[8];
cx q[6],q[8];
u2(-pi/2,-pi) q[9];
cx q[9],q[8];
u1(1.0) q[8];
cx q[9],q[8];
u2(0,-pi/2) q[9];
cx q[9],q[8];
u1(1.0) q[8];
cx q[6],q[8];
cx q[9],q[8];
u2(-pi/2,-pi) q[8];
cx q[6],q[8];
u2(-pi/2,-pi) q[9];
cx q[9],q[8];
u1(1.0) q[8];
cx q[6],q[8];
u2(pi/2,-pi/2) q[6];
cx q[9],q[8];
u2(0,-pi/2) q[8];
cx q[5],q[8];
u2(0,-pi/2) q[9];
cx q[9],q[8];
u1(1.0) q[8];
cx q[5],q[8];
cx q[9],q[8];
u2(-pi/2,-pi) q[8];
cx q[5],q[8];
u2(-pi/2,-pi) q[9];
cx q[9],q[8];
u1(1.0) q[8];
cx q[9],q[8];
u2(0,-pi/2) q[9];
cx q[9],q[8];
u1(1.0) q[8];
cx q[5],q[8];
cx q[9],q[8];
u2(0,-pi/2) q[8];
cx q[5],q[8];
u2(-pi/2,-pi) q[9];
cx q[9],q[8];
u1(1.0) q[8];
cx q[5],q[8];
u2(0,-pi/2) q[5];
cx q[9],q[8];
u2(-pi/2,-pi) q[8];
cx q[5],q[8];
u2(0,-pi/2) q[9];
cx q[9],q[8];
u1(1.0) q[8];
cx q[5],q[8];
cx q[9],q[8];
u2(0,-pi/2) q[8];
cx q[5],q[8];
u2(-pi/2,-pi) q[9];
cx q[9],q[8];
u1(1.0) q[8];
cx q[9],q[8];
u2(0,-pi/2) q[9];
cx q[9],q[8];
u1(1.0) q[8];
cx q[5],q[8];
cx q[9],q[8];
u2(-pi/2,-pi) q[8];
cx q[5],q[8];
u2(-pi/2,-pi) q[9];
cx q[9],q[8];
u1(1.0) q[8];
cx q[5],q[8];
u2(-pi/2,-pi) q[5];
u2(2.139911523761924,0) q[8];
u2(0,0.10110569981616457) q[9];
cx q[8],q[9];
u2(pi/2,2.5724774566227655) q[8];
cx q[5],q[8];
cx q[7],q[8];
u1(1.0) q[8];
cx q[5],q[8];
u2(0,-pi/2) q[5];
cx q[7],q[8];
u2(-pi/2,-pi) q[8];
cx q[5],q[8];
cx q[7],q[8];
u1(1.0) q[8];
cx q[5],q[8];
u2(-pi/2,-pi) q[5];
cx q[6],q[5];
cx q[7],q[8];
u3(pi/2,0,pi) q[8];
u2(0,-pi) q[7];
cx q[8],q[5];
u1(1.0) q[5];
cx q[6],q[5];
cx q[8],q[5];
u2(0,-pi/2) q[5];
cx q[6],q[5];
cx q[8],q[5];
u1(1.0) q[5];
cx q[6],q[5];
cx q[8],q[5];
u3(pi/2,-pi/2,pi/2) q[8];
u2(-pi/2,-pi) q[5];
cx q[6],q[5];
cx q[8],q[5];
u1(1.0) q[5];
cx q[8],q[5];
u2(-pi/2,-pi) q[8];
cx q[8],q[5];
u1(2.0) q[5];
cx q[8],q[5];
u2(0,-pi/2) q[8];
cx q[8],q[5];
u1(1.0) q[5];
cx q[6],q[5];
cx q[8],q[5];
u2(0,-pi/2) q[5];
cx q[6],q[5];
u2(0,0.10110569981616457) q[6];
u2(-pi/2,-pi) q[8];
cx q[8],q[5];
u1(1.0) q[5];
cx q[8],q[5];
u2(0,-pi/2) q[8];
cx q[8],q[5];
u1(2.0) q[5];
cx q[8],q[5];
u2(-pi/2,-pi) q[8];
cx q[8],q[5];
u2(0.5083472040082082,1.0) q[5];
cx q[5],q[6];
u1(2.237319125014527) q[5];
u2(-0.10110569981616546,-pi) q[6];
u2(0,0.10110569981616457) q[8];
cx q[5],q[8];
u3(pi,-1.6894574576077526,0.27726519375420233) q[5];
u3(1.4696906269787315,-pi/2,pi/2) q[8];
cx q[2],q[8];
cx q[7],q[8];
u1(1.0) q[8];
cx q[2],q[8];
cx q[7],q[8];
u2(-pi/2,-pi) q[8];
cx q[2],q[8];
cx q[7],q[8];
u1(1.0) q[8];
cx q[2],q[8];
cx q[7],q[8];
u2(0,-pi/2) q[7];
u2(0,-pi/2) q[8];
cx q[2],q[8];
cx q[7],q[8];
u1(1.0) q[8];
cx q[2],q[8];
cx q[7],q[8];
u2(-pi/2,-pi) q[8];
cx q[2],q[8];
cx q[7],q[8];
u1(1.0) q[8];
cx q[2],q[8];
u2(0,-pi/2) q[2];
cx q[7],q[8];
u2(-pi/2,-pi) q[7];
u2(0,-pi/2) q[8];
cx q[2],q[8];
cx q[7],q[8];
u1(1.0) q[8];
cx q[2],q[8];
cx q[7],q[8];
u2(-pi/2,-pi) q[8];
cx q[2],q[8];
cx q[7],q[8];
u1(1.0) q[8];
cx q[2],q[8];
cx q[7],q[8];
u2(0,-pi/2) q[7];
u2(0,-pi/2) q[8];
cx q[2],q[8];
cx q[7],q[8];
u1(1.0) q[8];
cx q[2],q[8];
cx q[7],q[8];
u2(-pi/2,-pi) q[8];
cx q[2],q[8];
cx q[7],q[8];
u1(1.0) q[8];
cx q[2],q[8];
u2(-pi/2,-pi) q[2];
cx q[3],q[2];
cx q[7],q[8];
u3(pi/2,0,pi) q[8];
u2(pi/2,-pi/2) q[7];
cx q[8],q[2];
u1(1.0) q[2];
cx q[3],q[2];
u2(0,-pi/2) q[3];
cx q[3],q[2];
u1(2.0) q[2];
cx q[3],q[2];
u2(-pi/2,-pi) q[3];
cx q[3],q[2];
u1(1.0) q[2];
cx q[3],q[2];
u2(0,-pi/2) q[3];
cx q[8],q[2];
u2(0,-pi/2) q[2];
cx q[3],q[2];
cx q[8],q[2];
u1(1.0) q[2];
cx q[3],q[2];
u2(-pi/2,-pi) q[3];
cx q[3],q[2];
u1(2.0) q[2];
cx q[3],q[2];
u2(0,-pi/2) q[3];
cx q[3],q[2];
u1(1.0) q[2];
cx q[3],q[2];
u2(-pi/2,-pi) q[3];
cx q[8],q[2];
u2(-pi/2,-pi) q[2];
cx q[3],q[2];
cx q[7],q[2];
u1(1.0) q[2];
cx q[3],q[2];
u2(0,-pi/2) q[3];
cx q[3],q[2];
u1(2.0) q[2];
cx q[3],q[2];
u2(-pi/2,-pi) q[3];
cx q[3],q[2];
u1(1.0) q[2];
cx q[3],q[2];
u2(0,-pi/2) q[3];
cx q[7],q[2];
u2(0,-pi/2) q[2];
cx q[3],q[2];
cx q[7],q[2];
u1(1.0) q[2];
cx q[3],q[2];
u2(-pi/2,-pi) q[3];
cx q[3],q[2];
u1(2.0) q[2];
cx q[3],q[2];
u2(0,-pi/2) q[3];
cx q[3],q[2];
u1(1.0) q[2];
cx q[3],q[2];
u2(-pi/2,-pi) q[3];
cx q[7],q[2];
u2(-pi/2,-pi) q[2];
cx q[3],q[2];
cx q[7],q[2];
u2(-pi/2,pi/2) q[8];
u3(1.469690626978729,-pi/2,pi/2) q[9];
cx q[9],q[2];
u1(1.0) q[2];
cx q[3],q[2];
u2(0,-pi/2) q[3];
cx q[9],q[2];
cx q[3],q[2];
u2(-pi/2,-pi) q[9];
cx q[9],q[2];
u1(1.0) q[2];
cx q[9],q[2];
u2(0,-pi/2) q[9];
cx q[9],q[2];
u1(1.0) q[2];
cx q[3],q[2];
u2(-pi/2,-pi) q[3];
cx q[9],q[2];
cx q[3],q[2];
u2(-pi/2,-pi) q[9];
cx q[9],q[2];
u1(1.0) q[2];
cx q[3],q[2];
u2(0,-pi/2) q[3];
cx q[7],q[2];
cx q[9],q[2];
u2(0,-pi/2) q[2];
cx q[3],q[2];
cx q[7],q[2];
u2(0,-pi/2) q[9];
cx q[9],q[2];
u1(1.0) q[2];
cx q[3],q[2];
u2(-pi/2,-pi) q[3];
cx q[9],q[2];
cx q[3],q[2];
u2(-pi/2,-pi) q[9];
cx q[9],q[2];
u1(1.0) q[2];
cx q[9],q[2];
u2(0,-pi/2) q[9];
cx q[9],q[2];
u1(1.0) q[2];
cx q[3],q[2];
u2(0,-pi/2) q[3];
cx q[9],q[2];
cx q[3],q[2];
u2(-pi/2,-pi) q[9];
cx q[9],q[2];
u1(1.0) q[2];
cx q[3],q[2];
u2(-pi/2,-pi) q[3];
cx q[7],q[2];
u3(pi/2,0,pi) q[7];
u2(1.2247291581313782,-pi) q[2];
u2(-pi,-0.10110569981616457) q[9];
cx q[2],q[9];
u1(0.3460671686635175) q[2];
cx q[3],q[2];
cx q[6],q[2];
cx q[8],q[2];
u1(1.0) q[2];
cx q[3],q[2];
u2(0,-pi/2) q[3];
cx q[8],q[2];
cx q[3],q[2];
u2(-pi/2,-pi) q[8];
cx q[8],q[2];
u1(1.0) q[2];
cx q[8],q[2];
u2(0,-pi/2) q[8];
cx q[8],q[2];
u1(1.0) q[2];
cx q[3],q[2];
u2(-pi/2,-pi) q[3];
cx q[8],q[2];
cx q[3],q[2];
u2(-pi/2,-pi) q[8];
cx q[8],q[2];
u1(1.0) q[2];
cx q[3],q[2];
u2(0,-pi/2) q[3];
cx q[6],q[2];
cx q[8],q[2];
u2(0,-pi/2) q[2];
cx q[3],q[2];
cx q[6],q[2];
u2(0,-pi/2) q[8];
cx q[8],q[2];
u1(1.0) q[2];
cx q[3],q[2];
u2(-pi/2,-pi) q[3];
cx q[8],q[2];
cx q[3],q[2];
u2(-pi/2,-pi) q[8];
cx q[8],q[2];
u1(1.0) q[2];
cx q[8],q[2];
u2(0,-pi/2) q[8];
cx q[8],q[2];
u1(1.0) q[2];
cx q[3],q[2];
u2(0,-pi/2) q[3];
cx q[8],q[2];
cx q[3],q[2];
u2(-pi/2,-pi) q[8];
cx q[8],q[2];
u1(1.0) q[2];
cx q[3],q[2];
u2(-pi/2,-pi) q[3];
cx q[6],q[2];
cx q[8],q[2];
u3(pi/2,0,pi) q[8];
u2(-pi/2,-pi) q[2];
cx q[3],q[2];
cx q[6],q[2];
cx q[8],q[2];
u1(1.0) q[2];
cx q[3],q[2];
u2(0,-pi/2) q[3];
cx q[3],q[2];
u1(2.0) q[2];
cx q[3],q[2];
u2(-pi/2,-pi) q[3];
cx q[3],q[2];
u1(1.0) q[2];
cx q[3],q[2];
u2(0,-pi/2) q[3];
cx q[6],q[2];
cx q[8],q[2];
u2(0,-pi/2) q[2];
cx q[3],q[2];
cx q[6],q[2];
cx q[8],q[2];
u1(1.0) q[2];
cx q[3],q[2];
u2(-pi/2,-pi) q[3];
cx q[3],q[2];
u1(2.0) q[2];
cx q[3],q[2];
u2(0,-pi/2) q[3];
cx q[3],q[2];
u1(1.0) q[2];
cx q[3],q[2];
u3(-pi/2,-pi/2,pi/2) q[3];
cx q[6],q[2];
u3(pi/2,0,pi) q[6];
cx q[8],q[2];
u2(-pi/2,-pi) q[2];
cx q[3],q[2];
cx q[7],q[2];
u1(2.0) q[2];
cx q[7],q[2];
u2(0,-pi/2) q[7];
cx q[7],q[2];
u1(2.0) q[2];
cx q[3],q[2];
cx q[7],q[2];
u2(0,-pi/2) q[2];
cx q[3],q[2];
u2(-pi/2,-pi) q[7];
cx q[7],q[2];
u1(2.0) q[2];
cx q[7],q[2];
u2(0,-pi/2) q[7];
cx q[7],q[2];
u1(2.0) q[2];
cx q[3],q[2];
cx q[7],q[2];
u2(-pi/2,-pi) q[2];
cx q[3],q[2];
u2(-pi/2,-pi) q[7];
cx q[7],q[2];
u2(-pi/2,pi/2) q[8];
u3(1.4696906269787322,pi/2,pi/2) q[9];
cx q[9],q[2];
u1(1.0) q[2];
cx q[9],q[2];
u2(-pi/2,-pi) q[9];
cx q[9],q[2];
u1(1.0) q[2];
cx q[7],q[2];
u2(0,-pi/2) q[7];
cx q[9],q[2];
cx q[7],q[2];
u2(0,-pi/2) q[9];
cx q[9],q[2];
u1(1.0) q[2];
cx q[9],q[2];
u2(-pi/2,-pi) q[9];
cx q[9],q[2];
u1(1.0) q[2];
cx q[3],q[2];
cx q[7],q[2];
u2(-pi/2,-pi) q[7];
cx q[9],q[2];
u2(0,-pi/2) q[2];
cx q[3],q[2];
cx q[7],q[2];
u2(0,-pi/2) q[9];
cx q[9],q[2];
u1(1.0) q[2];
cx q[9],q[2];
u2(-pi/2,-pi) q[9];
cx q[9],q[2];
u1(1.0) q[2];
cx q[7],q[2];
u2(0,-pi/2) q[7];
cx q[9],q[2];
cx q[7],q[2];
u2(0,-pi/2) q[9];
cx q[9],q[2];
u1(1.0) q[2];
cx q[9],q[2];
u2(-pi/2,-pi) q[9];
cx q[9],q[2];
u1(1.0) q[2];
cx q[3],q[2];
cx q[7],q[2];
u3(-pi/2,-pi/2,pi/2) q[7];
u2(1.2247291581313782,-pi) q[2];
u2(-pi,-0.10110569981616457) q[9];
cx q[2],q[9];
u1(0.3460671686635175) q[2];
cx q[3],q[2];
cx q[6],q[2];
cx q[8],q[2];
u1(1.0) q[2];
cx q[8],q[2];
u2(-pi/2,-pi) q[8];
cx q[8],q[2];
u1(1.0) q[2];
cx q[6],q[2];
u2(0,-pi/2) q[6];
cx q[8],q[2];
cx q[6],q[2];
u2(0,-pi/2) q[8];
cx q[8],q[2];
u1(1.0) q[2];
cx q[8],q[2];
u2(-pi/2,-pi) q[8];
cx q[8],q[2];
u1(1.0) q[2];
cx q[3],q[2];
cx q[6],q[2];
u2(-pi/2,-pi) q[6];
cx q[8],q[2];
u2(0,-pi/2) q[2];
cx q[3],q[2];
cx q[6],q[2];
u2(0,-pi/2) q[8];
cx q[8],q[2];
u1(1.0) q[2];
cx q[8],q[2];
u2(-pi/2,-pi) q[8];
cx q[8],q[2];
u1(1.0) q[2];
cx q[6],q[2];
u2(0,-pi/2) q[6];
cx q[8],q[2];
cx q[6],q[2];
u2(0,-pi/2) q[8];
cx q[8],q[2];
u1(1.0) q[2];
cx q[8],q[2];
u2(-pi/2,-pi) q[8];
cx q[8],q[2];
u1(1.0) q[2];
cx q[3],q[2];
cx q[6],q[2];
u2(-pi/2,-pi) q[6];
cx q[8],q[2];
u3(pi/2,0,pi) q[8];
u2(-pi/2,-pi) q[2];
cx q[3],q[2];
cx q[6],q[2];
cx q[8],q[2];
u1(2.0) q[2];
cx q[6],q[2];
u2(0,-pi/2) q[6];
cx q[6],q[2];
u1(2.0) q[2];
cx q[3],q[2];
cx q[6],q[2];
u2(-pi/2,-pi) q[6];
cx q[8],q[2];
u2(0,-pi/2) q[2];
cx q[3],q[2];
cx q[6],q[2];
cx q[8],q[2];
u1(2.0) q[2];
cx q[6],q[2];
u2(0,-pi/2) q[6];
cx q[6],q[2];
u1(2.0) q[2];
cx q[3],q[2];
cx q[6],q[2];
u2(-2.0036085556181638,-pi) q[2];
u2(-pi/2,-pi) q[6];
u2(0,0.10110569981616457) q[8];
cx q[2],q[8];
u1(0.43281222882326764) q[2];
cx q[3],q[2];
cx q[5],q[2];
cx q[7],q[2];
u1(2.0) q[2];
cx q[5],q[2];
u2(0,-pi/2) q[5];
cx q[5],q[2];
u1(2.0) q[2];
cx q[3],q[2];
cx q[5],q[2];
u2(-pi/2,-pi) q[5];
cx q[7],q[2];
u2(0,-pi/2) q[2];
cx q[3],q[2];
cx q[5],q[2];
cx q[7],q[2];
u1(2.0) q[2];
cx q[5],q[2];
u2(0,-pi/2) q[5];
cx q[5],q[2];
u1(2.0) q[2];
cx q[3],q[2];
cx q[5],q[2];
u2(-pi/2,-pi) q[5];
cx q[7],q[2];
u2(-pi/2,-pi) q[2];
cx q[3],q[2];
cx q[5],q[2];
cx q[7],q[2];
u3(1.4696906269787304,-pi,pi/2) q[8];
u3(1.4696906269787322,pi/2,pi/2) q[9];
cx q[9],q[2];
u1(1.0) q[2];
cx q[9],q[2];
u2(-pi/2,-pi) q[9];
cx q[9],q[2];
u1(1.0) q[2];
cx q[5],q[2];
u2(0,-pi/2) q[5];
cx q[9],q[2];
cx q[5],q[2];
u2(0,-pi/2) q[9];
cx q[9],q[2];
u1(1.0) q[2];
cx q[9],q[2];
u2(-pi/2,-pi) q[9];
cx q[9],q[2];
u1(1.0) q[2];
cx q[3],q[2];
cx q[5],q[2];
u2(-pi/2,-pi) q[5];
cx q[7],q[2];
cx q[9],q[2];
u2(0,-pi/2) q[2];
cx q[3],q[2];
cx q[5],q[2];
cx q[7],q[2];
u2(0,-pi/2) q[9];
cx q[9],q[2];
u1(1.0) q[2];
cx q[9],q[2];
u2(-pi/2,-pi) q[9];
cx q[9],q[2];
u1(1.0) q[2];
cx q[5],q[2];
u2(0,-pi/2) q[5];
cx q[9],q[2];
cx q[5],q[2];
u2(0,-pi/2) q[9];
cx q[9],q[2];
u1(1.0) q[2];
cx q[9],q[2];
u2(-pi/2,-pi) q[9];
cx q[9],q[2];
u1(1.0) q[2];
cx q[3],q[2];
u3(0.10110569981616685,-pi/2,-pi) q[3];
cx q[5],q[2];
u2(-pi/2,-pi) q[5];
cx q[7],q[2];
u3(pi/2,0,pi) q[7];
cx q[9],q[2];
u3(2.6462253643711198,-pi,0) q[2];
cx q[2],q[3];
u3(pi,0.004205296856392593,1.5750016236512892) q[2];
u3(1.4996329033974247,1.4738626979022218,-2.837005671244258) q[3];
cx q[2],q[3];
u2(0.4953672892186751,-pi) q[2];
cx q[2],q[1];
u3(0.10110569981616378,-pi,-pi/2) q[3];
cx q[7],q[1];
u1(2.0) q[1];
cx q[7],q[1];
u2(0,-pi/2) q[7];
cx q[7],q[1];
u1(2.0) q[1];
cx q[2],q[1];
cx q[7],q[1];
u2(0,-pi/2) q[1];
cx q[2],q[1];
u2(-pi/2,-pi) q[7];
cx q[7],q[1];
u1(2.0) q[1];
cx q[7],q[1];
u2(0,-pi/2) q[7];
cx q[7],q[1];
u1(2.0) q[1];
cx q[2],q[1];
cx q[7],q[1];
u2(-pi/2,-pi) q[1];
cx q[2],q[1];
u2(-pi/2,-pi) q[7];
cx q[7],q[1];
u2(0,-pi/2) q[9];
cx q[9],q[1];
u1(1.0) q[1];
cx q[9],q[1];
u2(-pi/2,-pi) q[9];
cx q[9],q[1];
u1(1.0) q[1];
cx q[7],q[1];
u2(0,-pi/2) q[7];
cx q[9],q[1];
cx q[7],q[1];
u2(0,-pi/2) q[9];
cx q[9],q[1];
u1(1.0) q[1];
cx q[9],q[1];
u2(-pi/2,-pi) q[9];
cx q[9],q[1];
u1(1.0) q[1];
cx q[2],q[1];
cx q[7],q[1];
u2(-pi/2,-pi) q[7];
cx q[9],q[1];
u2(0,-pi/2) q[1];
cx q[2],q[1];
cx q[7],q[1];
u2(0,-pi/2) q[9];
cx q[9],q[1];
u1(1.0) q[1];
cx q[9],q[1];
u2(-pi/2,-pi) q[9];
cx q[9],q[1];
u1(1.0) q[1];
cx q[7],q[1];
u2(0,-pi/2) q[7];
cx q[9],q[1];
cx q[7],q[1];
u2(0,-pi/2) q[9];
cx q[9],q[1];
u1(1.0) q[1];
cx q[9],q[1];
u2(-pi/2,-pi) q[9];
cx q[9],q[1];
u1(1.0) q[1];
cx q[2],q[1];
cx q[7],q[1];
u3(-pi/2,-pi/2,pi/2) q[7];
u2(1.2247291581313782,-pi) q[1];
u2(-pi,-0.10110569981616457) q[9];
cx q[1],q[9];
u1(0.3460671686635175) q[1];
cx q[2],q[1];
cx q[6],q[1];
cx q[8],q[1];
u1(1.0) q[1];
cx q[8],q[1];
u2(-pi/2,-pi) q[8];
cx q[8],q[1];
u1(1.0) q[1];
cx q[6],q[1];
u2(0,-pi/2) q[6];
cx q[8],q[1];
cx q[6],q[1];
u2(0,-pi/2) q[8];
cx q[8],q[1];
u1(1.0) q[1];
cx q[8],q[1];
u2(-pi/2,-pi) q[8];
cx q[8],q[1];
u1(1.0) q[1];
cx q[2],q[1];
cx q[6],q[1];
u2(-pi/2,-pi) q[6];
cx q[8],q[1];
u2(0,-pi/2) q[1];
cx q[2],q[1];
cx q[6],q[1];
u2(0,-pi/2) q[8];
cx q[8],q[1];
u1(1.0) q[1];
cx q[8],q[1];
u2(-pi/2,-pi) q[8];
cx q[8],q[1];
u1(1.0) q[1];
cx q[6],q[1];
u2(0,-pi/2) q[6];
cx q[8],q[1];
cx q[6],q[1];
u2(0,-pi/2) q[8];
cx q[8],q[1];
u1(1.0) q[1];
cx q[8],q[1];
u2(-pi/2,-pi) q[8];
cx q[8],q[1];
u1(1.0) q[1];
cx q[2],q[1];
cx q[6],q[1];
u2(-pi/2,-pi) q[6];
cx q[8],q[1];
u3(pi/2,0,pi) q[8];
u2(-pi/2,-pi) q[1];
cx q[2],q[1];
cx q[6],q[1];
cx q[8],q[1];
u1(2.0) q[1];
cx q[6],q[1];
u2(0,-pi/2) q[6];
cx q[6],q[1];
u1(2.0) q[1];
cx q[2],q[1];
cx q[6],q[1];
u2(-pi/2,-pi) q[6];
cx q[8],q[1];
u2(0,-pi/2) q[1];
cx q[2],q[1];
cx q[6],q[1];
cx q[8],q[1];
u1(2.0) q[1];
cx q[6],q[1];
u2(0,-pi/2) q[6];
cx q[6],q[1];
u1(2.0) q[1];
cx q[2],q[1];
u2(pi/4,0) q[1];
u2(0,0.10110569981616457) q[6];
cx q[1],q[6];
u2(-pi,3*pi/4) q[1];
u3(1.4696906269787318,0,pi/2) q[6];
cx q[8],q[1];
u2(-pi/2,-pi) q[1];
cx q[2],q[1];
cx q[5],q[1];
cx q[7],q[1];
u1(2.0) q[1];
cx q[5],q[1];
u2(0,-pi/2) q[5];
cx q[5],q[1];
u1(2.0) q[1];
cx q[2],q[1];
cx q[5],q[1];
u2(-pi/2,-pi) q[5];
cx q[7],q[1];
u2(0,-pi/2) q[1];
cx q[2],q[1];
cx q[5],q[1];
cx q[7],q[1];
u1(2.0) q[1];
cx q[5],q[1];
u2(0,-pi/2) q[5];
cx q[5],q[1];
u1(2.0) q[1];
cx q[2],q[1];
cx q[5],q[1];
u2(-pi/2,-pi) q[5];
cx q[7],q[1];
u2(-pi/2,-pi) q[1];
cx q[2],q[1];
cx q[5],q[1];
cx q[7],q[1];
u2(-pi/2,pi/2) q[8];
u3(1.4696906269787322,pi/2,pi/2) q[9];
cx q[9],q[1];
u1(1.0) q[1];
cx q[9],q[1];
u2(-pi/2,-pi) q[9];
cx q[9],q[1];
u1(1.0) q[1];
cx q[5],q[1];
u2(0,-pi/2) q[5];
cx q[9],q[1];
cx q[5],q[1];
u2(0,-pi/2) q[9];
cx q[9],q[1];
u1(1.0) q[1];
cx q[9],q[1];
u2(-pi/2,-pi) q[9];
cx q[9],q[1];
u1(1.0) q[1];
cx q[2],q[1];
cx q[5],q[1];
u2(-pi/2,-pi) q[5];
cx q[7],q[1];
cx q[9],q[1];
u2(0,-pi/2) q[1];
cx q[2],q[1];
cx q[5],q[1];
cx q[7],q[1];
u2(0,0.10110569981616457) q[7];
u2(0,-pi/2) q[9];
cx q[9],q[1];
u1(1.0) q[1];
cx q[9],q[1];
u2(-pi/2,-pi) q[9];
cx q[9],q[1];
u1(1.0) q[1];
cx q[5],q[1];
u2(0,-pi/2) q[5];
cx q[9],q[1];
cx q[5],q[1];
u2(-pi,-0.10110569981616457) q[5];
u2(0,-pi/2) q[9];
cx q[9],q[1];
u1(1.0) q[1];
cx q[9],q[1];
u2(-pi/2,-pi) q[9];
cx q[9],q[1];
u1(1.0) q[1];
cx q[2],q[1];
u2(2.1993187700512546,0) q[1];
cx q[1],q[5];
u1(1.4506210875467467) q[1];
cx q[1],q[7];
u3(pi,-3.096893058582391,-1.6883767671321845) q[1];
u2(0,0.10110569981616457) q[2];
u3(0.10110569981616445,-pi,-pi/2) q[5];
u2(-0.10110569981616546,-pi) q[7];
u2(-pi,-0.10110569981616457) q[9];
cx q[1],q[9];
u2(0.8384291219329079,0.3460671686635166) q[1];
cx q[1],q[2];
u3(1.867036871283499,-1.7457423296044778,-1.3337964111515817) q[1];
cx q[1],q[4];
u2(0,0.6703132920282573) q[1];
u2(-0.10110569981616546,-pi) q[2];
u2(-0.7274746214270404,-pi) q[4];
cx q[8],q[1];
u1(1.0) q[1];
cx q[4],q[1];
u2(0,-pi/2) q[4];
cx q[8],q[1];
cx q[4],q[1];
u2(-pi/2,-pi) q[8];
cx q[8],q[1];
u1(1.0) q[1];
cx q[8],q[1];
u2(0,-pi/2) q[8];
cx q[8],q[1];
u1(1.0) q[1];
cx q[4],q[1];
u2(-pi/2,-pi) q[4];
cx q[8],q[1];
cx q[4],q[1];
u2(-pi/2,-pi) q[8];
cx q[8],q[1];
u1(1.0) q[1];
cx q[2],q[1];
cx q[4],q[1];
u2(0,-pi/2) q[4];
cx q[8],q[1];
u2(0,-pi/2) q[1];
cx q[2],q[1];
cx q[4],q[1];
u2(0,-pi/2) q[8];
cx q[8],q[1];
u1(1.0) q[1];
cx q[4],q[1];
u2(-pi/2,-pi) q[4];
cx q[8],q[1];
cx q[4],q[1];
u2(-pi/2,-pi) q[8];
cx q[8],q[1];
u1(1.0) q[1];
cx q[8],q[1];
u2(0,-pi/2) q[8];
cx q[8],q[1];
u1(1.0) q[1];
cx q[4],q[1];
u2(0,-pi/2) q[4];
cx q[8],q[1];
cx q[4],q[1];
u2(-pi/2,-pi) q[8];
cx q[8],q[1];
u1(1.0) q[1];
cx q[2],q[1];
cx q[4],q[1];
u2(-pi/2,-pi) q[4];
cx q[8],q[1];
u3(pi/2,0,pi) q[8];
u2(-pi/2,-pi) q[1];
cx q[2],q[1];
cx q[4],q[1];
cx q[8],q[1];
u1(1.0) q[1];
cx q[4],q[1];
u2(0,-pi/2) q[4];
cx q[4],q[1];
u1(2.0) q[1];
cx q[4],q[1];
u2(-pi/2,-pi) q[4];
cx q[4],q[1];
u1(1.0) q[1];
cx q[2],q[1];
cx q[4],q[1];
u2(0,-pi/2) q[4];
cx q[8],q[1];
u2(0,-pi/2) q[1];
cx q[2],q[1];
cx q[4],q[1];
cx q[8],q[1];
u1(1.0) q[1];
cx q[4],q[1];
u2(-pi/2,-pi) q[4];
cx q[4],q[1];
u1(2.0) q[1];
cx q[4],q[1];
u2(0,-pi/2) q[4];
cx q[4],q[1];
u1(1.0) q[1];
cx q[2],q[1];
cx q[4],q[1];
u2(-pi/2,-pi) q[4];
cx q[8],q[1];
u2(-pi/2,-pi) q[1];
cx q[2],q[1];
cx q[4],q[1];
cx q[7],q[1];
u1(1.0) q[1];
cx q[4],q[1];
u2(0,-pi/2) q[4];
cx q[4],q[1];
u1(2.0) q[1];
cx q[4],q[1];
u2(-pi/2,-pi) q[4];
cx q[4],q[1];
u1(1.0) q[1];
cx q[2],q[1];
cx q[4],q[1];
u2(0,-pi/2) q[4];
cx q[7],q[1];
u2(0,-pi/2) q[1];
cx q[2],q[1];
cx q[4],q[1];
cx q[7],q[1];
u1(1.0) q[1];
cx q[4],q[1];
u2(-pi/2,-pi) q[4];
cx q[4],q[1];
u1(2.0) q[1];
cx q[4],q[1];
u2(0,-pi/2) q[4];
cx q[4],q[1];
u1(1.0) q[1];
cx q[2],q[1];
cx q[4],q[1];
u2(-pi/2,-pi) q[4];
cx q[7],q[1];
u2(-pi/2,-pi) q[1];
cx q[2],q[1];
cx q[4],q[1];
cx q[7],q[1];
u2(-pi/2,pi/2) q[8];
u3(1.4696906269787322,pi/2,pi/2) q[9];
cx q[9],q[1];
u1(1.0) q[1];
cx q[4],q[1];
u2(0,-pi/2) q[4];
cx q[9],q[1];
cx q[4],q[1];
u2(-pi/2,-pi) q[9];
cx q[9],q[1];
u1(1.0) q[1];
cx q[9],q[1];
u2(0,-pi/2) q[9];
cx q[9],q[1];
u1(1.0) q[1];
cx q[4],q[1];
u2(-pi/2,-pi) q[4];
cx q[9],q[1];
cx q[4],q[1];
u2(-pi/2,-pi) q[9];
cx q[9],q[1];
u1(1.0) q[1];
cx q[2],q[1];
cx q[4],q[1];
u2(0,-pi/2) q[4];
cx q[7],q[1];
cx q[9],q[1];
u2(0,-pi/2) q[1];
cx q[2],q[1];
cx q[4],q[1];
cx q[7],q[1];
u2(0,0.10110569981616457) q[7];
u2(0,-pi/2) q[9];
cx q[9],q[1];
u1(1.0) q[1];
cx q[4],q[1];
u2(-pi/2,-pi) q[4];
cx q[9],q[1];
cx q[4],q[1];
u2(-pi/2,-pi) q[9];
cx q[9],q[1];
u1(1.0) q[1];
cx q[9],q[1];
u2(0,-pi/2) q[9];
cx q[9],q[1];
u1(1.0) q[1];
cx q[4],q[1];
u2(0,-pi/2) q[4];
cx q[9],q[1];
cx q[4],q[1];
u2(-pi/2,-pi) q[9];
cx q[9],q[1];
u1(1.0) q[1];
cx q[2],q[1];
cx q[4],q[1];
u2(0.5083472040082082,0) q[1];
cx q[1],q[7];
u3(pi,1.2937453674056565,-1.7725266874372005) q[1];
u2(-pi/2,-pi) q[4];
u2(-0.10110569981616546,-pi) q[7];
u2(-pi,-0.10110569981616457) q[9];
cx q[1],q[9];
u1(2.154464129550041) q[1];
cx q[2],q[1];
cx q[4],q[1];
cx q[6],q[1];
cx q[8],q[1];
u1(1.0) q[1];
cx q[4],q[1];
u2(0,-pi/2) q[4];
cx q[8],q[1];
cx q[4],q[1];
u2(-pi/2,-pi) q[8];
cx q[8],q[1];
u1(1.0) q[1];
cx q[8],q[1];
u2(0,-pi/2) q[8];
cx q[8],q[1];
u1(1.0) q[1];
cx q[4],q[1];
u2(-pi/2,-pi) q[4];
cx q[8],q[1];
cx q[4],q[1];
u2(-pi/2,-pi) q[8];
cx q[8],q[1];
u1(1.0) q[1];
cx q[2],q[1];
cx q[4],q[1];
u2(0,-pi/2) q[4];
cx q[6],q[1];
cx q[8],q[1];
u2(0,-pi/2) q[1];
cx q[2],q[1];
cx q[4],q[1];
cx q[6],q[1];
u2(0,-pi/2) q[8];
cx q[8],q[1];
u1(1.0) q[1];
cx q[4],q[1];
u2(-pi/2,-pi) q[4];
cx q[8],q[1];
cx q[4],q[1];
u2(-pi/2,-pi) q[8];
cx q[8],q[1];
u1(1.0) q[1];
cx q[8],q[1];
u2(0,-pi/2) q[8];
cx q[8],q[1];
u1(1.0) q[1];
cx q[4],q[1];
u2(0,-pi/2) q[4];
cx q[8],q[1];
cx q[4],q[1];
u2(-pi/2,-pi) q[8];
cx q[8],q[1];
u1(1.0) q[1];
cx q[2],q[1];
cx q[4],q[1];
u2(-pi/2,-pi) q[4];
cx q[6],q[1];
cx q[8],q[1];
u3(pi/2,0,pi) q[8];
u2(-pi/2,-pi) q[1];
cx q[2],q[1];
cx q[4],q[1];
cx q[6],q[1];
cx q[8],q[1];
u1(1.0) q[1];
cx q[4],q[1];
u2(0,-pi/2) q[4];
cx q[4],q[1];
u1(2.0) q[1];
cx q[4],q[1];
u2(-pi/2,-pi) q[4];
cx q[4],q[1];
u1(1.0) q[1];
cx q[2],q[1];
cx q[4],q[1];
u2(0,-pi/2) q[4];
cx q[6],q[1];
cx q[8],q[1];
u2(0,-pi/2) q[1];
cx q[2],q[1];
cx q[4],q[1];
cx q[6],q[1];
u2(0,0.10110569981616457) q[6];
cx q[8],q[1];
u1(1.0) q[1];
cx q[4],q[1];
u2(-pi/2,-pi) q[4];
cx q[4],q[1];
u1(2.0) q[1];
cx q[4],q[1];
u2(0,-pi/2) q[4];
cx q[4],q[1];
u1(1.0) q[1];
cx q[2],q[1];
cx q[4],q[1];
u2(0.5083472040082082,0) q[1];
cx q[1],q[6];
u2(-pi,2.633245449581585) q[1];
u2(-0.10110569981616546,-pi) q[6];
cx q[8],q[1];
u3(pi/2,-pi/2,pi/2) q[8];
u2(-pi/2,-pi) q[1];
cx q[2],q[1];
cx q[4],q[1];
u1(1.0) q[1];
cx q[2],q[1];
cx q[4],q[1];
u2(0,-pi/2) q[1];
cx q[2],q[1];
u2(-pi/2,-pi) q[4];
cx q[4],q[1];
u1(1.0) q[1];
cx q[2],q[1];
u3(pi/2,-pi/2,pi/2) q[2];
cx q[4],q[1];
u2(-pi/2,-pi) q[1];
cx q[2],q[1];
u2(0,-pi/2) q[4];
cx q[4],q[1];
u1(1.0) q[1];
cx q[4],q[1];
u2(-pi/2,-pi) q[4];
cx q[4],q[1];
u1(1.0) q[1];
cx q[2],q[1];
u2(-pi/2,-pi) q[2];
cx q[4],q[1];
cx q[2],q[1];
u2(0,-pi/2) q[4];
cx q[4],q[1];
u1(1.0) q[1];
cx q[4],q[1];
u2(-pi/2,-pi) q[4];
cx q[4],q[1];
u1(1.0) q[1];
cx q[2],q[1];
u2(0,-pi/2) q[2];
cx q[4],q[1];
u2(0,-pi/2) q[1];
cx q[2],q[1];
u2(0,-pi/2) q[4];
cx q[4],q[1];
u1(1.0) q[1];
cx q[4],q[1];
u2(-pi/2,-pi) q[4];
cx q[4],q[1];
u1(1.0) q[1];
cx q[2],q[1];
u2(-pi/2,-pi) q[2];
cx q[4],q[1];
cx q[2],q[1];
u2(0,-pi/2) q[4];
cx q[4],q[1];
u1(1.0) q[1];
cx q[4],q[1];
u2(-pi/2,-pi) q[4];
cx q[4],q[1];
u1(1.0) q[1];
cx q[2],q[1];
u3(pi,-0.4355470098781713,1.135249316916724) q[2];
cx q[4],q[1];
cx q[1],q[0];
u2(0,-pi/2) q[4];
cx q[4],q[0];
u1(1.0) q[0];
cx q[4],q[0];
u2(-pi/2,-pi) q[4];
cx q[4],q[0];
u1(1.0) q[0];
cx q[1],q[0];
u2(-pi/2,-pi) q[1];
cx q[4],q[0];
cx q[1],q[0];
u2(0,-pi/2) q[4];
cx q[4],q[0];
u1(1.0) q[0];
cx q[4],q[0];
u2(-pi/2,-pi) q[4];
cx q[4],q[0];
u1(1.0) q[0];
cx q[1],q[0];
u2(0,-pi/2) q[1];
cx q[4],q[0];
u2(0,-pi/2) q[0];
cx q[1],q[0];
u2(0,-pi/2) q[4];
cx q[4],q[0];
u1(1.0) q[0];
cx q[4],q[0];
u2(-pi/2,-pi) q[4];
cx q[4],q[0];
u1(1.0) q[0];
cx q[1],q[0];
u2(-pi/2,-pi) q[1];
cx q[4],q[0];
cx q[1],q[0];
u2(0,-pi/2) q[4];
cx q[4],q[0];
u1(1.0) q[0];
cx q[4],q[0];
u2(-pi/2,-pi) q[4];
cx q[4],q[0];
u1(1.0) q[0];
cx q[1],q[0];
u2(-3.01403503547502,-pi) q[0];
u3(pi/2,0,pi) q[1];
u2(-pi,-0.10110569981616457) q[4];
cx q[0],q[4];
u1(-1.6983539449096696) q[0];
cx q[1],q[0];
cx q[3],q[0];
u3(3.0404869537736317,pi/2,pi/2) q[4];
cx q[8],q[0];
u1(1.0) q[0];
cx q[3],q[0];
u2(0,-pi/2) q[3];
cx q[8],q[0];
cx q[3],q[0];
u2(-pi/2,-pi) q[8];
cx q[8],q[0];
u1(1.0) q[0];
cx q[8],q[0];
u2(0,-pi/2) q[8];
cx q[8],q[0];
u1(1.0) q[0];
cx q[3],q[0];
u2(-pi/2,-pi) q[3];
cx q[8],q[0];
cx q[3],q[0];
u2(-pi/2,-pi) q[8];
cx q[8],q[0];
u1(1.0) q[0];
cx q[1],q[0];
cx q[3],q[0];
u2(0,-pi/2) q[3];
cx q[8],q[0];
u2(0,-pi/2) q[0];
cx q[1],q[0];
cx q[3],q[0];
u2(0,-pi/2) q[8];
cx q[8],q[0];
u1(1.0) q[0];
cx q[3],q[0];
u2(-pi/2,-pi) q[3];
cx q[8],q[0];
cx q[3],q[0];
u2(-pi/2,-pi) q[8];
cx q[8],q[0];
u1(1.0) q[0];
cx q[8],q[0];
u2(0,-pi/2) q[8];
cx q[8],q[0];
u1(1.0) q[0];
cx q[3],q[0];
u2(0,-pi/2) q[3];
cx q[8],q[0];
cx q[3],q[0];
u2(-pi/2,-pi) q[8];
cx q[8],q[0];
u1(1.0) q[0];
cx q[1],q[0];
cx q[3],q[0];
u2(-pi/2,-pi) q[3];
cx q[8],q[0];
u2(-pi/2,-pi) q[0];
u3(pi/2,0,pi) q[8];
cx q[1],q[0];
cx q[3],q[0];
cx q[8],q[0];
u1(1.0) q[0];
cx q[3],q[0];
u2(0,-pi/2) q[3];
cx q[3],q[0];
u1(2.0) q[0];
cx q[3],q[0];
u2(-pi/2,-pi) q[3];
cx q[3],q[0];
u1(1.0) q[0];
cx q[1],q[0];
cx q[3],q[0];
u2(0,-pi/2) q[3];
cx q[8],q[0];
u2(0,-pi/2) q[0];
cx q[1],q[0];
cx q[3],q[0];
cx q[8],q[0];
u1(1.0) q[0];
cx q[3],q[0];
u2(-pi/2,-pi) q[3];
cx q[3],q[0];
u1(2.0) q[0];
cx q[3],q[0];
u2(0,-pi/2) q[3];
cx q[3],q[0];
u1(1.0) q[0];
cx q[1],q[0];
cx q[3],q[0];
u2(-pi/2,-pi) q[3];
cx q[8],q[0];
u2(-pi/2,-pi) q[0];
cx q[1],q[0];
cx q[3],q[0];
cx q[7],q[0];
u1(1.0) q[0];
cx q[3],q[0];
u2(0,-pi/2) q[3];
cx q[3],q[0];
u1(2.0) q[0];
cx q[3],q[0];
u2(-pi/2,-pi) q[3];
cx q[3],q[0];
u1(1.0) q[0];
cx q[1],q[0];
cx q[3],q[0];
u2(0,-pi/2) q[3];
cx q[7],q[0];
u2(0,-pi/2) q[0];
cx q[1],q[0];
cx q[3],q[0];
cx q[7],q[0];
u1(1.0) q[0];
cx q[3],q[0];
u2(-pi/2,-pi) q[3];
cx q[3],q[0];
u1(2.0) q[0];
cx q[3],q[0];
u2(0,-pi/2) q[3];
cx q[3],q[0];
u1(1.0) q[0];
cx q[1],q[0];
cx q[3],q[0];
u2(-pi/2,-pi) q[3];
cx q[7],q[0];
u2(-pi/2,-pi) q[0];
cx q[1],q[0];
cx q[3],q[0];
cx q[7],q[0];
u2(-pi/2,pi/2) q[8];
u3(1.469690626978732,pi/2,pi/2) q[9];
cx q[9],q[0];
u1(1.0) q[0];
cx q[3],q[0];
u2(0,-pi/2) q[3];
cx q[9],q[0];
cx q[3],q[0];
u2(-pi/2,-pi) q[9];
cx q[9],q[0];
u1(1.0) q[0];
cx q[9],q[0];
u2(0,-pi/2) q[9];
cx q[9],q[0];
u1(1.0) q[0];
cx q[3],q[0];
u2(-pi/2,-pi) q[3];
cx q[9],q[0];
cx q[3],q[0];
u2(-pi/2,-pi) q[9];
cx q[9],q[0];
u1(1.0) q[0];
cx q[1],q[0];
cx q[3],q[0];
u2(0,-pi/2) q[3];
cx q[7],q[0];
cx q[9],q[0];
u2(0,-pi/2) q[0];
cx q[1],q[0];
cx q[3],q[0];
cx q[7],q[0];
u2(0,-pi/2) q[9];
cx q[9],q[0];
u1(1.0) q[0];
cx q[3],q[0];
u2(-pi/2,-pi) q[3];
cx q[9],q[0];
cx q[3],q[0];
u2(-pi/2,-pi) q[9];
cx q[9],q[0];
u1(1.0) q[0];
cx q[9],q[0];
u2(0,-pi/2) q[9];
cx q[9],q[0];
u1(1.0) q[0];
cx q[3],q[0];
u2(0,-pi/2) q[3];
cx q[9],q[0];
cx q[3],q[0];
u2(-pi/2,-pi) q[9];
cx q[9],q[0];
u1(1.0) q[0];
cx q[1],q[0];
cx q[3],q[0];
u2(-pi/2,-pi) q[3];
cx q[7],q[0];
u2(1.2247291581313782,-pi) q[0];
u3(pi/2,0,pi) q[7];
u2(-pi,-0.10110569981616457) q[9];
cx q[0],q[9];
u1(0.3460671686635175) q[0];
cx q[1],q[0];
cx q[3],q[0];
cx q[6],q[0];
cx q[8],q[0];
u1(1.0) q[0];
cx q[3],q[0];
u2(0,-pi/2) q[3];
cx q[8],q[0];
cx q[3],q[0];
u2(-pi/2,-pi) q[8];
cx q[8],q[0];
u1(1.0) q[0];
cx q[8],q[0];
u2(0,-pi/2) q[8];
cx q[8],q[0];
u1(1.0) q[0];
cx q[3],q[0];
u2(-pi/2,-pi) q[3];
cx q[8],q[0];
cx q[3],q[0];
u2(-pi/2,-pi) q[8];
cx q[8],q[0];
u1(1.0) q[0];
cx q[1],q[0];
cx q[3],q[0];
u2(0,-pi/2) q[3];
cx q[6],q[0];
cx q[8],q[0];
u2(0,-pi/2) q[0];
cx q[1],q[0];
cx q[3],q[0];
cx q[6],q[0];
u2(0,-pi/2) q[8];
cx q[8],q[0];
u1(1.0) q[0];
cx q[3],q[0];
u2(-pi/2,-pi) q[3];
cx q[8],q[0];
cx q[3],q[0];
u2(-pi/2,-pi) q[8];
cx q[8],q[0];
u1(1.0) q[0];
cx q[8],q[0];
u2(0,-pi/2) q[8];
cx q[8],q[0];
u1(1.0) q[0];
cx q[3],q[0];
u2(0,-pi/2) q[3];
cx q[8],q[0];
cx q[3],q[0];
u2(-pi/2,-pi) q[8];
cx q[8],q[0];
u1(1.0) q[0];
cx q[1],q[0];
cx q[3],q[0];
u2(-pi/2,-pi) q[3];
cx q[6],q[0];
cx q[8],q[0];
u2(-pi/2,-pi) q[0];
u3(pi/2,0,pi) q[8];
cx q[1],q[0];
cx q[3],q[0];
cx q[6],q[0];
cx q[8],q[0];
u1(1.0) q[0];
cx q[3],q[0];
u2(0,-pi/2) q[3];
cx q[3],q[0];
u1(2.0) q[0];
cx q[3],q[0];
u2(-pi/2,-pi) q[3];
cx q[3],q[0];
u1(1.0) q[0];
cx q[1],q[0];
cx q[3],q[0];
u2(0,-pi/2) q[3];
cx q[6],q[0];
cx q[8],q[0];
u2(0,-pi/2) q[0];
cx q[1],q[0];
cx q[3],q[0];
cx q[6],q[0];
cx q[8],q[0];
u1(1.0) q[0];
cx q[3],q[0];
u2(-pi/2,-pi) q[3];
cx q[3],q[0];
u1(2.0) q[0];
cx q[3],q[0];
u2(0,-pi/2) q[3];
cx q[3],q[0];
u1(1.0) q[0];
cx q[1],q[0];
cx q[3],q[0];
cx q[6],q[0];
u3(pi/2,0,pi) q[6];
cx q[8],q[0];
u2(-pi/2,-pi) q[0];
cx q[1],q[0];
cx q[3],q[0];
u1(1.0) q[0];
cx q[1],q[0];
cx q[3],q[0];
u2(0,-pi/2) q[0];
cx q[1],q[0];
u2(-pi/2,-pi) q[3];
cx q[3],q[0];
u1(1.0) q[0];
cx q[1],q[0];
cx q[3],q[0];
u2(-pi/2,-pi) q[0];
u3(pi/2,0,pi) q[3];
cx q[1],q[0];
cx q[3],q[0];
cx q[7],q[0];
u1(2.0) q[0];
cx q[7],q[0];
u2(0,-pi/2) q[7];
cx q[7],q[0];
u1(2.0) q[0];
cx q[1],q[0];
cx q[3],q[0];
cx q[7],q[0];
u2(0,-pi/2) q[0];
cx q[1],q[0];
cx q[3],q[0];
u2(-pi/2,-pi) q[7];
cx q[7],q[0];
u1(2.0) q[0];
cx q[7],q[0];
u2(0,-pi/2) q[7];
cx q[7],q[0];
u1(2.0) q[0];
cx q[1],q[0];
cx q[3],q[0];
cx q[7],q[0];
u2(-pi/2,-pi) q[0];
cx q[1],q[0];
cx q[3],q[0];
u2(-pi/2,-pi) q[7];
cx q[7],q[0];
u2(-pi/2,pi/2) q[8];
u3(1.4696906269787322,pi/2,pi/2) q[9];
cx q[9],q[0];
u1(1.0) q[0];
cx q[9],q[0];
u2(-pi/2,-pi) q[9];
cx q[9],q[0];
u1(1.0) q[0];
cx q[7],q[0];
u2(0,-pi/2) q[7];
cx q[9],q[0];
cx q[7],q[0];
u2(0,-pi/2) q[9];
cx q[9],q[0];
u1(1.0) q[0];
cx q[9],q[0];
u2(-pi/2,-pi) q[9];
cx q[9],q[0];
u1(1.0) q[0];
cx q[1],q[0];
cx q[3],q[0];
cx q[7],q[0];
u2(-pi/2,-pi) q[7];
cx q[9],q[0];
u2(0,-pi/2) q[0];
cx q[1],q[0];
cx q[3],q[0];
cx q[7],q[0];
u2(0,-pi/2) q[9];
cx q[9],q[0];
u1(1.0) q[0];
cx q[9],q[0];
u2(-pi/2,-pi) q[9];
cx q[9],q[0];
u1(1.0) q[0];
cx q[7],q[0];
u2(0,-pi/2) q[7];
cx q[9],q[0];
cx q[7],q[0];
u2(0,-pi/2) q[9];
cx q[9],q[0];
u1(1.0) q[0];
cx q[9],q[0];
u2(-pi/2,-pi) q[9];
cx q[9],q[0];
u1(1.0) q[0];
cx q[1],q[0];
cx q[3],q[0];
cx q[7],q[0];
u2(1.2247291581313782,-pi) q[0];
u3(-pi/2,-pi/2,pi/2) q[7];
u2(-pi,-0.10110569981616457) q[9];
cx q[0],q[9];
u1(0.3460671686635175) q[0];
cx q[1],q[0];
cx q[3],q[0];
cx q[6],q[0];
cx q[8],q[0];
u1(1.0) q[0];
cx q[8],q[0];
u2(-pi/2,-pi) q[8];
cx q[8],q[0];
u1(1.0) q[0];
cx q[6],q[0];
u2(0,-pi/2) q[6];
cx q[8],q[0];
cx q[6],q[0];
u2(0,-pi/2) q[8];
cx q[8],q[0];
u1(1.0) q[0];
cx q[8],q[0];
u2(-pi/2,-pi) q[8];
cx q[8],q[0];
u1(1.0) q[0];
cx q[1],q[0];
cx q[3],q[0];
cx q[6],q[0];
u2(-pi/2,-pi) q[6];
cx q[8],q[0];
u2(0,-pi/2) q[0];
cx q[1],q[0];
cx q[3],q[0];
cx q[6],q[0];
u2(0,-pi/2) q[8];
cx q[8],q[0];
u1(1.0) q[0];
cx q[8],q[0];
u2(-pi/2,-pi) q[8];
cx q[8],q[0];
u1(1.0) q[0];
cx q[6],q[0];
u2(0,-pi/2) q[6];
cx q[8],q[0];
cx q[6],q[0];
u2(0,-pi/2) q[8];
cx q[8],q[0];
u1(1.0) q[0];
cx q[8],q[0];
u2(-pi/2,-pi) q[8];
cx q[8],q[0];
u1(1.0) q[0];
cx q[1],q[0];
cx q[3],q[0];
cx q[6],q[0];
u2(-pi/2,-pi) q[6];
cx q[8],q[0];
u2(-pi/2,-pi) q[0];
u3(pi/2,0,pi) q[8];
cx q[1],q[0];
cx q[3],q[0];
cx q[6],q[0];
cx q[8],q[0];
u1(2.0) q[0];
cx q[6],q[0];
u2(0,-pi/2) q[6];
cx q[6],q[0];
u1(2.0) q[0];
cx q[1],q[0];
cx q[3],q[0];
cx q[6],q[0];
u2(-pi/2,-pi) q[6];
cx q[8],q[0];
u2(0,-pi/2) q[0];
cx q[1],q[0];
cx q[3],q[0];
cx q[6],q[0];
cx q[8],q[0];
u1(2.0) q[0];
cx q[6],q[0];
u2(0,-pi/2) q[6];
cx q[6],q[0];
u1(2.0) q[0];
cx q[1],q[0];
cx q[3],q[0];
cx q[6],q[0];
u3(-pi/2,-pi/2,pi/2) q[6];
u2(-1.0317207894834528,0) q[0];
u2(-pi,-0.10110569981616457) q[8];
cx q[0],q[8];
u3(pi,0.8721326299675347,-2.8085355609337017) q[0];
cx q[1],q[0];
cx q[3],q[0];
cx q[5],q[0];
cx q[7],q[0];
u1(2.0) q[0];
cx q[5],q[0];
u2(0,-pi/2) q[5];
cx q[5],q[0];
u1(2.0) q[0];
cx q[1],q[0];
cx q[3],q[0];
cx q[5],q[0];
u2(-pi/2,-pi) q[5];
cx q[7],q[0];
u2(0,-pi/2) q[0];
cx q[1],q[0];
cx q[3],q[0];
cx q[5],q[0];
cx q[7],q[0];
u1(2.0) q[0];
cx q[5],q[0];
u2(0,-pi/2) q[5];
cx q[5],q[0];
u1(2.0) q[0];
cx q[1],q[0];
cx q[3],q[0];
cx q[5],q[0];
u2(-pi/2,-pi) q[5];
cx q[7],q[0];
u2(-pi/2,-pi) q[0];
cx q[1],q[0];
cx q[3],q[0];
cx q[5],q[0];
cx q[7],q[0];
u3(1.470208709471339,-0.01023951313391791,1.672417449869001) q[8];
u3(1.4696906269787322,pi/2,pi/2) q[9];
cx q[9],q[0];
u1(1.0) q[0];
cx q[9],q[0];
u2(-pi/2,-pi) q[9];
cx q[9],q[0];
u1(1.0) q[0];
cx q[5],q[0];
u2(0,-pi/2) q[5];
cx q[9],q[0];
cx q[5],q[0];
u2(0,-pi/2) q[9];
cx q[9],q[0];
u1(1.0) q[0];
cx q[9],q[0];
u2(-pi/2,-pi) q[9];
cx q[9],q[0];
u1(1.0) q[0];
cx q[1],q[0];
cx q[3],q[0];
cx q[5],q[0];
u2(-pi/2,-pi) q[5];
cx q[7],q[0];
cx q[9],q[0];
u2(0,-pi/2) q[0];
cx q[1],q[0];
cx q[3],q[0];
cx q[5],q[0];
cx q[7],q[0];
u2(0,-pi/2) q[9];
cx q[9],q[0];
u1(1.0) q[0];
cx q[9],q[0];
u2(-pi/2,-pi) q[9];
cx q[9],q[0];
u1(1.0) q[0];
cx q[5],q[0];
u2(0,-pi/2) q[5];
cx q[9],q[0];
cx q[5],q[0];
u2(0,-pi/2) q[9];
cx q[9],q[0];
u1(1.0) q[0];
cx q[9],q[0];
u2(-pi/2,-pi) q[9];
cx q[9],q[0];
u1(1.0) q[0];
cx q[1],q[0];
cx q[3],q[0];
cx q[5],q[0];
u3(-pi/2,-pi/2,pi/2) q[5];
cx q[7],q[0];
u2(pi/2,0) q[7];
cx q[7],q[8];
u3(2.1415926535897936,0,pi/2) q[7];
u3(1.5246179561159006,1.4856702783578895,-2.5645270074069986) q[8];
cx q[7],q[8];
u2(0,-pi) q[7];
u3(3.040486953773628,-pi/2,pi/2) q[8];
cx q[9],q[0];
u2(-pi/2,-pi) q[0];
u3(pi/2,0,pi) q[9];
cx q[1],q[0];
cx q[3],q[0];
u1(1.0) q[0];
cx q[1],q[0];
cx q[3],q[0];
u2(0,-pi/2) q[0];
cx q[1],q[0];
cx q[3],q[0];
u1(1.0) q[0];
cx q[1],q[0];
cx q[3],q[0];
u2(-pi/2,-pi) q[0];
u3(pi/2,-pi/2,pi/2) q[3];
cx q[1],q[0];
cx q[3],q[0];
u1(1.0) q[0];
cx q[3],q[0];
u2(-pi/2,-pi) q[3];
cx q[3],q[0];
u1(2.0) q[0];
cx q[3],q[0];
u2(0,-pi/2) q[3];
cx q[3],q[0];
u1(1.0) q[0];
cx q[1],q[0];
cx q[3],q[0];
u2(0,-pi/2) q[0];
cx q[1],q[0];
u2(-pi/2,-pi) q[3];
cx q[3],q[0];
u1(1.0) q[0];
cx q[3],q[0];
u2(0,-pi/2) q[3];
cx q[3],q[0];
u1(2.0) q[0];
cx q[3],q[0];
u2(-pi/2,-pi) q[3];
cx q[3],q[0];
u1(1.0) q[0];
cx q[1],q[0];
cx q[3],q[0];
u3(-pi/2,-pi/2,pi/2) q[0];
u2(0,-1.469690626978732) q[3];
cx q[2],q[3];
u3(2.1415926535897936,0,pi/2) q[2];
u3(1.5246179561159006,1.4856702783578895,-2.5645270074069986) q[3];
cx q[2],q[3];
u2(0,-pi) q[2];
u3(3.040486953773628,-pi/2,pi/2) q[3];
