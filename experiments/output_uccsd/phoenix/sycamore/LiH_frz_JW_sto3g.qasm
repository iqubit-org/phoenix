OPENQASM 2.0;
include "qelib1.inc";
qreg q[54];
u2(0,pi) q[40];
u2(0,pi) q[41];
u2(0,pi) q[42];
cx q[41],q[42];
cx q[41],q[40];
u2(0,pi) q[46];
u2(0,pi) q[47];
cx q[41],q[47];
cx q[47],q[41];
cx q[41],q[47];
u2(0,pi) q[48];
cx q[47],q[48];
cx q[47],q[46];
cx q[47],q[41];
u3(pi,0,pi) q[50];
u2(0,pi) q[51];
cx q[50],q[51];
cx q[51],q[50];
cx q[50],q[51];
u1(-pi/2) q[50];
cx q[51],q[47];
u3(0.101105699816165,-pi/2,0) q[51];
cx q[50],q[51];
u2(0,pi/2) q[50];
u3(1.4702087094713374,1.4691752037207921,-3.1313531404558756) q[51];
cx q[50],q[51];
u2(pi/2,0) q[50];
u3(1.098595559009321,-pi,-pi/2) q[51];
u3(1.6719020266110614,-pi/2,-0.9974898591931565) q[52];
cx q[51],q[52];
u2(0,pi/2) q[51];
u3(1.4702087094713374,1.4691752037207921,-3.1313531404558756) q[52];
cx q[51],q[52];
u3(0.9974898591931551,pi/2,-pi/2) q[51];
u3(1.4858988984718022,-3.0866194354550656,0.9951543270171284) q[52];
u1(-pi/2) q[53];
cx q[51],q[53];
u2(0,-pi) q[51];
cx q[52],q[51];
u3(0.02499999999999999,0,0) q[51];
cx q[50],q[51];
cx q[51],q[50];
cx q[50],q[51];
u1(-pi/2) q[50];
u3(1.4696906269787318,pi/2,-pi) q[52];
u1(pi/2) q[53];
cx q[51],q[53];
u2(-pi/2,-pi/2) q[51];
cx q[51],q[52];
u2(0,pi/2) q[51];
u3(1.4702087094713374,1.4691752037207921,-3.1313531404558756) q[52];
cx q[51],q[52];
u2(-pi/2,-pi) q[51];
u3(1.4453380362902235,1.4688869109232243,-3.1287971551894294) q[52];
u2(0,pi) q[53];
cx q[51],q[53];
u1(0.024999999999999998) q[53];
cx q[51],q[53];
u2(pi/2,-pi) q[51];
cx q[51],q[52];
u2(0,pi/2) q[51];
u3(1.4702087094713374,1.4691752037207921,-3.1313531404558756) q[52];
cx q[51],q[52];
u2(-pi,-0.024999999999999467) q[51];
u3(0.1011056998161646,-pi/2,-pi/2) q[52];
u2(0,pi) q[53];
cx q[51],q[53];
u3(0.6502796335185623,-0.1337742990928894,-1.4032945865224777) q[51];
cx q[50],q[51];
u2(0,pi/2) q[50];
u3(1.4702087094713374,1.4691752037207921,-3.1313531404558756) q[51];
cx q[50],q[51];
u2(2.214297435588181,0) q[50];
u3(1.5956686296756561,-3.0404555650491405,0.0025238587510041377) q[51];
cx q[52],q[51];
u3(0.02499999999999999,0,0) q[51];
cx q[50],q[51];
u2(0,pi) q[50];
u3(1.5957963267948965,0,-pi) q[51];
u2(0,pi) q[52];
u3(0.02499999999999999,0,-pi/2) q[53];
cx q[51],q[53];
u2(-pi/2,pi/2) q[51];
cx q[52],q[51];
u3(1.0303768265243223,-pi,-pi/2) q[51];
cx q[50],q[51];
cx q[51],q[50];
cx q[50],q[51];
cx q[51],q[52];
u2(0,pi) q[51];
cx q[51],q[47];
cx q[47],q[41];
u2(-pi/2,-pi) q[41];
cx q[40],q[41];
cx q[41],q[40];
cx q[40],q[41];
cx q[47],q[46];
u3(0.4000876836355586,-2.89933288623853,-1.832924640594726) q[46];
cx q[47],q[48];
cx q[47],q[41];
u3(0.4000876836355586,-2.89933288623853,-1.832924640594726) q[41];
u3(0.5488714528705644,-0.1666621289768191,1.7654851592281329) q[47];
u3(0.5488714528705644,-0.1666621289768191,1.7654851592281329) q[48];
cx q[50],q[46];
cx q[46],q[50];
cx q[50],q[46];
u3(pi/2,pi/2,2.111215827065471) q[51];
cx q[51],q[47];
u3(1.5702946546377712,1.5607196157102798,-1.6702839605185331) q[47];
u3(3.041592653589794,0,pi/2) q[51];
cx q[51],q[47];
u3(1.657454537221788,-0.052149106524069566,-2.1134773278793375) q[47];
u3(pi/2,-2.1112158270654713,-pi) q[51];
cx q[51],q[47];
cx q[48],q[47];
cx q[47],q[48];
cx q[48],q[47];
cx q[46],q[47];
u3(3.041592653589794,0,pi/2) q[46];
u3(1.5702946546377712,1.5607196157102798,-1.6702839605185331) q[47];
cx q[46],q[47];
u3(pi/2,-2.1112158270654713,-pi) q[46];
u3(1.657454537221788,-0.052149106524069566,-2.1134773278793375) q[47];
cx q[46],q[47];
u3(pi/2,pi/2,1.1830261549195793) q[46];
cx q[46],q[50];
u3(3.0415926535897935,0,pi/2) q[46];
u3(0.1011056998161648,-pi/2,-pi) q[48];
u3(1.5702946546377712,1.5607196157102798,-1.6702839605185333) q[50];
cx q[46],q[50];
u3(pi/2,-1.1830261549195793,-pi) q[46];
cx q[46],q[47];
u2(0,pi) q[46];
u2(-pi/2,-pi) q[47];
u3(0.10110569981617738,1.9585664986702307,-pi/2) q[50];
u3(pi/2,pi/2,1.1830261549195793) q[51];
cx q[51],q[47];
cx q[47],q[51];
cx q[51],q[47];
cx q[47],q[41];
u3(1.5702946546377712,1.5607196157102798,-1.6702839605185333) q[41];
u3(3.0415926535897935,0,pi/2) q[47];
cx q[47],q[41];
u3(1.6643725059626524,0.03834260415838742,-1.1812306479977877) q[41];
u3(0.3877701718753174,0,pi/2) q[47];
cx q[47],q[48];
u2(0,pi/2) q[47];
u3(1.4702087094713374,1.4691752037207921,-3.1313531404558756) q[48];
cx q[47],q[48];
u2(-pi/2,pi/2) q[47];
cx q[47],q[41];
u3(0.10110569981616474,0,-pi/2) q[48];
cx q[48],q[47];
cx q[46],q[47];
cx q[47],q[46];
cx q[46],q[47];
cx q[47],q[51];
u2(-pi/2,-pi) q[48];
cx q[51],q[47];
cx q[47],q[51];
u2(-pi,-0.10110569981616413) q[52];
cx q[48],q[52];
u3(3.0415926535897935,0,pi/2) q[48];
u3(1.4702087094713374,1.4691752037207921,-3.1313531404558756) q[52];
cx q[48],q[52];
u3(pi,pi/2,-pi) q[48];
u3(0.0011056998161648977,-1.6357137866407356e-11,-1.5707963267785396) q[52];
cx q[52],q[51];
u2(pi/2,-pi/2) q[51];
cx q[51],q[47];
u1(pi/2) q[47];
cx q[48],q[47];
u2(0,pi) q[47];
u2(0,-pi) q[51];
u2(0,pi) q[52];
cx q[52],q[51];
u3(0.02499999999999999,0,0) q[51];
u2(0,pi) q[52];
cx q[48],q[52];
u3(0.025000000000000116,pi/2,-pi/2) q[48];
cx q[47],q[48];
cx q[48],q[47];
cx q[47],q[48];
u2(0,pi) q[52];
cx q[52],q[48];
u1(0.024999999999999998) q[48];
cx q[52],q[48];
u2(0,pi) q[48];
u2(0,pi) q[52];
cx q[52],q[51];
cx q[51],q[52];
cx q[52],q[51];
cx q[47],q[51];
u3(0.024999999999999873,-pi/2,pi/2) q[47];
cx q[47],q[48];
u2(0,pi) q[47];
u3(0.02499999999999999,0,-pi/2) q[48];
u2(0,pi) q[51];
cx q[52],q[51];
cx q[51],q[52];
cx q[52],q[51];
cx q[47],q[51];
u2(pi/2,-pi/2) q[47];
u3(0.02499999999999999,-pi,-pi) q[51];
cx q[52],q[51];
u3(0.0761056998161648,-pi/2,-pi) q[51];
cx q[47],q[51];
u2(0,pi/2) q[47];
u3(1.4702087094713374,1.4691752037207921,-3.1313531404558756) q[51];
cx q[47],q[51];
u2(pi/2,-3.1165926535897936) q[47];
cx q[47],q[48];
u2(-pi/2,pi/2) q[47];
u3(1.469690626978732,0,pi/2) q[51];
cx q[47],q[51];
cx q[51],q[47];
cx q[47],q[51];
u2(0,pi) q[52];
cx q[52],q[51];
cx q[52],q[48];
cx q[48],q[52];
cx q[52],q[48];
cx q[47],q[48];
u2(0,pi) q[47];
cx q[47],q[46];
cx q[41],q[47];
cx q[47],q[41];
cx q[41],q[47];
cx q[46],q[47];
cx q[46],q[40];
cx q[40],q[46];
cx q[46],q[40];
cx q[41],q[40];
u1(-pi/2) q[40];
u2(0,pi) q[41];
cx q[41],q[47];
cx q[47],q[51];
u2(0,pi) q[47];
u2(pi/2,-pi/2) q[51];
cx q[51],q[52];
u2(0,-pi) q[51];
cx q[47],q[51];
u2(0,pi) q[47];
u3(0.02499999999999999,0,0) q[51];
u1(pi/2) q[52];
cx q[52],q[48];
cx q[48],q[52];
cx q[52],q[48];
cx q[48],q[42];
cx q[42],q[48];
cx q[48],q[42];
cx q[41],q[42];
cx q[41],q[47];
u3(0.025000000000000116,pi/2,-pi/2) q[41];
u2(0,pi) q[42];
cx q[42],q[48];
u2(0,pi) q[47];
cx q[48],q[42];
cx q[42],q[48];
cx q[47],q[48];
u1(0.024999999999999998) q[48];
cx q[47],q[48];
u2(0,pi) q[47];
cx q[41],q[47];
u3(0.024999999999999873,-pi/2,pi/2) q[41];
u2(0,pi) q[47];
u2(0,pi) q[48];
cx q[48],q[42];
cx q[42],q[48];
cx q[48],q[42];
cx q[41],q[42];
u2(0,pi) q[41];
u3(0.02499999999999999,0,-pi/2) q[42];
cx q[42],q[48];
cx q[48],q[42];
cx q[42],q[48];
cx q[51],q[47];
cx q[47],q[51];
cx q[51],q[47];
cx q[41],q[47];
u3(0.02499999999999999,-pi,-pi) q[47];
cx q[51],q[47];
u3(0.02499999999999999,0,0) q[47];
cx q[41],q[47];
u2(0,pi) q[41];
u3(1.5957963267948965,0,-pi) q[47];
cx q[47],q[48];
u2(-pi,pi/2) q[47];
u2(0,-pi/2) q[48];
cx q[48],q[42];
cx q[42],q[48];
cx q[48],q[42];
u3(0.101105699816165,-pi/2,0) q[51];
cx q[47],q[51];
u2(0,pi/2) q[47];
u3(1.4702087094713374,1.4691752037207921,-3.1313531404558756) q[51];
cx q[47],q[51];
u2(pi/2,0) q[47];
cx q[41],q[47];
u3(0.101105699816165,-pi/2,0) q[41];
cx q[40],q[41];
u2(0,pi/2) q[40];
u3(1.4702087094713374,1.4691752037207921,-3.1313531404558756) q[41];
cx q[40],q[41];
u2(pi/2,0) q[40];
u2(-3.0404869537736285,0) q[41];
cx q[41],q[42];
cx q[40],q[41];
u2(0,pi) q[40];
cx q[40],q[46];
u1(-pi/2) q[41];
cx q[46],q[40];
cx q[40],q[46];
cx q[46],q[47];
u2(-3.0404869537736285,0) q[51];
cx q[47],q[51];
u2(0,pi) q[47];
u2(pi/2,-pi/2) q[51];
cx q[51],q[50];
u1(pi/2) q[50];
cx q[46],q[50];
u2(-pi/2,-pi/2) q[46];
u2(0,pi) q[50];
u2(0,-pi) q[51];
cx q[47],q[51];
u3(1.4696906269787318,pi/2,-pi) q[47];
cx q[46],q[47];
u2(0,pi/2) q[46];
u3(1.4702087094713374,1.4691752037207921,-3.1313531404558756) q[47];
cx q[46],q[47];
u2(-pi/2,-pi) q[46];
cx q[46],q[50];
u3(3.0374522131493493,-2.90000773437424,1.8136462106786775) q[47];
u1(0.024999999999999998) q[50];
cx q[46],q[50];
u2(0,pi) q[46];
cx q[47],q[46];
u2(0,pi) q[46];
u3(0.024999999999999873,-pi/2,pi/2) q[47];
u2(0,pi) q[50];
u3(0.02499999999999999,0,0) q[51];
cx q[50],q[51];
cx q[51],q[50];
cx q[50],q[51];
cx q[47],q[51];
u2(0,pi) q[47];
cx q[50],q[46];
cx q[46],q[50];
cx q[50],q[46];
cx q[47],q[46];
u3(0.02499999999999999,-pi,-pi) q[46];
cx q[50],q[46];
u3(0.02499999999999999,0,0) q[46];
cx q[47],q[46];
u3(1.5957963267948965,0,-pi) q[46];
u2(0,pi) q[47];
u2(0,pi) q[50];
cx q[46],q[50];
cx q[50],q[46];
cx q[46],q[50];
u3(0.02499999999999999,0,-pi/2) q[51];
cx q[50],q[51];
u2(-pi/2,pi/2) q[50];
cx q[46],q[50];
cx q[47],q[46];
u3(0.101105699816165,-pi/2,0) q[47];
cx q[41],q[47];
u2(0,pi/2) q[41];
u3(1.4702087094713374,1.4691752037207921,-3.1313531404558756) q[47];
cx q[41],q[47];
u3(pi,0,pi/2) q[41];
u2(-3.0404869537736285,0) q[47];
u3(0.10110569981616482,-pi/2,0) q[50];
u2(0,-pi/2) q[51];
cx q[47],q[51];
u3(0.1011056998161648,-pi/2,-pi) q[47];
cx q[41],q[47];
u2(0,pi/2) q[41];
u3(1.4702087094713374,1.4691752037207921,-3.1313531404558756) q[47];
cx q[41],q[47];
u2(pi/2,-pi/2) q[41];
u3(1.6719020266110614,-pi,-pi/2) q[47];
cx q[47],q[46];
u2(pi/2,-pi/2) q[46];
cx q[46],q[50];
u2(0,pi/2) q[46];
u3(1.4702087094713374,1.4691752037207921,-3.1313531404558756) q[50];
cx q[46],q[50];
u3(pi,0.9592151556974544,0.9592151556974553) q[46];
cx q[46],q[40];
u1(pi/2) q[40];
u2(0,-pi) q[46];
u3(1.6719020266110614,0,-pi/2) q[50];
cx q[50],q[46];
u3(0.02499999999999999,0,0) q[46];
cx q[47],q[46];
cx q[46],q[47];
cx q[47],q[46];
cx q[46],q[40];
u2(0,pi) q[40];
u2(-pi/2,-pi/2) q[46];
u3(0.10110569981616448,-pi/2,0) q[47];
u3(1.4696906269787318,pi/2,-pi) q[50];
cx q[46],q[50];
u2(0,pi/2) q[46];
u3(1.4702087094713374,1.4691752037207921,-3.1313531404558756) q[50];
cx q[46],q[50];
u2(-pi/2,-pi) q[46];
cx q[46],q[40];
u1(0.024999999999999998) q[40];
cx q[46],q[40];
u2(0,pi) q[40];
u2(pi/2,-pi) q[46];
u3(1.4453380362902235,1.4688869109232243,-3.1287971551894294) q[50];
cx q[46],q[50];
u2(0,pi/2) q[46];
u3(1.4702087094713374,1.4691752037207921,-3.1313531404558756) q[50];
cx q[46],q[50];
u2(-pi,-0.024999999999999467) q[46];
cx q[46],q[40];
u3(0.02499999999999999,0,-pi/2) q[40];
u3(1.1071487177940904,0,-pi/2) q[46];
cx q[46],q[47];
u2(0,pi/2) q[46];
u3(1.4702087094713374,1.4691752037207921,-3.1313531404558756) q[47];
cx q[46],q[47];
u2(-pi/2,-1.5457963267948966) q[46];
u3(3.0404869537736285,2.6779450445889843,pi/2) q[47];
u3(0.1011056998161646,-pi/2,-pi/2) q[50];
cx q[50],q[46];
u3(0.02499999999999999,0,0) q[46];
cx q[47],q[46];
u3(1.5957963267948965,0,-pi) q[46];
cx q[46],q[40];
u2(0,-pi/2) q[40];
u2(-pi,pi/2) q[46];
u2(0,pi) q[47];
u3(0.101105699816165,-pi/2,0) q[50];
cx q[46],q[50];
u2(0,pi/2) q[46];
u3(1.4702087094713374,1.4691752037207921,-3.1313531404558756) q[50];
cx q[46],q[50];
u2(pi/2,0) q[46];
cx q[47],q[46];
u2(0,pi) q[47];
cx q[47],q[41];
cx q[41],q[40];
cx q[47],q[41];
u2(0,pi) q[47];
cx q[47],q[46];
u2(-3.0404869537736285,0) q[50];
cx q[46],q[50];
u2(0,pi) q[46];
u2(pi/2,-pi/2) q[50];
cx q[53],q[51];
cx q[51],q[53];
cx q[53],q[51];
cx q[50],q[51];
u2(0,-pi) q[50];
cx q[46],q[50];
u2(0,pi) q[46];
u3(0.02499999999999999,0,0) q[50];
u1(pi/2) q[51];
cx q[47],q[51];
cx q[47],q[46];
u2(0,pi) q[46];
cx q[46],q[50];
u3(0.025000000000000116,pi/2,-pi/2) q[47];
cx q[50],q[46];
cx q[46],q[50];
u2(-pi/2,pi/2) q[50];
u3(3.040486953773628,pi/2,-pi) q[51];
cx q[50],q[51];
u2(-pi/2,-pi/2) q[50];
u3(1.5809840807701345,-3.040999798350607,-0.10059285523918593) q[51];
cx q[50],q[51];
u3(1.5457963267948958,0,-pi/2) q[50];
u3(1.6713839441184557,0.01023951313391791,-1.4691752037207917) q[51];
cx q[50],q[51];
u1(-pi/2) q[50];
u3(3.0404869537736285,pi/2,pi/2) q[51];
cx q[47],q[51];
u3(0.024999999999999873,-pi/2,pi/2) q[47];
u2(0,pi) q[51];
cx q[50],q[51];
cx q[51],q[50];
cx q[50],q[51];
cx q[47],q[51];
u2(0,pi) q[47];
cx q[47],q[46];
u3(0.02499999999999999,-pi,-pi) q[46];
cx q[50],q[46];
u3(0.02499999999999999,0,0) q[46];
cx q[47],q[46];
u3(1.5957963267948965,0,-pi) q[46];
u2(0,pi) q[47];
u2(0,pi) q[50];
u3(0.02499999999999999,0,-pi/2) q[51];
cx q[51],q[50];
cx q[50],q[51];
cx q[51],q[50];
cx q[46],q[50];
u2(-pi/2,pi/2) q[46];
u3(1.4696906269787187,pi/2,-1.9585664986702176) q[50];
cx q[51],q[50];
cx q[50],q[51];
cx q[51],q[50];
cx q[50],q[46];
u2(0,pi) q[46];
cx q[50],q[51];
cx q[51],q[50];
cx q[50],q[51];
cx q[47],q[51];
u2(0,pi) q[47];
cx q[47],q[41];
cx q[41],q[40];
u3(0.4000876836355586,-2.89933288623853,-1.832924640594726) q[40];
cx q[41],q[47];
cx q[47],q[41];
cx q[41],q[47];
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[41];
cx q[53],q[51];
cx q[51],q[53];
cx q[53],q[51];
cx q[47],q[51];
cx q[47],q[41];
cx q[47],q[48];
cx q[41],q[47];
cx q[47],q[41];
cx q[41],q[47];
cx q[42],q[41];
u2(-pi/2,-pi) q[42];
cx q[47],q[51];
cx q[46],q[47];
u3(pi/2,pi/2,1.1830261549195793) q[46];
cx q[46],q[40];
u3(1.5702946546377712,1.5607196157102798,-1.6702839605185333) q[40];
u3(3.0415926535897935,0,pi/2) q[46];
cx q[46],q[40];
u3(1.6643725059626524,0.03834260415838742,-1.1812306479977877) q[40];
cx q[40],q[41];
cx q[41],q[40];
cx q[40],q[41];
u3(pi/2,-1.1830261549195793,-pi) q[46];
cx q[46],q[47];
cx q[47],q[41];
cx q[46],q[47];
u3(pi/2,pi/2,1.1830261549195793) q[46];
cx q[46],q[50];
u3(3.0415926535897935,0,pi/2) q[46];
u2(-pi,-0.10110569981616413) q[48];
cx q[42],q[48];
u3(3.0415926535897935,0,pi/2) q[42];
u3(1.4702087094713374,1.4691752037207921,-3.1313531404558756) q[48];
cx q[42],q[48];
u3(pi,pi/2,-pi) q[42];
u3(0.0011056998161648977,-1.6357137866407356e-11,-1.5707963267785396) q[48];
u3(1.5702946546377712,1.5607196157102798,-1.6702839605185333) q[50];
cx q[46],q[50];
u3(pi/2,-1.1830261549195793,-pi) q[46];
cx q[46],q[47];
u2(0,pi) q[46];
cx q[47],q[41];
u2(-pi/2,-pi) q[41];
cx q[47],q[51];
u2(-pi/2,-pi) q[47];
cx q[48],q[47];
cx q[47],q[48];
cx q[48],q[47];
cx q[47],q[46];
u2(pi/2,-pi/2) q[46];
u2(0,pi) q[47];
cx q[48],q[47];
cx q[47],q[48];
cx q[48],q[47];
cx q[46],q[47];
u2(0,-pi) q[46];
u1(pi/2) q[47];
cx q[47],q[48];
cx q[48],q[47];
cx q[47],q[48];
cx q[42],q[48];
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[41];
cx q[47],q[46];
u3(0.02499999999999999,0,0) q[46];
u2(0,pi) q[47];
cx q[41],q[47];
u3(0.025000000000000116,pi/2,-pi/2) q[41];
u2(0,pi) q[47];
u2(0,pi) q[48];
cx q[47],q[48];
u1(0.024999999999999998) q[48];
cx q[47],q[48];
u2(0,pi) q[47];
cx q[41],q[47];
u3(0.024999999999999873,-pi/2,pi/2) q[41];
u2(0,pi) q[47];
u2(0,pi) q[48];
cx q[48],q[47];
cx q[47],q[48];
cx q[48],q[47];
cx q[41],q[47];
u2(0,pi) q[41];
u3(0.02499999999999999,0,-pi/2) q[47];
cx q[46],q[47];
cx q[47],q[46];
cx q[46],q[47];
cx q[41],q[47];
u3(0.02499999999999999,-pi,-pi) q[47];
cx q[48],q[47];
u3(0.02499999999999999,0,0) q[47];
cx q[41],q[47];
u2(0,pi) q[41];
u3(1.5957963267948965,0,-pi) q[47];
cx q[47],q[46];
u2(0,-pi/2) q[46];
u2(-pi/2,pi/2) q[47];
u2(0,pi) q[48];
cx q[48],q[47];
cx q[46],q[47];
cx q[47],q[46];
cx q[46],q[47];
cx q[47],q[51];
cx q[48],q[47];
cx q[47],q[48];
cx q[41],q[47];
u2(0,pi) q[41];
cx q[41],q[40];
cx q[47],q[48];
cx q[48],q[47];
cx q[41],q[47];
u2(0,pi) q[41];
cx q[41],q[40];
cx q[40],q[46];
u2(0,pi) q[40];
cx q[42],q[48];
u2(pi/2,-pi/2) q[46];
cx q[48],q[42];
cx q[42],q[48];
cx q[48],q[47];
cx q[47],q[48];
cx q[48],q[47];
cx q[46],q[47];
u2(0,-pi) q[46];
cx q[40],q[46];
u2(0,pi) q[40];
u3(0.02499999999999999,0,0) q[46];
u1(pi/2) q[47];
cx q[41],q[47];
cx q[41],q[40];
u2(0,pi) q[40];
u3(0.025000000000000116,pi/2,-pi/2) q[41];
u2(0,pi) q[47];
cx q[47],q[41];
cx q[41],q[47];
cx q[47],q[41];
cx q[40],q[41];
u1(0.024999999999999998) q[41];
cx q[40],q[41];
u2(0,pi) q[40];
cx q[40],q[46];
u2(0,pi) q[41];
cx q[46],q[40];
cx q[40],q[46];
cx q[47],q[46];
u2(0,pi) q[46];
cx q[40],q[46];
cx q[46],q[40];
cx q[40],q[46];
u3(0.024999999999999873,-pi/2,pi/2) q[47];
cx q[47],q[41];
u3(0.02499999999999999,0,-pi/2) q[41];
u2(0,pi) q[47];
cx q[47],q[46];
u3(0.02499999999999999,-pi,-pi) q[46];
cx q[40],q[46];
u2(0,pi) q[40];
u3(0.02499999999999999,0,0) q[46];
cx q[47],q[46];
u3(1.5957963267948965,0,-pi) q[46];
cx q[46],q[40];
cx q[40],q[46];
cx q[46],q[40];
cx q[40],q[41];
u2(-pi/2,pi/2) q[40];
u2(0,-pi/2) q[41];
cx q[41],q[42];
cx q[42],q[41];
cx q[41],q[42];
cx q[46],q[40];
u1(pi/2) q[40];
u2(0,pi) q[47];
cx q[47],q[46];
u2(0,pi) q[47];
cx q[47],q[48];
cx q[48],q[42];
cx q[47],q[48];
u2(0,pi) q[47];
cx q[47],q[46];
u3(1.6719020266110614,-pi/2,pi/2) q[46];
cx q[40],q[46];
u2(0,pi/2) q[40];
u3(1.4702087094713374,1.4691752037207921,-3.1313531404558756) q[46];
cx q[40],q[46];
u1(-pi) q[40];
u3(1.6719020266110611,0,-pi/2) q[46];
u3(0.10110569981617738,1.9585664986702307,-pi/2) q[50];
cx q[46],q[50];
u2(0,-pi) q[46];
cx q[40],q[46];
u2(-pi/2,-pi) q[40];
u3(0.02499999999999999,0,0) q[46];
cx q[47],q[46];
cx q[46],q[47];
cx q[47],q[46];
u3(0.10110569981616448,-pi/2,0) q[47];
u1(pi/2) q[50];
cx q[46],q[50];
u3(1.4696906269787318,pi/2,0) q[46];
cx q[40],q[46];
u2(0,pi/2) q[40];
u3(1.4702087094713374,1.4691752037207921,-3.1313531404558756) q[46];
cx q[40],q[46];
u3(0.7857105983190388,3.106244676545364,1.5957885179548974) q[40];
u3(3.0404869537736285,-pi/2,pi/2) q[46];
u2(0,pi) q[50];
cx q[46],q[50];
u1(0.024999999999999998) q[50];
cx q[46],q[50];
u3(1.4696906269787318,pi/2,-pi) q[46];
cx q[40],q[46];
u2(0,pi/2) q[40];
u3(1.4702087094713374,1.4691752037207921,-3.1313531404558756) q[46];
cx q[40],q[46];
u2(-pi/2,-pi) q[40];
u3(3.0563185593676736,-0.5772388958256918,1.779849659666052) q[46];
u2(0,pi) q[50];
cx q[46],q[50];
u3(1.1071487177940904,0,-pi/2) q[46];
cx q[46],q[47];
u2(0,pi/2) q[46];
u3(1.4702087094713374,1.4691752037207921,-3.1313531404558756) q[47];
cx q[46],q[47];
u2(-pi/2,-1.5457963267948966) q[46];
cx q[40],q[46];
u2(0,pi) q[40];
u3(0.02499999999999999,0,0) q[46];
u3(3.0404869537736285,2.6779450445889843,pi/2) q[47];
cx q[47],q[46];
u3(1.5957963267948965,0,-pi) q[46];
u2(0,pi) q[47];
u3(0.02499999999999999,0,-pi/2) q[50];
cx q[46],q[50];
u2(-pi/2,pi/2) q[46];
cx q[40],q[46];
cx q[40],q[41];
cx q[41],q[40];
cx q[40],q[41];
u1(pi/2) q[46];
cx q[47],q[41];
u1(-pi/2) q[41];
u2(0,pi) q[47];
cx q[47],q[48];
u3(0.101105699816165,-pi/2,0) q[47];
cx q[41],q[47];
u2(0,pi/2) q[41];
u3(1.4702087094713374,1.4691752037207921,-3.1313531404558756) q[47];
cx q[41],q[47];
u2(pi/2,0) q[41];
u3(1.671383944118456,3.131353140455875,-1.6724174498690019) q[47];
cx q[46],q[47];
u2(0,pi/2) q[46];
u3(1.4702087094713374,1.4691752037207921,-3.1313531404558756) q[47];
cx q[46],q[47];
u1(-pi) q[46];
u3(1.6719020266110611,0,-pi/2) q[47];
cx q[48],q[42];
u2(-pi/2,-pi) q[42];
cx q[51],q[52];
cx q[52],q[51];
cx q[51],q[52];
cx q[48],q[52];
u2(-pi/2,-pi) q[48];
cx q[47],q[48];
u2(0,-pi) q[47];
cx q[46],q[47];
u2(-pi/2,-pi) q[46];
u3(0.02499999999999999,0,0) q[47];
cx q[41],q[47];
cx q[47],q[41];
cx q[41],q[47];
u1(-pi/2) q[41];
u1(pi/2) q[48];
cx q[47],q[48];
u3(1.4696906269787318,pi/2,0) q[47];
cx q[46],q[47];
u2(0,pi/2) q[46];
u3(1.4702087094713374,1.4691752037207921,-3.1313531404558756) q[47];
cx q[46],q[47];
u3(0.7857105983190388,3.106244676545364,1.5957885179548974) q[46];
u3(3.0404869537736285,-pi/2,pi/2) q[47];
u2(0,pi) q[48];
cx q[47],q[48];
u1(0.024999999999999998) q[48];
cx q[47],q[48];
u3(1.4696906269787318,pi/2,-pi) q[47];
cx q[46],q[47];
u2(0,pi/2) q[46];
u3(1.4702087094713374,1.4691752037207921,-3.1313531404558756) q[47];
cx q[46],q[47];
u2(-pi/2,-pi) q[46];
u3(3.0563185593676736,-0.5772388958256918,1.779849659666052) q[47];
u2(0,pi) q[48];
cx q[47],q[48];
u3(0.6502796335185623,-0.1337742990928894,-1.4032945865224777) q[47];
cx q[41],q[47];
u2(0,pi/2) q[41];
u3(1.4702087094713374,1.4691752037207921,-3.1313531404558756) q[47];
cx q[41],q[47];
u2(2.214297435588181,0) q[41];
u3(1.5956686296756561,-3.0404555650491405,0.0025238587510041377) q[47];
cx q[46],q[47];
u2(0,pi) q[46];
u3(0.02499999999999999,0,0) q[47];
cx q[41],q[47];
u2(0,pi) q[41];
u3(1.5957963267948965,0,-pi) q[47];
u3(0.02499999999999999,0,-pi/2) q[48];
cx q[47],q[48];
u2(-pi/2,pi/2) q[47];
cx q[46],q[47];
cx q[41],q[47];
cx q[47],q[41];
cx q[41],q[47];
u1(pi/2) q[41];
cx q[47],q[46];
u1(-pi/2) q[46];
u2(0,pi) q[47];
u2(0,-pi/2) q[48];
cx q[47],q[48];
u3(0.101105699816165,-pi/2,0) q[47];
cx q[46],q[47];
u2(0,pi/2) q[46];
u3(1.4702087094713374,1.4691752037207921,-3.1313531404558756) q[47];
cx q[46],q[47];
u2(pi/2,0) q[46];
u3(1.671383944118456,3.131353140455875,-1.6724174498690019) q[47];
cx q[41],q[47];
u2(0,pi/2) q[41];
u3(1.4702087094713374,1.4691752037207921,-3.1313531404558756) q[47];
cx q[41],q[47];
u1(-pi) q[41];
u3(1.6719020266110611,0,-pi/2) q[47];
u2(-pi/2,-pi) q[52];
cx q[52],q[48];
cx q[48],q[52];
cx q[52],q[48];
cx q[47],q[48];
u2(0,-pi) q[47];
cx q[41],q[47];
u2(-pi/2,-pi) q[41];
u3(0.02499999999999999,0,0) q[47];
cx q[46],q[47];
cx q[47],q[46];
cx q[46],q[47];
u1(pi/2) q[48];
cx q[47],q[48];
u3(1.4696906269787318,pi/2,0) q[47];
cx q[41],q[47];
u2(0,pi/2) q[41];
u3(1.4702087094713374,1.4691752037207921,-3.1313531404558756) q[47];
cx q[41],q[47];
u3(0.7857105983190388,3.106244676545364,1.5957885179548974) q[41];
u3(3.0404869537736285,-pi/2,pi/2) q[47];
u2(0,pi) q[48];
cx q[47],q[48];
u1(0.024999999999999998) q[48];
cx q[47],q[48];
u3(1.4696906269787318,pi/2,-pi) q[47];
cx q[41],q[47];
u2(0,pi/2) q[41];
u3(1.4702087094713374,1.4691752037207921,-3.1313531404558756) q[47];
cx q[41],q[47];
u2(-pi/2,-pi) q[41];
cx q[41],q[40];
cx q[40],q[41];
cx q[41],q[40];
u3(3.0563185593676736,-0.5772388958256918,1.779849659666052) q[47];
u2(0,pi) q[48];
cx q[47],q[48];
u2(0,pi) q[47];
cx q[47],q[46];
u3(0.02499999999999999,-pi,-pi) q[46];
cx q[40],q[46];
u2(0,pi) q[40];
cx q[40],q[41];
cx q[41],q[40];
cx q[40],q[41];
u3(0.024999999999999988,-pi/2,0) q[46];
u3(1.6719020266110611,-pi/2,2.6224465393432705) q[47];
cx q[46],q[47];
u2(0,pi/2) q[46];
u3(1.4702087094713374,1.4691752037207921,-3.1313531404558756) q[47];
cx q[46],q[47];
u3(2.089942441041419,-pi/2,-pi/2) q[46];
u3(3.037452213149349,1.8123812460104496,1.8136462106786766) q[47];
u3(0.02499999999999999,0,-pi/2) q[48];
cx q[47],q[48];
u2(-pi/2,pi/2) q[47];
cx q[41],q[47];
cx q[46],q[47];
cx q[47],q[46];
cx q[46],q[47];
cx q[47],q[41];
cx q[41],q[40];
cx q[46],q[40];
cx q[40],q[46];
cx q[46],q[40];
u1(pi/2) q[40];
u2(0,pi) q[47];
cx q[47],q[51];
cx q[48],q[42];
cx q[42],q[48];
cx q[48],q[42];
cx q[51],q[47];
cx q[47],q[51];
cx q[41],q[47];
cx q[47],q[41];
cx q[41],q[47];
cx q[51],q[52];
cx q[47],q[51];
cx q[51],q[47];
cx q[47],q[51];
cx q[51],q[53];
cx q[51],q[52];
cx q[47],q[51];
u2(0,pi) q[47];
cx q[47],q[41];
u3(1.6719020266110614,-pi/2,pi/2) q[41];
cx q[40],q[41];
u2(0,pi/2) q[40];
u3(1.4702087094713374,1.4691752037207921,-3.1313531404558756) q[41];
cx q[40],q[41];
u1(-pi) q[40];
u3(1.6719020266110611,0,-pi/2) q[41];
cx q[41],q[42];
u2(0,-pi) q[41];
cx q[40],q[41];
u2(0,pi) q[40];
u3(0.02499999999999999,0,0) q[41];
u1(pi/2) q[42];
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[41];
cx q[47],q[41];
u2(0,pi) q[41];
cx q[40],q[41];
cx q[41],q[40];
cx q[40],q[41];
cx q[47],q[41];
u2(0,pi) q[41];
cx q[41],q[40];
u1(0.024999999999999998) q[40];
cx q[41],q[40];
u2(0,pi) q[40];
cx q[40],q[46];
u2(0,pi) q[41];
cx q[46],q[40];
cx q[40],q[46];
u3(0.025000000000000116,pi/2,-pi/2) q[47];
cx q[47],q[41];
u2(0,pi) q[41];
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[41];
u3(0.024999999999999873,-pi/2,pi/2) q[47];
cx q[47],q[46];
u3(0.02499999999999999,0,-pi/2) q[46];
cx q[46],q[40];
cx q[40],q[46];
cx q[46],q[40];
u2(0,pi) q[47];
cx q[47],q[41];
u3(0.02499999999999999,-pi,-pi) q[41];
cx q[42],q[41];
u3(0.02499999999999999,0,0) q[41];
u3(0.101105699816165,-pi/2,0) q[42];
cx q[47],q[41];
u3(1.5957963267948965,0,-pi) q[41];
cx q[41],q[40];
u2(0,-pi/2) q[40];
cx q[40],q[46];
u2(-pi,pi/2) q[41];
cx q[41],q[42];
u2(0,pi/2) q[41];
u3(1.4702087094713374,1.4691752037207921,-3.1313531404558756) q[42];
cx q[41],q[42];
u2(pi/2,0) q[41];
u2(-3.0404869537736285,0) q[42];
cx q[46],q[40];
cx q[40],q[46];
cx q[46],q[50];
u2(0,pi) q[47];
cx q[47],q[41];
u2(0,pi) q[47];
cx q[47],q[51];
cx q[50],q[46];
cx q[46],q[50];
cx q[40],q[46];
cx q[46],q[40];
cx q[40],q[46];
cx q[51],q[50];
cx q[47],q[51];
u2(0,pi) q[47];
cx q[47],q[41];
cx q[41],q[42];
u2(0,pi) q[41];
u2(pi/2,-pi/2) q[42];
cx q[42],q[48];
u2(0,-pi) q[42];
cx q[41],q[42];
u2(0,pi) q[41];
u3(0.02499999999999999,0,0) q[42];
u1(pi/2) q[48];
cx q[47],q[48];
cx q[47],q[41];
u2(0,pi) q[41];
cx q[41],q[42];
cx q[42],q[41];
cx q[41],q[42];
u2(-pi/2,pi/2) q[42];
u3(0.025000000000000116,pi/2,-pi/2) q[47];
u3(3.040486953773628,pi/2,-pi) q[48];
cx q[42],q[48];
u2(-pi/2,-pi/2) q[42];
u3(1.5809840807701345,-3.040999798350607,-0.10059285523918593) q[48];
cx q[42],q[48];
u3(1.5457963267948958,0,-pi/2) q[42];
u3(1.6713839441184557,0.01023951313391791,-1.4691752037207917) q[48];
cx q[42],q[48];
u1(-pi/2) q[42];
u3(3.0404869537736285,pi/2,pi/2) q[48];
cx q[47],q[48];
u3(0.024999999999999873,-pi/2,pi/2) q[47];
u2(0,pi) q[48];
cx q[42],q[48];
cx q[48],q[42];
cx q[42],q[48];
cx q[47],q[48];
u2(0,pi) q[47];
cx q[47],q[41];
u3(0.02499999999999999,-pi,-pi) q[41];
cx q[42],q[41];
u3(0.02499999999999999,0,0) q[41];
u2(0,pi) q[42];
cx q[47],q[41];
u3(1.5957963267948965,0,-pi) q[41];
u2(0,pi) q[47];
u3(0.02499999999999999,0,-pi/2) q[48];
cx q[48],q[42];
cx q[42],q[48];
cx q[41],q[42];
u2(-pi/2,pi/2) q[41];
cx q[42],q[48];
cx q[48],q[42];
cx q[42],q[41];
u1(pi/2) q[41];
cx q[42],q[48];
cx q[48],q[42];
cx q[42],q[48];
cx q[47],q[48];
u2(0,pi) q[47];
cx q[47],q[51];
cx q[46],q[47];
cx q[47],q[46];
cx q[46],q[47];
u2(0,pi) q[48];
cx q[51],q[50];
u2(-pi/2,-pi) q[50];
cx q[51],q[53];
cx q[51],q[47];
cx q[51],q[50];
cx q[50],q[51];
cx q[51],q[50];
cx q[46],q[50];
u2(0,pi) q[46];
cx q[46],q[47];
u3(1.6719020266110614,-pi/2,pi/2) q[47];
cx q[41],q[47];
u2(0,pi/2) q[41];
u3(1.4702087094713374,1.4691752037207921,-3.1313531404558756) q[47];
cx q[41],q[47];
u1(-pi) q[41];
u3(1.6719020266110611,0,-pi/2) q[47];
cx q[47],q[51];
u2(0,-pi) q[47];
cx q[41],q[47];
u2(-pi/2,-pi) q[41];
u3(0.02499999999999999,0,0) q[47];
cx q[46],q[47];
cx q[47],q[46];
cx q[46],q[47];
u1(-pi/2) q[46];
u1(pi/2) q[51];
cx q[47],q[51];
u3(1.4696906269787318,pi/2,0) q[47];
cx q[41],q[47];
u2(0,pi/2) q[41];
u3(1.4702087094713374,1.4691752037207921,-3.1313531404558756) q[47];
cx q[41],q[47];
u3(0.7857105983190388,3.106244676545364,1.5957885179548974) q[41];
u3(3.0404869537736285,-pi/2,pi/2) q[47];
u2(0,pi) q[51];
cx q[47],q[51];
u1(0.024999999999999998) q[51];
cx q[47],q[51];
u3(1.4696906269787318,pi/2,-pi) q[47];
cx q[41],q[47];
u2(0,pi/2) q[41];
u3(1.4702087094713374,1.4691752037207921,-3.1313531404558756) q[47];
cx q[41],q[47];
u2(-pi/2,-pi) q[41];
u3(3.0563185593676736,-0.5772388958256918,1.779849659666052) q[47];
u2(0,pi) q[51];
cx q[47],q[51];
u3(0.6502796335185623,-0.1337742990928894,-1.4032945865224777) q[47];
cx q[46],q[47];
u2(0,pi/2) q[46];
u3(1.4702087094713374,1.4691752037207921,-3.1313531404558756) q[47];
cx q[46],q[47];
u2(2.214297435588181,0) q[46];
u3(1.5956686296756561,-3.0404555650491405,0.0025238587510041377) q[47];
cx q[41],q[47];
u2(0,pi/2) q[41];
u3(0.02499999999999999,0,0) q[47];
cx q[46],q[47];
u2(0,pi) q[46];
u3(1.5957963267948965,0,-pi) q[47];
u3(0.02499999999999999,0,-pi/2) q[51];
cx q[47],q[51];
u2(0,1.6719020266110611) q[47];
cx q[41],q[47];
u2(0,pi/2) q[41];
u3(1.4702087094713374,1.4691752037207921,-3.1313531404558756) q[47];
cx q[41],q[47];
u2(-pi/2,pi/2) q[41];
u3(0.10110569981616474,0,-pi/2) q[47];
cx q[46],q[47];
u2(0,pi) q[46];
cx q[46],q[50];
u2(0,-pi/2) q[51];
cx q[50],q[51];
cx q[46],q[50];
u2(0,pi) q[46];
cx q[46],q[47];
cx q[47],q[41];
u2(pi/2,-pi/2) q[41];
cx q[41],q[42];
u2(0,-pi) q[41];
u1(pi/2) q[42];
cx q[42],q[48];
u2(0,pi) q[47];
cx q[47],q[41];
u3(0.02499999999999999,0,0) q[41];
u2(0,pi) q[47];
cx q[48],q[42];
cx q[42],q[48];
cx q[48],q[47];
cx q[47],q[48];
cx q[48],q[47];
cx q[46],q[47];
u2(0,pi) q[47];
cx q[48],q[47];
cx q[47],q[48];
cx q[48],q[47];
cx q[46],q[47];
u3(1.5531195780778955,-1.5884758378635468,2.356350756468556) q[46];
u2(0,pi) q[47];
cx q[47],q[48];
u1(0.024999999999999998) q[48];
cx q[47],q[48];
u3(1.4696906269787318,pi/2,-pi) q[47];
cx q[46],q[47];
u2(0,pi/2) q[46];
u3(1.4702087094713374,1.4691752037207921,-3.1313531404558756) q[47];
cx q[46],q[47];
u2(-pi/2,-pi) q[46];
u3(3.0563185593676736,-0.5772388958256918,1.779849659666052) q[47];
u2(0,pi) q[48];
cx q[47],q[48];
u2(0,pi) q[47];
cx q[47],q[41];
u3(0.024999999999999988,pi/2,-pi) q[41];
cx q[46],q[47];
cx q[47],q[46];
cx q[46],q[47];
u3(1.4696906269787318,pi/2,pi/2) q[47];
cx q[41],q[47];
u2(0,pi/2) q[41];
u3(1.4702087094713374,1.4691752037207921,-3.1313531404558756) q[47];
cx q[41],q[47];
u2(pi/2,-pi/2) q[41];
u3(1.5459240239141372,3.0404555650491414,-3.13906879483879) q[47];
cx q[46],q[47];
u2(0,pi) q[46];
u3(1.5957963267948965,0,-pi) q[47];
u3(0.02499999999999999,0,-pi/2) q[48];
cx q[47],q[48];
u2(0,1.6719020266110611) q[47];
cx q[41],q[47];
u2(0,pi/2) q[41];
u3(1.4702087094713374,1.4691752037207921,-3.1313531404558756) q[47];
cx q[41],q[47];
u2(-pi/2,pi/2) q[41];
u3(0.10110569981616474,0,-pi/2) q[47];
cx q[46],q[47];
u2(0,pi) q[46];
cx q[46],q[50];
u2(0,-pi/2) q[48];
cx q[48],q[52];
cx q[52],q[48];
cx q[48],q[52];
cx q[52],q[51];
cx q[51],q[52];
cx q[52],q[51];
cx q[50],q[51];
cx q[46],q[50];
u2(0,pi) q[46];
cx q[46],q[47];
cx q[47],q[41];
u2(pi/2,-pi/2) q[41];
cx q[41],q[40];
u1(pi/2) q[40];
u2(0,-pi) q[41];
cx q[46],q[40];
u2(0,pi) q[40];
u2(0,pi) q[47];
cx q[47],q[41];
u3(0.02499999999999999,0,0) q[41];
cx q[40],q[41];
cx q[41],q[40];
cx q[40],q[41];
u2(0,pi) q[47];
cx q[46],q[47];
u3(0.025000000000000116,pi/2,-pi/2) q[46];
u2(0,pi) q[47];
cx q[47],q[41];
u1(0.024999999999999998) q[41];
cx q[47],q[41];
u2(0,pi) q[41];
u2(0,pi) q[47];
cx q[46],q[47];
u3(0.024999999999999873,-pi/2,pi/2) q[46];
u2(0,pi) q[47];
cx q[41],q[47];
cx q[47],q[41];
cx q[41],q[47];
cx q[46],q[47];
u2(0,pi) q[46];
cx q[46],q[40];
u3(0.02499999999999999,-pi,-pi) q[40];
cx q[41],q[40];
u3(0.02499999999999999,0,0) q[40];
u2(0,pi) q[41];
cx q[46],q[40];
u3(1.5957963267948965,0,-pi) q[40];
cx q[40],q[41];
cx q[41],q[40];
cx q[40],q[41];
u2(0,pi) q[46];
u3(0.02499999999999999,0,-pi/2) q[47];
cx q[41],q[47];
u2(-pi/2,pi/2) q[41];
cx q[40],q[41];
u2(0,pi) q[41];
cx q[46],q[40];
u2(0,pi) q[40];
u2(0,pi) q[46];
cx q[46],q[50];
u1(pi/2) q[47];
u2(-pi/2,-pi) q[50];
u3(0.1011056998161648,-pi/2,-pi) q[51];
cx q[50],q[51];
u2(0,pi/2) q[50];
u3(1.4702087094713374,1.4691752037207921,-3.1313531404558756) q[51];
cx q[50],q[51];
u2(-pi/2,-pi) q[50];
u3(3.0404869537736285,-pi/2,pi/2) q[51];
cx q[51],q[52];
cx q[51],q[47];
cx q[47],q[51];
cx q[51],q[47];
cx q[47],q[48];
u2(0,pi) q[47];
u2(0,pi) q[48];
u2(0,pi) q[52];
u2(0,pi) q[53];
