OPENQASM 2.0;
include "qelib1.inc";

qreg q[4];
u3(0.5*pi,3.0366493267508994*pi,4.0*pi) q[0];
u3(1.5*pi,0.0*pi,1.0*pi) q[1];
u3(1.5*pi,-0.5*pi,4.0*pi) q[2];
u3(0.0*pi,-0.5*pi,1.0*pi) q[3];
cx q[1],q[0];
u3(0.0*pi,-0.5*pi,0.892972373126333*pi) q[0];
cx q[2],q[0];
u3(0.0*pi,-0.5*pi,1.0091027990364574*pi) q[0];
cx q[3],q[0];
u3(0.0*pi,-0.5*pi,2.4101163586123855*pi) q[0];
cx q[1],q[0];
cx q[2],q[0];
cx q[3],q[0];
u3(0.5*pi,0.0*pi,0.5*pi) q[2];
cx q[2],q[1];
u3(0.5*pi,-0.5*pi,0.5*pi) q[3];
cx q[1],q[0];
u3(0.0*pi,-0.5*pi,3.9658625463778403*pi) q[0];
cx q[3],q[0];
u3(1.0*pi,-0.5*pi,0.6268773275143462*pi) q[0];
cx q[3],q[0];
cx q[1],q[0];
u3(0.5*pi,-0.5*pi,0.5*pi) q[3];
cx q[2],q[1];
cx q[3],q[2];
cx q[2],q[1];
cx q[1],q[0];
u3(0.0*pi,-0.5*pi,1.4268494060525898*pi) q[0];
cx q[1],q[0];
cx q[2],q[1];
u3(0.5*pi,-0.5*pi,0.5*pi) q[2];
cx q[3],q[2];
u3(0.5*pi,0.0*pi,0.5*pi) q[3];
cx q[3],q[1];
cx q[1],q[0];
u3(0.5*pi,0.0*pi,0.5*pi) q[3];
u3(1.0*pi,-0.5*pi,4.373122672485653*pi) q[0];
cx q[1],q[0];
u3(0.5*pi,-0.5*pi,0.5*pi) q[1];
cx q[3],q[1];
cx q[1],q[0];
u3(0.5*pi,-0.5*pi,0.5*pi) q[3];
u3(1.0*pi,-0.5*pi,4.395890216352254*pi) q[0];
cx q[1],q[0];
u3(0.5*pi,-0.5*pi,0.5*pi) q[1];
cx q[3],q[1];
u3(0.0*pi,-0.5*pi,1.0*pi) q[1];
u3(0.5*pi,-0.5*pi,0.5*pi) q[3];
cx q[2],q[1];
cx q[1],q[0];
u3(1.0*pi,-0.5*pi,3.706133155940149*pi) q[0];
cx q[1],q[0];
cx q[2],q[1];
u3(0.5*pi,-0.5*pi,0.5*pi) q[2];
cx q[3],q[2];
cx q[2],q[1];
cx q[1],q[0];
u3(0.0*pi,-0.5*pi,2.3205656540510775*pi) q[0];
cx q[1],q[0];
cx q[2],q[1];
u3(0.0*pi,-0.5*pi,1.0*pi) q[1];
u3(0.5*pi,-0.5*pi,0.5*pi) q[2];
cx q[3],q[2];
u3(0.0*pi,-0.5*pi,1.0*pi) q[2];
u3(0.5*pi,-0.5*pi,0.5*pi) q[3];
cx q[3],q[0];
cx q[0],q[2];
u3(0.0*pi,-0.5*pi,1.0*pi) q[2];
cx q[2],q[1];
u3(1.0*pi,-0.5*pi,3.5987061060073358*pi) q[1];
u3(0.03462822763584381*pi,1.0*pi,0.5*pi) q[2];
cx q[2],q[1];
u3(0.0*pi,-0.5*pi,1.0*pi) q[1];
u3(0.5*pi,-0.5*pi,0.5*pi) q[2];
cx q[0],q[2];
cx q[3],q[0];
u3(0.0*pi,-0.5*pi,1.0*pi) q[2];
u3(0.5*pi,0.0*pi,0.5*pi) q[3];
cx q[3],q[1];
cx q[1],q[0];
u3(0.0*pi,-0.5*pi,2.6794343459489225*pi) q[0];
cx q[1],q[0];
u3(0.5*pi,-0.5*pi,0.5*pi) q[1];
cx q[3],q[1];
cx q[1],q[0];
u3(0.5*pi,0.0*pi,0.5*pi) q[3];
u3(1.0*pi,-0.5*pi,3.8837234764739628*pi) q[0];
cx q[2],q[0];
u3(0.0*pi,-0.5*pi,3.4749412061475584*pi) q[0];
cx q[2],q[0];
cx q[1],q[0];
u3(0.5*pi,0.0*pi,0.5*pi) q[2];
cx q[2],q[1];
cx q[1],q[0];
u3(1.0*pi,-0.5*pi,4.389786733693812*pi) q[0];
cx q[3],q[0];
u3(1.0*pi,-0.5*pi,4.362186917570817*pi) q[0];
cx q[3],q[0];
cx q[1],q[0];
u3(0.5*pi,-0.5*pi,0.5*pi) q[3];
cx q[2],q[1];
cx q[3],q[2];
cx q[2],q[1];
cx q[1],q[0];
u3(1.0*pi,-0.5*pi,4.141206647430727*pi) q[0];
cx q[1],q[0];
cx q[2],q[1];
cx q[3],q[2];
u3(0.5*pi,0.0*pi,1.0*pi) q[2];
u3(0.5*pi,0.0*pi,0.5*pi) q[3];
cx q[2],q[0];
cx q[3],q[1];
u3(0.0*pi,-0.5*pi,1.5250587938524416*pi) q[0];
u3(0.5*pi,0.0*pi,0.5*pi) q[3];
cx q[2],q[0];
cx q[1],q[0];
u3(1.0*pi,-0.5*pi,2.6378130824291834*pi) q[0];
cx q[1],q[0];
u3(0.5*pi,0.0*pi,1.0*pi) q[0];
u3(0.5*pi,-0.5*pi,0.5*pi) q[1];
cx q[3],q[1];
u3(0.5*pi,0.0*pi,0.5*pi) q[3];
cx q[3],q[2];
u3(3.5*pi,-0.0026962319454588712*pi,0.5*pi) q[2];
cx q[2],q[0];
u3(1.4753631335167356*pi,1.9652673767934665*pi,4.0*pi) q[2];
cx q[2],q[0];
u3(0.5*pi,0.0*pi,1.0*pi) q[0];
cx q[1],q[0];
u3(1.0*pi,-0.5*pi,0.8708380359456369*pi) q[0];
cx q[1],q[0];
cx q[3],q[1];
u3(0.5*pi,0.0*pi,1.0*pi) q[1];
u3(0.5*pi,0.0*pi,0.5*pi) q[3];
cx q[3],q[0];
u3(1.0*pi,-0.5*pi,4.362186917570817*pi) q[0];
cx q[3],q[0];
cx q[2],q[3];
u3(0.5*pi,-0.5*pi,0.5*pi) q[2];
u3(0.5*pi,-0.5*pi,1.0*pi) q[3];
cx q[2],q[0];
u3(0.5*pi,-0.5*pi,3.6480117915553176*pi) q[0];
cx q[2],q[0];
u3(0.0*pi,-0.5*pi,1.0*pi) q[0];
u3(0.5*pi,-0.5*pi,0.5*pi) q[2];
cx q[3],q[2];
cx q[2],q[1];
u3(0.5*pi,0.43391451889075494*pi,0.5*pi) q[3];
cx q[3],q[0];
u3(1.366470767759901*pi,-0.5*pi,4.0*pi) q[3];
cx q[3],q[0];
cx q[1],q[0];
u3(1.0*pi,-0.5*pi,1.8480703223610493*pi) q[3];
u3(1.0*pi,-0.5*pi,1.2938668440598509*pi) q[0];
cx q[1],q[0];
cx q[2],q[1];
u3(0.5*pi,-0.5*pi,0.5*pi) q[2];
cx q[3],q[2];
cx q[2],q[1];
cx q[1],q[0];
u3(0.0*pi,-0.5*pi,4.3205656540510775*pi) q[0];
cx q[1],q[0];
cx q[2],q[1];
u3(0.0*pi,-0.5*pi,1.0*pi) q[1];
u3(0.5*pi,-0.5*pi,0.5*pi) q[2];
cx q[3],q[2];
u3(0.0*pi,-0.5*pi,1.0*pi) q[2];
u3(0.5*pi,-0.5*pi,0.5*pi) q[3];
cx q[3],q[0];
cx q[0],q[2];
u3(0.0*pi,-0.5*pi,1.0*pi) q[2];
cx q[2],q[1];
u3(3.0*pi,-0.5*pi,1.4012938939926642*pi) q[1];
u3(0.9101163586123862*pi,1.0*pi,0.5*pi) q[2];
cx q[2],q[1];
u3(0.0*pi,-0.5*pi,1.0*pi) q[1];
u3(0.5*pi,-0.5*pi,0.5*pi) q[2];
cx q[0],q[2];
cx q[3],q[0];
u3(0.0*pi,-0.5*pi,1.0*pi) q[2];
u3(0.5*pi,0.0*pi,0.5*pi) q[3];
cx q[3],q[1];
cx q[1],q[0];
u3(1.0*pi,-0.5*pi,0.6794343459489227*pi) q[0];
cx q[1],q[0];
u3(0.5*pi,-0.5*pi,0.5*pi) q[1];
cx q[3],q[1];
u3(0.0*pi,-0.5*pi,1.0*pi) q[1];
u3(0.5*pi,0.0*pi,0.5*pi) q[3];
cx q[1],q[0];
u3(0.0*pi,-0.5*pi,4.335541753914762*pi) q[0];
cx q[2],q[0];
u3(1.0*pi,-0.5*pi,1.0091027990364574*pi) q[0];
cx q[2],q[0];
cx q[1],q[0];
u3(0.5*pi,0.0*pi,0.5*pi) q[2];
cx q[2],q[1];
cx q[1],q[0];
u3(0.0*pi,-0.5*pi,4.147706323266071*pi) q[0];
cx q[3],q[0];
u3(0.0*pi,-0.5*pi,4.373122672485653*pi) q[0];
cx q[3],q[0];
cx q[1],q[0];
u3(0.5*pi,-0.5*pi,0.5*pi) q[3];
cx q[2],q[1];
cx q[3],q[2];
cx q[2],q[1];
cx q[1],q[0];
u3(0.0*pi,-0.5*pi,1.2825031588590738*pi) q[0];
cx q[1],q[0];
cx q[2],q[1];
u3(0.5*pi,-0.5*pi,0.5*pi) q[2];
cx q[3],q[2];
u3(0.5*pi,0.0*pi,0.5*pi) q[3];
cx q[3],q[1];
cx q[1],q[0];
u3(0.5*pi,0.0*pi,0.5*pi) q[3];
u3(0.0*pi,-0.5*pi,0.6268773275143461*pi) q[0];
cx q[1],q[0];
u3(0.5*pi,-0.5*pi,0.5*pi) q[1];
cx q[3],q[1];
cx q[1],q[0];
u3(0.5*pi,-0.5*pi,0.5*pi) q[3];
u3(0.0*pi,-0.5*pi,0.5243885652240645*pi) q[0];
cx q[1],q[0];
u3(0.5*pi,-0.5*pi,0.5*pi) q[1];
cx q[3],q[1];
cx q[2],q[1];
u3(0.5*pi,-0.5*pi,0.5*pi) q[3];
cx q[1],q[0];
u3(0.0*pi,-0.5*pi,4.313539381721958*pi) q[0];
cx q[1],q[0];
cx q[2],q[1];
u3(0.5*pi,-0.5*pi,0.5*pi) q[2];
cx q[3],q[2];
cx q[2],q[1];
cx q[1],q[0];
u3(0.0*pi,-0.5*pi,0.6893822130470084*pi) q[0];
cx q[1],q[0];
cx q[2],q[1];
u3(0.5*pi,-0.5*pi,0.5*pi) q[2];
cx q[3],q[2];
cx q[2],q[0];
u3(0.0*pi,-0.5*pi,0.6864606182780418*pi) q[0];
cx q[2],q[0];
cx q[0],q[3];
u3(0.5*pi,0.0*pi,0.5*pi) q[2];
cx q[3],q[2];
u3(0.0*pi,-0.5*pi,0.6893822130470084*pi) q[2];
u3(0.5*pi,-0.5*pi,0.5*pi) q[3];
cx q[3],q[1];
u3(1.0*pi,-0.5*pi,0.8809033206481014*pi) q[1];
cx q[3],q[1];
u3(0.5*pi,-0.5*pi,0.5*pi) q[3];
cx q[3],q[2];
cx q[0],q[3];
u3(0.5*pi,0.0*pi,1.0*pi) q[2];
cx q[3],q[1];
cx q[1],q[0];
u3(0.0*pi,-0.5*pi,4.310617786952991*pi) q[0];
cx q[1],q[0];
u3(0.0*pi,-0.5*pi,1.0*pi) q[0];
cx q[3],q[1];
u3(0.5*pi,0.0*pi,1.0*pi) q[1];
u3(0.5*pi,-0.5*pi,0.5*pi) q[3];
cx q[3],q[2];
cx q[2],q[0];
u3(0.0*pi,-0.5*pi,2.1190966793518986*pi) q[0];
u3(0.5*pi,-0.5*pi,0.5*pi) q[2];
cx q[2],q[1];
u3(0.0*pi,-0.5*pi,4.378911451820455*pi) q[1];
cx q[2],q[1];
u3(0.5*pi,-0.5*pi,0.5*pi) q[2];
cx q[2],q[0];
u3(0.0*pi,-0.5*pi,1.0*pi) q[0];
u3(0.5*pi,-0.5*pi,0.5*pi) q[2];
cx q[3],q[2];
u3(0.0*pi,-0.5*pi,1.0*pi) q[2];
u3(0.5*pi,0.0*pi,0.5*pi) q[3];
cx q[3],q[0];
u3(0.5*pi,-0.5*pi,4.310617786952991*pi) q[0];
cx q[3],q[0];
u3(0.0*pi,-0.5*pi,0.6417932107064213*pi) q[0];
u3(0.5*pi,0.0*pi,0.5*pi) q[3];
cx q[1],q[0];
u3(1.0*pi,-0.5*pi,1.3830259932473412*pi) q[0];
cx q[2],q[0];
u3(1.0*pi,-0.5*pi,1.400930011031504*pi) q[0];
cx q[1],q[0];
cx q[2],q[0];
u3(0.5*pi,0.0*pi,0.5*pi) q[2];
cx q[2],q[1];
cx q[1],q[0];
u3(1.0*pi,-0.5*pi,3.7866413212251135*pi) q[0];
cx q[3],q[0];
u3(1.0*pi,-0.5*pi,3.992601036972883*pi) q[0];
cx q[3],q[0];
cx q[1],q[0];
u3(0.5*pi,-0.5*pi,0.5*pi) q[3];
cx q[2],q[1];
cx q[3],q[2];
cx q[2],q[1];
cx q[1],q[0];
u3(0.0*pi,-0.5*pi,3.9439128890670156*pi) q[0];
cx q[1],q[0];
cx q[2],q[1];
u3(0.5*pi,-0.5*pi,0.5*pi) q[2];
cx q[3],q[2];
u3(0.5*pi,0.0*pi,0.5*pi) q[3];
cx q[3],q[1];
cx q[1],q[0];
u3(0.5*pi,0.0*pi,0.5*pi) q[3];
u3(1.0*pi,-0.5*pi,1.0073989630271167*pi) q[0];
cx q[1],q[0];
u3(0.5*pi,-0.5*pi,0.5*pi) q[1];
cx q[3],q[1];
cx q[1],q[0];
u3(0.5*pi,-0.5*pi,0.5*pi) q[3];
u3(0.0*pi,-0.5*pi,3.349234502002687*pi) q[0];
cx q[1],q[0];
u3(0.5*pi,-0.5*pi,0.5*pi) q[1];
cx q[3],q[1];
u3(0.0*pi,-0.5*pi,1.0*pi) q[1];
u3(0.5*pi,-0.5*pi,0.5*pi) q[3];
cx q[2],q[1];
cx q[1],q[0];
u3(0.0*pi,-0.5*pi,2.5718874456018828*pi) q[0];
cx q[1],q[0];
cx q[2],q[1];
u3(0.5*pi,-0.5*pi,0.5*pi) q[2];
cx q[3],q[2];
cx q[2],q[1];
cx q[1],q[0];
u3(0.0*pi,-0.5*pi,1.6064327124643438*pi) q[0];
cx q[1],q[0];
cx q[2],q[1];
u3(0.0*pi,-0.5*pi,1.0*pi) q[1];
u3(0.5*pi,-0.5*pi,0.5*pi) q[2];
cx q[3],q[2];
u3(0.0*pi,-0.5*pi,1.0*pi) q[2];
u3(0.5*pi,-0.5*pi,0.5*pi) q[3];
cx q[3],q[0];
cx q[0],q[2];
u3(0.0*pi,-0.5*pi,1.0*pi) q[2];
cx q[2],q[1];
u3(0.0*pi,-0.5*pi,2.8619585590265393*pi) q[1];
u3(3.242102372457977*pi,1.0*pi,0.5*pi) q[2];
cx q[2],q[1];
u3(0.0*pi,-0.5*pi,1.0*pi) q[1];
u3(0.5*pi,-0.5*pi,0.5*pi) q[2];
cx q[0],q[2];
cx q[3],q[0];
u3(0.0*pi,-0.5*pi,1.0*pi) q[2];
u3(0.5*pi,0.0*pi,0.5*pi) q[3];
cx q[3],q[1];
cx q[1],q[0];
u3(1.0*pi,-0.5*pi,4.393567287535657*pi) q[0];
cx q[1],q[0];
u3(0.5*pi,-0.5*pi,0.5*pi) q[1];
cx q[3],q[1];
cx q[1],q[0];
u3(0.5*pi,0.0*pi,0.5*pi) q[3];
u3(0.0*pi,-0.5*pi,0.7835864214146517*pi) q[0];
cx q[2],q[0];
u3(0.0*pi,-0.5*pi,1.2787462016683473*pi) q[0];
cx q[2],q[0];
cx q[1],q[0];
u3(0.5*pi,0.0*pi,0.5*pi) q[2];
cx q[2],q[1];
cx q[1],q[0];
u3(0.0*pi,-0.5*pi,1.0671728428274943*pi) q[0];
cx q[3],q[0];
u3(0.0*pi,-0.5*pi,3.7985417641913077*pi) q[0];
cx q[3],q[0];
cx q[1],q[0];
u3(0.5*pi,-0.5*pi,0.5*pi) q[3];
cx q[2],q[1];
cx q[3],q[2];
cx q[2],q[1];
cx q[1],q[0];
u3(0.0*pi,-0.5*pi,1.0201235727260267*pi) q[0];
cx q[1],q[0];
cx q[2],q[1];
cx q[3],q[2];
u3(0.5*pi,0.0*pi,1.0*pi) q[2];
u3(0.5*pi,0.0*pi,0.5*pi) q[3];
cx q[2],q[0];
u3(1.5*pi,-0.5*pi,3.7212537983316527*pi) q[0];
u3(0.5*pi,0.0*pi,0.5*pi) q[2];
cx q[3],q[0];
cx q[2],q[0];
u3(3.0*pi,-0.5*pi,1.2014582358086923*pi) q[0];
u3(0.5*pi,0.0*pi,0.5*pi) q[2];
cx q[2],q[0];
cx q[1],q[0];
u3(0.5*pi,0.0*pi,0.5*pi) q[2];
u3(1.5*pi,-0.5*pi,1.2014582358086925*pi) q[0];
cx q[1],q[0];
u3(1.3445168124551783*pi,0.0*pi,1.0*pi) q[0];
u3(3.5*pi,0.22661077157401077*pi,4.0*pi) q[1];
cx q[2],q[1];
u3(1.5*pi,3.3108114946118086*pi,3.8530483690902675*pi) q[1];
u3(0.5*pi,0.0*pi,0.5*pi) q[2];
cx q[3],q[2];
u3(2.3202342393169393*pi,3.026402775323154*pi,4.033658711181683*pi) q[2];
u3(0.5*pi,0.0*pi,0.5*pi) q[3];
cx q[3],q[0];
u3(0.0*pi,-0.5*pi,1.3663174886511777*pi) q[0];
u3(0.5*pi,-0.5*pi,0.5*pi) q[3];
cx q[3],q[0];
u3(3.153914042261715*pi,-0.446391856568916*pi,0.5145983898220038*pi) q[3];
cx q[2],q[3];
u3(1.4380459625958186*pi,-0.5*pi,4.0*pi) q[2];
u3(1.5*pi,-0.5*pi,2.234501370959869*pi) q[3];
cx q[2],q[3];
u3(2.201047930897186*pi,2.971652451296569*pi,0.9027765893794157*pi) q[2];
u3(3.349135012464152*pi,0.9431336409590569*pi,1.7610058563506188*pi) q[3];
cx q[2],q[1];
u3(0.0*pi,-0.5*pi,1.0556343603333915*pi) q[1];
cx q[3],q[1];
u3(0.0*pi,-0.5*pi,4.262562752879896*pi) q[1];
cx q[2],q[1];
u3(0.0*pi,-0.5*pi,4.393687323287296*pi) q[1];
u3(3.4040064681026543*pi,-0.19596882374546531*pi,3.8391803265585285*pi) q[2];
cx q[3],q[1];
u3(2.035165329166186*pi,0.0*pi,3.0*pi) q[1];
u3(3.3929275515287425*pi,0.5820017797406676*pi,4.455649527757146*pi) q[3];
cx q[2],q[3];
u3(1.2335392234908853*pi,-0.5*pi,4.0*pi) q[2];
u3(1.5*pi,-0.5*pi,2.3927675851201116*pi) q[3];
cx q[2],q[3];
u3(3.3385155477672526*pi,2.8512472436402234*pi,3.780490267897113*pi) q[2];
u3(1.400870979856561*pi,1.0007319635663519*pi,0.9763763363951738*pi) q[3];
cx q[2],q[1];
u3(3.7302853892840377*pi,0.0*pi,4.0*pi) q[1];
cx q[3],q[1];
u3(3.99247082322404*pi,0.0*pi,4.0*pi) q[1];
u3(3.5958649322757856*pi,-0.12774480545307315*pi,4.3969726357971695*pi) q[3];
cx q[2],q[1];
u3(3.34195765888673*pi,1.4361929899161754*pi,4.0*pi) q[1];
u3(2.4085622832068676*pi,1.4857243327991765*pi,4.039898287649281*pi) q[2];
cx q[2],q[3];
u3(1.4190617012603697*pi,-0.5*pi,4.0*pi) q[2];
u3(1.5*pi,-0.5*pi,2.1428006573396003*pi) q[3];
cx q[2],q[3];
u3(1.751193756031258*pi,-0.12321979765181168*pi,1.1180646418656415*pi) q[2];
u3(3.870644016726887*pi,-0.451546492582383*pi,1.4783429303029656*pi) q[3];
cx q[2],q[1];
u3(0.0*pi,-0.5*pi,4.341781852163982*pi) q[1];
cx q[3],q[1];
u3(0.0*pi,-0.5*pi,4.194727028934678*pi) q[1];
cx q[2],q[1];
u3(0.0*pi,-0.5*pi,4.250093495821069*pi) q[1];
u3(0.47729609885368146*pi,0.2058672097604589*pi,4.23223241926466*pi) q[2];
cx q[3],q[1];
u3(2.422914192600709*pi,-0.1938898198419307*pi,1.0606346684555512*pi) q[3];
cx q[2],q[3];
u3(1.5*pi,1.4888025709789297*pi,0.5*pi) q[2];
u3(1.0*pi,-0.5*pi,0.7377953610696326*pi) q[3];
cx q[2],q[3];
u3(0.5*pi,0.0*pi,0.5*pi) q[2];
u3(0.0*pi,-0.5*pi,0.5443862264940842*pi) q[3];
cx q[2],q[3];
u3(3.4833781694561927*pi,2.0597951638741656*pi,1.8066161013702617*pi) q[2];
u3(3.426072925906487*pi,0.3029243635407304*pi,4.030818646180152*pi) q[3];
