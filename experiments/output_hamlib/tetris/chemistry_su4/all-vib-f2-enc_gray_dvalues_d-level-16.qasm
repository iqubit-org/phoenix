OPENQASM 2.0;
include "qelib1.inc";


// Customized 'ryy' gate definition
gate ryy(param0) q0,q1 {
    rx(pi/2) q0;
    rx(pi/2) q1;
    cx q0, q1;
    rz(param0) q1;
    cx q0, q1;
    rx(-pi/2) q0;
    rx(-pi/2) q1;
}

// Customized 'can' gate definition
gate can (param0, param1, param2) q0,q1 {
    rxx(param0) q0, q1;
    ryy(param1) q0, q1;
    rzz(param2) q0, q1;
}

// Qubits: [0, 1, 2, 3]
qreg q[4];

// Quantum gate operations
u3(0.5*pi, 0.9158926265318419*pi, 0.49999999999999983*pi) q[2];
u3(0.5896612263666222*pi, -0.1569378771126058*pi, 0.346515154357279*pi) q[1];
u3(0.30850547899454855*pi, 0.0*pi, 0.0*pi) q[0];
u3(0.49999999999999983*pi, -0.9158926265318421*pi, -0.5*pi) q[3];
can(0.4452487469354465*pi, 0.12693886075165572*pi, 0.0*pi) q[2], q[3];
u3(0.5000000000000002*pi, 0.5000000000000001*pi, 0.617010957989097*pi) q[2];
can(0.44524874693544647*pi, 0.3183098861837907*pi, -0.12693886075165578*pi) q[0], q[2];
u3(0.5485670638714657*pi, -0.23215526569533862*pi, 0.3302051110495753*pi) q[0];
u3(0.6127510933130473*pi, 0.752187872943173*pi, 0.2682521786841183*pi) q[2];
u3(0.8240436112523946*pi, -0.8123395901943017*pi, -0.9860906755573862*pi) q[3];
can(0.44524874693544636*pi, 0.3183098861837907*pi, -0.1269388607516557*pi) q[1], q[3];
u3(0.3121543091601783*pi, 1.0*pi, 0.0*pi) q[1];
u3(0.2243981055263902*pi, 0.0*pi, 0.0*pi) q[3];
can(0.44524874693544647*pi, 0.3183098861837907*pi, -0.12693886075165564*pi) q[0], q[3];
u3(0.3121543091601784*pi, 1.0*pi, 0.0*pi) q[0];
can(0.46854507569000264*pi, 0.31830988618379064*pi, 0.16807469667757888*pi) q[0], q[1];
u3(0.29252336525667283*pi, 1.0*pi, -1.0*pi) q[1];
can(0.46854507569000264*pi, 0.31830988618379064*pi, 0.16807469667757888*pi) q[1], q[2];
u3(0.42730736839509875*pi, -0.9253338719202746*pi, 0.741398018689139*pi) q[1];
u3(0.5955908205995679*pi, 0.6087920662894124*pi, 0.5333739373181451*pi) q[2];
u3(0.42730736839509875*pi, -0.9253338719202746*pi, 0.741398018689139*pi) q[0];
u3(0.4408773814727032*pi, 0.9398302854014134*pi, 0.2556206580811882*pi) q[3];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[3];
u3(0.7250413337984101*pi, -0.8802021521339969*pi, 0.6739829421941058*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.12932248001367283*pi, -1.0*pi, 0.5*pi) q[0];
u3(0.49999999999999983*pi, -0.398917289408448*pi, 1.0*pi) q[2];
u3(0.25000000000000006*pi, 1.0*pi, -0.5*pi) q[3];
can(0.49999999999999994*pi, 0.0*pi, 0.0*pi) q[0], q[3];
u3(0.5*pi, -0.5*pi, 0.0*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.49999999999999983*pi, 1.0*pi, -0.49999999999999994*pi) q[0];
u3(0.8014686432731347*pi, 0.5000000000000002*pi, -0.4999999999999998*pi) q[2];
u3(1.0*pi, 0.0*pi, 0.0*pi) q[3];
can(0.49999999999999994*pi, 0.0*pi, 0.0*pi) q[0], q[3];
u3(0.7437786953395665*pi, -0.5*pi, 0.0*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.2448643641646762*pi, 0.026747458345829964*pi, -0.5371490939226126*pi) q[0];
u3(0.2710262980485665*pi, 0.7829614394618781*pi, -0.21703856053812176*pi) q[2];
u3(0.013483800039472126*pi, 0.5000000000000002*pi, 0.9999999999999997*pi) q[3];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[3];
u3(0.21796788210555104*pi, 0.49999999999999994*pi, -0.49999999999999994*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.5*pi, -0.7562213046604334*pi, 1.0*pi) q[0];
u3(0.4184314413870781*pi, -0.29366789530310333*pi, 0.9396294614373178*pi) q[2];
u3(0.5*pi, -0.013483800039472166*pi, 0.0*pi) q[3];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[3];
u3(0.5000000000000001*pi, -1.0*pi, -0.3706775199863271*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.3522449149188719*pi, -0.6457163094872017*pi, 0.06910445654391716*pi) q[0];
u3(0.8582701700224297*pi, -0.7581600181636384*pi, -0.24183998183636182*pi) q[2];
u3(0.16204302058590742*pi, -0.5000000000000001*pi, -0.5*pi) q[3];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[3];
u3(0.2037072537574419*pi, 0.19611362892463796*pi, 0.2698012735461133*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.12932248001367283*pi, -1.0*pi, 0.5*pi) q[0];
u3(0.8582701700224297*pi, -0.7581600181636384*pi, -0.24183998183636182*pi) q[2];
u3(0.5*pi, -0.6620430205859075*pi, 1.0*pi) q[3];
can(0.49999999999999994*pi, 0.0*pi, 0.0*pi) q[0], q[3];
u3(0.12932248001367283*pi, 0.5000000000000001*pi, 0.0*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.5000000000000001*pi, -0.6293224800136726*pi, 0.0*pi) q[0];
u3(0.8582701700224297*pi, -0.7581600181636384*pi, -0.24183998183636182*pi) q[2];
u3(0.0*pi, 0.25000000000000006*pi, 0.25000000000000006*pi) q[3];
can(0.49999999999999994*pi, 0.0*pi, 0.0*pi) q[0], q[3];
u3(0.24999999999999992*pi, 0.49999999999999994*pi, 0.0*pi) q[0];
u3(0.7499999999999999*pi, 0.5*pi, 0.0*pi) q[3];
can(0.5*pi, 0.0*pi, 0.0*pi) q[1], q[3];
u3(0.7250413337984101*pi, -0.8802021521339969*pi, 0.6739829421941058*pi) q[1];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[1], q[2];
u3(0.12932248001367283*pi, -1.0*pi, 0.5*pi) q[1];
u3(0.49999999999999983*pi, -0.398917289408448*pi, 1.0*pi) q[2];
u3(0.25000000000000006*pi, 1.0*pi, -0.5*pi) q[3];
can(0.49999999999999994*pi, 0.0*pi, 0.0*pi) q[1], q[3];
u3(0.5*pi, -0.5*pi, 0.0*pi) q[1];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[1], q[2];
u3(0.49999999999999983*pi, 1.0*pi, -0.49999999999999994*pi) q[1];
u3(0.8014686432731347*pi, 0.5000000000000002*pi, -0.4999999999999998*pi) q[2];
u3(1.0*pi, 0.0*pi, 0.0*pi) q[3];
can(0.49999999999999994*pi, 0.0*pi, 0.0*pi) q[1], q[3];
u3(0.7437786953395665*pi, -0.5*pi, 0.0*pi) q[1];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[1], q[2];
u3(0.2448643641646762*pi, 0.026747458345829964*pi, -0.5371490939226126*pi) q[1];
u3(0.2710262980485665*pi, 0.7829614394618781*pi, -0.21703856053812176*pi) q[2];
u3(0.013483800039472126*pi, 0.5000000000000002*pi, 0.9999999999999997*pi) q[3];
can(0.5*pi, 0.0*pi, 0.0*pi) q[1], q[3];
u3(0.21796788210555104*pi, 0.49999999999999994*pi, -0.49999999999999994*pi) q[1];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[1], q[2];
u3(0.5*pi, -0.7562213046604334*pi, 1.0*pi) q[1];
u3(0.4184314413870781*pi, -0.29366789530310333*pi, 0.9396294614373178*pi) q[2];
u3(0.5*pi, -0.013483800039472166*pi, 0.0*pi) q[3];
can(0.5*pi, 0.0*pi, 0.0*pi) q[1], q[3];
u3(0.5000000000000001*pi, -1.0*pi, -0.3706775199863271*pi) q[1];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[1], q[2];
u3(0.3522449149188719*pi, -0.6457163094872017*pi, 0.06910445654391716*pi) q[1];
u3(0.8582701700224297*pi, -0.7581600181636384*pi, -0.24183998183636182*pi) q[2];
u3(0.16204302058590742*pi, -0.5000000000000001*pi, -0.5*pi) q[3];
can(0.5*pi, 0.0*pi, 0.0*pi) q[1], q[3];
u3(0.2037072537574419*pi, 0.19611362892463796*pi, 0.2698012735461133*pi) q[1];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[1], q[2];
u3(0.12932248001367283*pi, -1.0*pi, 0.5*pi) q[1];
u3(0.49999999999999983*pi, -0.398917289408448*pi, 1.0*pi) q[2];
u3(0.5*pi, -0.6620430205859075*pi, 1.0*pi) q[3];
can(0.49999999999999994*pi, 0.0*pi, 0.0*pi) q[1], q[3];
u3(0.5*pi, -0.5*pi, 0.0*pi) q[1];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[1], q[2];
u3(0.49999999999999983*pi, 1.0*pi, -0.49999999999999994*pi) q[1];
u3(0.6010827105915516*pi, 0.49999999999999994*pi, 0.4999999999999998*pi) q[2];
u3(1.0*pi, 0.0*pi, 0.0*pi) q[3];
can(0.49999999999999994*pi, 0.0*pi, 0.0*pi) q[1], q[3];
u3(0.12932248001367283*pi, 0.5000000000000001*pi, 0.0*pi) q[1];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[1], q[2];
u3(0.5000000000000001*pi, -0.6293224800136726*pi, 0.0*pi) q[1];
u3(0.7344576757897833*pi, 0.3614976892953867*pi, -0.09633879103212237*pi) q[2];
u3(0.0*pi, 0.25000000000000006*pi, 0.25000000000000006*pi) q[3];
can(0.49999999999999994*pi, 0.0*pi, 0.0*pi) q[1], q[3];
u3(0.5000000000000001*pi, 0.4999999999999998*pi, 0.5*pi) q[1];
u3(0.7499999999999999*pi, 0.5*pi, 0.0*pi) q[3];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[3];
u3(0.5*pi, 0.24999999999999986*pi, -1.0*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[1];
u3(0.49999999999999983*pi, 1.0*pi, -0.49999999999999994*pi) q[0];
u3(0.5509175200662026*pi, -0.5*pi, 0.5000000000000001*pi) q[1];
u3(0.25000000000000006*pi, 1.0*pi, -0.5*pi) q[3];
can(0.49999999999999994*pi, 0.0*pi, 0.0*pi) q[0], q[3];
u3(0.9490824799337971*pi, -0.4999999999999989*pi, 0.0*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[1];
u3(0.5*pi, -0.5509175200662025*pi, 0.0*pi) q[0];
u3(0.5296544059764817*pi, 0.9585489520867865*pi, -0.803403602795885*pi) q[1];
u3(0.5*pi, 0.0*pi, -0.5*pi) q[3];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[3];
u3(0.49999999999999994*pi, -1.0*pi, 0.24377869533956673*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[1];
u3(0.25622130466043336*pi, 1.0*pi, -0.5000000000000001*pi) q[0];
u3(0.8014686432731348*pi, -0.9999999999999999*pi, -0.49999999999999983*pi) q[1];
u3(0.5000000000000001*pi, 0.5*pi, 0.0*pi) q[3];
can(0.49999999999999994*pi, 0.0*pi, 0.0*pi) q[0], q[3];
u3(0.5000000000000001*pi, 0.49999999999999983*pi, 0.0*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[1];
u3(0.49999999999999983*pi, 1.0*pi, -0.49999999999999994*pi) q[0];
u3(0.6010827105915516*pi, 0.49999999999999994*pi, 0.4999999999999998*pi) q[1];
u3(0.5000000000000001*pi, -1.0*pi, 1.0*pi) q[3];
can(0.49999999999999994*pi, 0.0*pi, 0.0*pi) q[0], q[3];
u3(0.12932248001367283*pi, 0.5000000000000001*pi, 0.0*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[1];
u3(0.5*pi, -0.6436301018020841*pi, 0.0*pi) q[0];
u3(0.8872062914213403*pi, 0.1436381974164088*pi, -0.34819148546292134*pi) q[1];
u3(0.5000000000000001*pi, -0.985692378211589*pi, 0.0*pi) q[3];
can(0.5*pi, 0.49999999999999994*pi, 0.0*pi) q[0], q[3];
u3(0.9490824799337972*pi, -0.4999999999999993*pi, 0.98569237821159*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[1];
u3(0.050917520066202634*pi, -1.0*pi, 0.49999999999999994*pi) q[0];
u3(0.5000000000000001*pi, 0.05091752006620259*pi, 0.0*pi) q[1];
u3(0.5000000000000001*pi, -0.9999999999999997*pi, 0.014307621788410822*pi) q[3];
can(0.49999999999999994*pi, 0.0*pi, 0.0*pi) q[0], q[3];
u3(0.5000000000000001*pi, 0.49999999999999983*pi, 0.0*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[1];
u3(0.49999999999999983*pi, 1.0*pi, -0.49999999999999994*pi) q[0];
u3(0.8014686432731348*pi, -0.5000000000000001*pi, -0.5000000000000001*pi) q[1];
u3(0.5000000000000001*pi, -1.0*pi, 1.0*pi) q[3];
can(0.49999999999999994*pi, 0.0*pi, 0.0*pi) q[0], q[3];
u3(0.7437786953395665*pi, 0.5*pi, 0.0*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[1];
u3(0.5*pi, -0.7562213046604334*pi, 1.0*pi) q[0];
u3(0.4184314413870781*pi, -0.29366789530310333*pi, 0.9396294614373178*pi) q[1];
u3(0.5*pi, 0.0*pi, -0.5*pi) q[3];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[3];
u3(0.5000000000000001*pi, -1.0*pi, -0.3706775199863271*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[1];
u3(0.6357475123229015*pi, 0.35707653954881624*pi, -0.0626221192258292*pi) q[0];
u3(0.49999999999999983*pi, -0.398917289408448*pi, 1.0*pi) q[1];
u3(0.20513924737754533*pi, 0.5*pi, -1.0*pi) q[3];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[3];
u3(0.5000000000000001*pi, 0.6486864246030065*pi, -0.9999999999999999*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[1];
u3(0.49999999999999983*pi, 1.0*pi, -0.49999999999999994*pi) q[0];
u3(0.5509175200662026*pi, -0.5*pi, 0.5000000000000001*pi) q[1];
u3(0.5000000000000001*pi, 0.2948607526224547*pi, 0.0*pi) q[3];
can(0.49999999999999994*pi, 0.0*pi, 0.0*pi) q[0], q[3];
u3(0.9490824799337971*pi, -0.4999999999999989*pi, 0.0*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[1];
u3(0.5*pi, -0.5509175200662025*pi, 0.0*pi) q[0];
u3(0.5296544059764817*pi, 0.9585489520867865*pi, -0.803403602795885*pi) q[1];
u3(0.5*pi, 0.0*pi, -0.5*pi) q[3];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[3];
u3(0.49999999999999994*pi, -1.0*pi, 0.24377869533956673*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[1];
u3(0.25622130466043336*pi, 1.0*pi, -0.5000000000000001*pi) q[0];
u3(0.4184314413870781*pi, -0.29366789530310333*pi, 0.9396294614373178*pi) q[1];
u3(0.5000000000000001*pi, 0.5*pi, 0.0*pi) q[3];
can(0.49999999999999994*pi, 0.0*pi, 0.0*pi) q[0], q[3];
u3(0.8706775199863271*pi, -0.4999999999999997*pi, 0.0*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[1];
u3(0.5000000000000001*pi, -0.6293224800136726*pi, 0.0*pi) q[0];
u3(0.49999999999999983*pi, -0.398917289408448*pi, 1.0*pi) q[1];
u3(0.5*pi, 0.5*pi, 0.0*pi) q[3];
can(0.49999999999999994*pi, 0.0*pi, 0.0*pi) q[0], q[3];
u3(0.24999999999999992*pi, 0.49999999999999994*pi, 0.0*pi) q[0];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.5*pi, 0.24999999999999986*pi, -1.0*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[1];
u3(0.49999999999999983*pi, 1.0*pi, -0.49999999999999994*pi) q[0];
u3(0.5509175200662026*pi, -0.5*pi, 0.5000000000000001*pi) q[1];
u3(0.25000000000000006*pi, 1.0*pi, -0.5*pi) q[2];
can(0.49999999999999994*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.9490824799337971*pi, -0.4999999999999989*pi, 0.0*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[1];
u3(0.5*pi, -0.5509175200662025*pi, 0.0*pi) q[0];
u3(0.5296544059764817*pi, 0.9585489520867865*pi, -0.803403602795885*pi) q[1];
u3(0.5*pi, 0.0*pi, -0.5*pi) q[2];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.49999999999999994*pi, -1.0*pi, 0.24377869533956673*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[1];
u3(0.25622130466043336*pi, 1.0*pi, -0.5000000000000001*pi) q[0];
u3(0.8014686432731348*pi, -0.9999999999999999*pi, -0.49999999999999983*pi) q[1];
u3(0.5000000000000001*pi, 0.5*pi, 0.0*pi) q[2];
can(0.49999999999999994*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.5000000000000001*pi, 0.49999999999999983*pi, 0.0*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[1];
u3(0.49999999999999983*pi, 1.0*pi, -0.49999999999999994*pi) q[0];
u3(0.6010827105915516*pi, 0.49999999999999994*pi, 0.4999999999999998*pi) q[1];
u3(0.5000000000000001*pi, -1.0*pi, 1.0*pi) q[2];
can(0.49999999999999994*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.12932248001367283*pi, 0.5000000000000001*pi, 0.0*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[1];
u3(0.5*pi, -0.6436301018020841*pi, 0.0*pi) q[0];
u3(0.8872062914213403*pi, 0.1436381974164088*pi, -0.34819148546292134*pi) q[1];
u3(0.5000000000000001*pi, -0.985692378211589*pi, 0.0*pi) q[2];
can(0.5*pi, 0.49999999999999994*pi, 0.0*pi) q[0], q[2];
u3(0.9490824799337972*pi, -0.4999999999999993*pi, 0.98569237821159*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[1];
u3(0.050917520066202634*pi, -1.0*pi, 0.49999999999999994*pi) q[0];
u3(0.5000000000000001*pi, 0.05091752006620259*pi, 0.0*pi) q[1];
u3(0.5000000000000001*pi, -0.9999999999999997*pi, 0.014307621788410822*pi) q[2];
can(0.49999999999999994*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.5000000000000001*pi, 0.49999999999999983*pi, 0.0*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[1];
u3(0.49999999999999983*pi, 1.0*pi, -0.49999999999999994*pi) q[0];
u3(0.8014686432731348*pi, -0.5000000000000001*pi, -0.5000000000000001*pi) q[1];
u3(0.5000000000000001*pi, -1.0*pi, 1.0*pi) q[2];
can(0.49999999999999994*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.7437786953395665*pi, 0.5*pi, 0.0*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[1];
u3(0.5*pi, -0.7562213046604334*pi, 1.0*pi) q[0];
u3(0.4184314413870781*pi, -0.29366789530310333*pi, 0.9396294614373178*pi) q[1];
u3(0.5*pi, 0.0*pi, -0.5*pi) q[2];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.5000000000000001*pi, -1.0*pi, -0.3706775199863271*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[1];
u3(0.6357475123229015*pi, 0.35707653954881624*pi, -0.0626221192258292*pi) q[0];
u3(0.49999999999999983*pi, -0.398917289408448*pi, 1.0*pi) q[1];
u3(0.20513924737754533*pi, 0.5*pi, -1.0*pi) q[2];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.5000000000000001*pi, 0.6486864246030065*pi, -0.9999999999999999*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[1];
u3(0.49999999999999983*pi, 1.0*pi, -0.49999999999999994*pi) q[0];
u3(0.5509175200662026*pi, -0.5*pi, 0.5000000000000001*pi) q[1];
u3(0.5000000000000001*pi, 0.2948607526224547*pi, 0.0*pi) q[2];
can(0.49999999999999994*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.9490824799337971*pi, -0.4999999999999989*pi, 0.0*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[1];
u3(0.5*pi, -0.5509175200662025*pi, 0.0*pi) q[0];
u3(0.5296544059764817*pi, 0.9585489520867865*pi, -0.803403602795885*pi) q[1];
u3(0.5*pi, 0.0*pi, -0.5*pi) q[2];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.49999999999999994*pi, -1.0*pi, 0.24377869533956673*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[1];
u3(0.25622130466043336*pi, 1.0*pi, -0.5000000000000001*pi) q[0];
u3(0.8014686432731348*pi, -0.9999999999999999*pi, -0.49999999999999983*pi) q[1];
u3(0.5000000000000001*pi, 0.5*pi, 0.0*pi) q[2];
can(0.49999999999999994*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.5000000000000001*pi, 0.49999999999999983*pi, 0.0*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[1];
u3(0.49999999999999983*pi, 1.0*pi, -0.49999999999999994*pi) q[0];
u3(0.6010827105915516*pi, 0.49999999999999994*pi, 0.4999999999999998*pi) q[1];
u3(0.5000000000000001*pi, -1.0*pi, 1.0*pi) q[2];
can(0.49999999999999994*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.12932248001367283*pi, 0.5000000000000001*pi, 0.0*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[1];
u3(0.5000000000000001*pi, -0.6293224800136726*pi, 0.0*pi) q[0];
u3(0.49999999999999983*pi, -0.398917289408448*pi, 1.0*pi) q[1];
u3(0.0*pi, 0.25000000000000006*pi, 0.25000000000000006*pi) q[2];
can(0.49999999999999994*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.24999999999999992*pi, 0.49999999999999994*pi, 0.0*pi) q[0];
u3(0.7499999999999999*pi, 0.5*pi, 0.0*pi) q[2];
u3(0.7499999999999999*pi, 0.5*pi, 0.0*pi) q[3];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[3];
u3(0.5000000000000001*pi, -0.5*pi, 0.5000000000000001*pi) q[0];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.5*pi, 0.24999999999999986*pi, -1.0*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[1];
u3(0.5*pi, 0.0*pi, 0.0*pi) q[0];
u3(0.0*pi, 0.0*pi, 0.0*pi) q[1];
u3(0.75*pi, 0.49999999999999983*pi, 0.49999999999999983*pi) q[2];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(1.0*pi, -1.0*pi, 1.0*pi) q[0];
u3(0.2499999999999999*pi, -0.5000000000000001*pi, 0.0*pi) q[2];
u3(0.25000000000000006*pi, 1.0*pi, -0.5*pi) q[3];
can(0.49999999999999994*pi, 0.0*pi, 0.0*pi) q[0], q[3];
u3(0.25000000000000006*pi, 0.49999999999999983*pi, 0.5*pi) q[0];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.5*pi, 0.24999999999999986*pi, -1.0*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[1];
u3(0.5*pi, 0.0*pi, -0.75*pi) q[0];
u3(0.5509175200662026*pi, -0.5*pi, 0.5000000000000001*pi) q[1];
u3(0.5*pi, -0.5000000000000001*pi, 0.49999999999999983*pi) q[2];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.7500000000000001*pi, 0.5*pi, 0.5*pi) q[0];
u3(0.5*pi, -0.5000000000000001*pi, 0.49999999999999983*pi) q[2];
u3(1.0*pi, 0.0*pi, 0.0*pi) q[3];
can(0.49999999999999994*pi, 0.0*pi, 0.0*pi) q[0], q[3];
u3(0.25000000000000006*pi, 0.49999999999999983*pi, 0.5*pi) q[0];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.25403825346919595*pi, 0.05027829846765116*pi, -0.5714063529802608*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[1];
u3(0.5*pi, -0.5509175200662025*pi, 0.0*pi) q[0];
u3(0.5000000000000001*pi, 0.05091752006620259*pi, 0.0*pi) q[1];
u3(0.5*pi, -0.7500000000000001*pi, 0.0*pi) q[2];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.5000000000000001*pi, 0.0*pi, 0.0*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[1];
u3(0.5*pi, 0.0*pi, 0.14436151941788888*pi) q[0];
u3(0.8014686432731348*pi, -0.5000000000000001*pi, -0.5000000000000001*pi) q[1];
u3(0.5*pi, 0.0*pi, 0.4105936200306131*pi) q[2];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.355638480582111*pi, 0.5*pi, 0.5000000000000001*pi) q[0];
u3(0.7377585144258398*pi, -0.6233104475028977*pi, -0.0860742261088034*pi) q[2];
u3(0.5*pi, 0.0*pi, -0.5*pi) q[3];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[3];
u3(0.25000000000000006*pi, -0.5000000000000002*pi, -0.49999999999999994*pi) q[0];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.3369483031399278*pi, -0.8012015102944762*pi, -0.8099143464016404*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[1];
u3(0.49999999999999994*pi, 0.24377869533956673*pi, -1.0*pi) q[0];
u3(0.8014686432731348*pi, -0.9999999999999999*pi, -0.49999999999999983*pi) q[1];
u3(0.75*pi, 0.49999999999999983*pi, 0.49999999999999983*pi) q[2];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.0*pi, 0.5*pi, 0.5*pi) q[0];
u3(0.2499999999999999*pi, -0.5000000000000001*pi, 0.0*pi) q[2];
u3(0.5000000000000001*pi, -0.5*pi, -0.5*pi) q[3];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[3];
u3(0.25000000000000006*pi, 0.49999999999999983*pi, 0.5*pi) q[0];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.5*pi, 0.24999999999999986*pi, -1.0*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[1];
u3(0.5*pi, 0.0*pi, 0.0*pi) q[0];
u3(0.0*pi, 0.0*pi, 0.0*pi) q[1];
u3(0.75*pi, 0.49999999999999983*pi, 0.49999999999999983*pi) q[2];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(1.0*pi, -1.0*pi, 1.0*pi) q[0];
u3(0.2499999999999999*pi, -0.5000000000000001*pi, 0.0*pi) q[2];
u3(0.5*pi, -0.9999999999999999*pi, -0.9999999999999999*pi) q[3];
can(0.49999999999999994*pi, 0.0*pi, 0.0*pi) q[0], q[3];
u3(0.25000000000000006*pi, 0.49999999999999983*pi, 0.5*pi) q[0];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.5*pi, 0.24999999999999986*pi, -1.0*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[1];
u3(0.5*pi, 0.0*pi, -0.75*pi) q[0];
u3(0.6010827105915516*pi, 0.49999999999999994*pi, 0.4999999999999998*pi) q[1];
u3(0.5*pi, -0.5000000000000001*pi, 0.49999999999999983*pi) q[2];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.7500000000000001*pi, 0.5*pi, 0.5*pi) q[0];
u3(0.5*pi, -0.5000000000000001*pi, 0.49999999999999983*pi) q[2];
u3(1.0*pi, 0.0*pi, 0.0*pi) q[3];
can(0.49999999999999994*pi, 0.0*pi, 0.0*pi) q[0], q[3];
u3(0.25000000000000006*pi, 0.49999999999999983*pi, 0.5*pi) q[0];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.7250413337984101*pi, -0.8802021521339969*pi, 0.6739829421941058*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[1];
u3(0.14103811274709235*pi, 0.8732663914698463*pi, 0.6386951519505655*pi) q[0];
u3(0.49999999999999983*pi, -0.398917289408448*pi, 1.0*pi) q[1];
u3(0.19208221820738955*pi, 0.5000000000000001*pi, 0.5000000000000001*pi) q[2];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.4746161854357732*pi, -0.5581045062762725*pi, 0.0046804375272466656*pi) q[0];
u3(0.7447877734833401*pi, -0.4189725681958862*pi, 0.05698213413295873*pi) q[2];
u3(0.013483800039472126*pi, 0.5000000000000002*pi, 0.9999999999999997*pi) q[3];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[3];
u3(0.5182409893554937*pi, -0.01827101459623001*pi, 0.249476195806506*pi) q[0];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.5*pi, 0.24999999999999986*pi, -1.0*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[1];
u3(0.5*pi, 0.0*pi, 0.14436151941788888*pi) q[0];
u3(0.5509175200662026*pi, -0.5*pi, 0.5000000000000001*pi) q[1];
u3(0.1605936200306132*pi, -0.5000000000000003*pi, 0.5000000000000003*pi) q[2];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.355638480582111*pi, 0.5*pi, 0.5000000000000001*pi) q[0];
u3(0.7377585144258398*pi, -0.6233104475028977*pi, -0.0860742261088034*pi) q[2];
u3(0.5*pi, -0.013483800039472166*pi, 0.0*pi) q[3];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[3];
u3(0.25000000000000006*pi, -0.5000000000000002*pi, -0.49999999999999994*pi) q[0];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.25403825346919595*pi, 0.05027829846765116*pi, -0.5714063529802608*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[1];
u3(0.5*pi, 0.4490824799337974*pi, 0.0*pi) q[0];
u3(0.5000000000000001*pi, 0.05091752006620259*pi, 0.0*pi) q[1];
u3(0.75*pi, 0.49999999999999983*pi, 0.49999999999999983*pi) q[2];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.0*pi, 0.5*pi, 0.5*pi) q[0];
u3(0.2499999999999999*pi, -0.5000000000000001*pi, 0.0*pi) q[2];
u3(0.5000000000000001*pi, -0.5*pi, -0.5*pi) q[3];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[3];
u3(0.25000000000000006*pi, 0.49999999999999983*pi, 0.5*pi) q[0];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.5*pi, 0.24999999999999986*pi, -1.0*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[1];
u3(0.5*pi, 0.0*pi, 0.0*pi) q[0];
u3(0.0*pi, 0.0*pi, 0.0*pi) q[1];
u3(0.75*pi, 0.49999999999999983*pi, 0.49999999999999983*pi) q[2];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(1.0*pi, -1.0*pi, 1.0*pi) q[0];
u3(0.2499999999999999*pi, -0.5000000000000001*pi, 0.0*pi) q[2];
u3(0.5*pi, -0.9999999999999999*pi, -0.9999999999999999*pi) q[3];
can(0.49999999999999994*pi, 0.0*pi, 0.0*pi) q[0], q[3];
u3(0.25000000000000006*pi, 0.49999999999999983*pi, 0.5*pi) q[0];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.5*pi, 0.24999999999999986*pi, -1.0*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[1];
u3(0.5*pi, 0.0*pi, -0.75*pi) q[0];
u3(0.8014686432731348*pi, -0.5000000000000001*pi, -0.5000000000000001*pi) q[1];
u3(0.5*pi, -0.5000000000000001*pi, 0.49999999999999983*pi) q[2];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.7500000000000001*pi, 0.5*pi, 0.5*pi) q[0];
u3(0.5*pi, -0.5000000000000001*pi, 0.49999999999999983*pi) q[2];
u3(1.0*pi, 0.0*pi, 0.0*pi) q[3];
can(0.49999999999999994*pi, 0.0*pi, 0.0*pi) q[0], q[3];
u3(0.25000000000000006*pi, 0.49999999999999983*pi, 0.5*pi) q[0];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.3369483031399278*pi, -0.8012015102944762*pi, -0.8099143464016404*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[1];
u3(0.5*pi, -0.7562213046604334*pi, 1.0*pi) q[0];
u3(0.8014686432731348*pi, -0.9999999999999999*pi, -0.49999999999999983*pi) q[1];
u3(0.5*pi, -0.7500000000000001*pi, 0.0*pi) q[2];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.5000000000000001*pi, 0.0*pi, 0.0*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[1];
u3(0.5*pi, 0.0*pi, 0.14436151941788888*pi) q[0];
u3(0.6010827105915516*pi, 0.49999999999999994*pi, 0.4999999999999998*pi) q[1];
u3(0.5*pi, 0.0*pi, 0.4105936200306131*pi) q[2];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.355638480582111*pi, 0.5*pi, 0.5000000000000001*pi) q[0];
u3(0.7377585144258398*pi, -0.6233104475028977*pi, -0.0860742261088034*pi) q[2];
u3(0.5*pi, 0.0*pi, -0.5*pi) q[3];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[3];
u3(0.25000000000000006*pi, -0.5000000000000002*pi, -0.49999999999999994*pi) q[0];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.7250413337984101*pi, -0.8802021521339969*pi, 0.6739829421941058*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[1];
u3(0.5*pi, -0.37067751998632714*pi, 1.0*pi) q[0];
u3(0.49999999999999983*pi, -0.398917289408448*pi, 1.0*pi) q[1];
u3(0.75*pi, 0.49999999999999983*pi, 0.49999999999999983*pi) q[2];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.6620430205859075*pi, 0.49999999999999983*pi, 0.5*pi) q[0];
u3(0.2499999999999999*pi, -0.5000000000000001*pi, 0.0*pi) q[2];
u3(0.16204302058590742*pi, -0.5000000000000001*pi, -0.5*pi) q[3];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[3];
u3(0.4120430205859076*pi, 0.49999999999999983*pi, 0.5*pi) q[0];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.5*pi, 0.24999999999999986*pi, -1.0*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[1];
u3(0.5*pi, 0.0*pi, 0.0*pi) q[0];
u3(0.0*pi, 0.0*pi, 0.0*pi) q[1];
u3(0.75*pi, 0.49999999999999983*pi, 0.49999999999999983*pi) q[2];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(1.0*pi, -1.0*pi, 1.0*pi) q[0];
u3(0.2499999999999999*pi, -0.5000000000000001*pi, 0.0*pi) q[2];
u3(0.5*pi, -0.6620430205859075*pi, 1.0*pi) q[3];
can(0.49999999999999994*pi, 0.0*pi, 0.0*pi) q[0], q[3];
u3(0.25000000000000006*pi, 0.49999999999999983*pi, 0.5*pi) q[0];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.5*pi, 0.24999999999999986*pi, -1.0*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[1];
u3(0.5*pi, 0.0*pi, -0.75*pi) q[0];
u3(0.5509175200662026*pi, -0.5*pi, 0.5000000000000001*pi) q[1];
u3(0.5*pi, -0.5000000000000001*pi, 0.49999999999999983*pi) q[2];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.7500000000000001*pi, 0.5*pi, 0.5*pi) q[0];
u3(0.5*pi, -0.5000000000000001*pi, 0.49999999999999983*pi) q[2];
u3(1.0*pi, 0.0*pi, 0.0*pi) q[3];
can(0.49999999999999994*pi, 0.0*pi, 0.0*pi) q[0], q[3];
u3(0.25000000000000006*pi, 0.49999999999999983*pi, 0.5*pi) q[0];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.25403825346919595*pi, 0.05027829846765116*pi, -0.5714063529802608*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[1];
u3(0.5*pi, -0.5509175200662025*pi, 0.0*pi) q[0];
u3(0.5000000000000001*pi, 0.05091752006620259*pi, 0.0*pi) q[1];
u3(0.5*pi, -0.7500000000000001*pi, 0.0*pi) q[2];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.5000000000000001*pi, 0.0*pi, 0.0*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[1];
u3(0.5*pi, 0.0*pi, 0.14436151941788888*pi) q[0];
u3(0.8014686432731348*pi, -0.5000000000000001*pi, -0.5000000000000001*pi) q[1];
u3(0.5*pi, 0.0*pi, 0.4105936200306131*pi) q[2];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.355638480582111*pi, 0.5*pi, 0.5000000000000001*pi) q[0];
u3(0.7377585144258398*pi, -0.6233104475028977*pi, -0.0860742261088034*pi) q[2];
u3(0.5*pi, 0.0*pi, -0.5*pi) q[3];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[3];
u3(0.25000000000000006*pi, -0.5000000000000002*pi, -0.49999999999999994*pi) q[0];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.3369483031399278*pi, -0.8012015102944762*pi, -0.8099143464016404*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[1];
u3(0.49999999999999994*pi, 0.24377869533956673*pi, -1.0*pi) q[0];
u3(0.8014686432731348*pi, -0.9999999999999999*pi, -0.49999999999999983*pi) q[1];
u3(0.75*pi, 0.49999999999999983*pi, 0.49999999999999983*pi) q[2];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.0*pi, 0.5*pi, 0.5*pi) q[0];
u3(0.2499999999999999*pi, -0.5000000000000001*pi, 0.0*pi) q[2];
u3(0.5000000000000001*pi, -0.5*pi, -0.5*pi) q[3];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[3];
u3(0.25000000000000006*pi, 0.49999999999999983*pi, 0.5*pi) q[0];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.5*pi, 0.24999999999999986*pi, -1.0*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[1];
u3(0.5*pi, 0.0*pi, 0.0*pi) q[0];
u3(0.0*pi, 0.0*pi, 0.0*pi) q[1];
u3(0.75*pi, 0.49999999999999983*pi, 0.49999999999999983*pi) q[2];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(1.0*pi, -1.0*pi, 1.0*pi) q[0];
u3(0.2499999999999999*pi, -0.5000000000000001*pi, 0.0*pi) q[2];
u3(0.5*pi, -0.9999999999999999*pi, -0.9999999999999999*pi) q[3];
can(0.49999999999999994*pi, 0.0*pi, 0.0*pi) q[0], q[3];
u3(0.25000000000000006*pi, 0.49999999999999983*pi, 0.5*pi) q[0];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.5*pi, 0.24999999999999986*pi, -1.0*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[1];
u3(0.5*pi, 0.0*pi, -0.75*pi) q[0];
u3(0.0*pi, 0.0*pi, 0.0*pi) q[1];
u3(0.5*pi, -0.5000000000000001*pi, 0.49999999999999983*pi) q[2];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.7500000000000001*pi, 0.5*pi, 0.5*pi) q[0];
u3(0.5*pi, -0.5000000000000001*pi, 0.49999999999999983*pi) q[2];
u3(1.0*pi, 0.0*pi, 0.0*pi) q[3];
can(0.49999999999999994*pi, 0.0*pi, 0.0*pi) q[0], q[3];
u3(0.25000000000000006*pi, 0.49999999999999983*pi, 0.5*pi) q[0];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.5*pi, 0.24999999999999986*pi, -1.0*pi) q[0];
can(0.3183098861837907*pi, 0.0*pi, 0.0*pi) q[0], q[1];
u3(0.5*pi, 0.0*pi, -0.75*pi) q[0];
u3(0.5*pi, 0.0*pi, 0.0*pi) q[1];
u3(0.5*pi, -0.5000000000000001*pi, 0.49999999999999983*pi) q[2];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[2];
u3(0.5000000000000001*pi, -0.5*pi, 0.5000000000000001*pi) q[0];
u3(0.5*pi, 0.24999999999999997*pi, 0.0*pi) q[2];
u3(0.25000000000000006*pi, 0.5*pi, -1.0*pi) q[3];
can(0.5*pi, 0.0*pi, 0.0*pi) q[0], q[3];
u3(0.75*pi, -0.5000000000000001*pi, -0.5000000000000001*pi) q[0];
u3(0.5*pi, 0.24999999999999997*pi, 0.0*pi) q[3];
