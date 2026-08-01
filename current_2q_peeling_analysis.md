Herein we just analyze the limitations of current 2Q block peeling/emission operation for our holistic BSF simplification algorithm (holistic.py).


## Perhaps 1Q emission is better than only 2Q emission sometimes

For example, when the BSF to simplify currently is (都假设首行是current target row):
```
[
    'XXII',
    'XYYY',
    'XYZZ'
]
```
按照2Q emission模式，'XXII'被peel掉，剩下
```
[
    'XYYY',
    'XYZZ'
]
```
如此一来，peel掉的'XXII'一般而言需要两个Clifford2Q去简化为single-qubit rotation gate，而剩下的['XYYY', 'XYZZ']还需要继续做Clifford search去简化。

然而，如果此时暂时不降'XXII' peel掉，而是应用C(Z,X)@{1,0}，那么原本的three Pauli strings会简化为
```
[
    'IXII',
    'IYYY',
    'IYZZ'
]
```
如此一来仅仅通过一个Clifford conjugation不仅将原本应该peel掉然后再用一个Clifford conjugation综合成1Q rotation的'XXII'给简化到最佳，并且顺带地简化了剩下了['XYYY', 'XYZZ']->['IYYY','IYZZ]

从这个方面来说，2Q emission会错过一些优化机会




## The conclusive proposal: Adaptive emission based on weight calculation


当然并不是说2Q emission这个basic mechanism不合适。毕竟尽早emission能够降低active rows的pattern复杂性，更利于高效的simultaneous simplification（这个intuition在之前的program pattern analysis中已经得到过）

再比如说，如果a series of 2Q Paulis such as 
```
[
    'XXII',
    'XYII',
    'YZII',
    'ZZII,
]
```
被peel掉以后，他们的implementation overhead（CNOT synthesis overhead仅仅是需要两个CNOT gates，这一点可以通过rank-J验证）——不过这个结果同样能够被仅仅一个Clifford conjugation （taht is, C(X,Z)）都给降权到single-qubit Pauli rotations ['IXII', 'IYII', 'YIII', 'ZIII']来得到的。从这个意义上来讲，我们选择这种Clifford options是容易得到跟rank-J theorem相辅相成的结论的，是自洽的


所以说，sometimes 1Q emission比2Q emission更可取，也就是不要那么早的emission……不够2Q emission优于1Q emission的例子也很容易找，例如
```
[
    'XXII',
    'IXYZ',
    'IYYZ'
    'IZZX'
]
```
如果此时还不去peel掉'XXII'而是在整体去简化'XXII'的话，不管是作用C(X,Y)@{0,1}, C(X,Z)@{0,1}还是其他啥Clifford candidate，都会对后面的['IXYZ', 'IYYZ', 'IZZX']某一行或者某两行或者某三行的权重有增加的效果（这是我们不愿看到的），所以这个时候就应该提早peel掉 'XXII'

To conclude, 是否有一种adaptive peeling的方式，sometimes 1Q emission, sometimes 2Q emission，也就是当target row是2Q Pauli的时候，peel不peel掉应该用quantified metric去评估，可以结合原本的arg min（dW, nben, -nharm）这种代价函数去评估？！

当然原本的简单的2Q emission是比较简单而且目前效果不错，如果做adaptive peeling (1Q or 2Q adaptive emission)的时候，其中似乎有仔细的考量避免算法退化和程度bug……
