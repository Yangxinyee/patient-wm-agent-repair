The scaffold is essentially free. Killing dims [4, 67, 108], with the prelude-norm1, `query_norm`-bias and value-MLP-column cuts, changes factual value NLL by +0.00016 on dsE and −0.00025 on dsF, an average of −0.00004. The set [4, 67, 2] is noisier (+0.0013/−0.0009). I take J = [4, 67, 108] with these roles:
- jR = 4, the zero reference;
- jI = 67, the insulin memory;
- jN = 108, the NE memory.

**Units to repurpose.** I need 4 block-0 FFN units (two per drug for the gated write) and 4 value-MLP units (two per drug for the gated readout). I will pick the least important ones, scoring importance as the variance of each unit's contribution to its output. That uses `n3_0` for the FFN units and the value-MLP input `with_code` for the value units, both computed on the scaffold over dsE.

I will then mean-ablate the chosen units, folding their mean output into the biases, and re-measure factual NLL on dsE and dsF. If the total stays within about +0.001, the scaffold leaves most of the 0.005 budget for the circuit itself.

```python
src2 = r'''
import torch, numpy as np
G = None  # set from kernel: dict of kernel functions
def internals(P, ds, chunk=2000):
    H, O, ST = G['tf_loop'](P, ds, keep=('E','hf','hn'))
    Mx, Ma, Mm = G['mem_for'](ds, ds['srows'], torch.as_tensor(ds['e']))
    n3 = []
    for i in range(0, len(ST['hf']), chunk):
        s = slice(i, i+chunk)
        _, _, _, inf = G['core_rest'](P, ST['hf'][s], ST['E'][s], Mx[s], Ma[s], Mm[s], keep=True)
        n3.append(inf['n3_0'])
    ST['n3_0'] = torch.cat(n3); ST['H'] = H; ST['O'] = O
    return ST
def capture_wc(P, ds):
    CW = {}; orig = G['mlp2']
    def cap(x, pre):
        if pre == 'attr_heads.value_bin': CW['wc'] = x.detach().clone()
        if pre == 'z_prior': CW['ctx'] = x.detach().clone()
        return orig(x, pre)
    G['mlp2'] = cap
    try: r = G['tf_eval'](P, ds, False)
    finally: G['mlp2'] = orig
    return CW['wc'], CW['ctx'], r
def mean_ablate(P, vunits, funits, WC, N3):
    import torch.nn.functional as F
    P = {k: v.clone() for k, v in P.items()}
    W0, b0 = P['attr_heads.value_bin.0.weight'], P['attr_heads.value_bin.0.bias']
    h1 = F.gelu(WC.reshape(-1, 192) @ W0.T + b0)
    for k in vunits:
        P['attr_heads.value_bin.2.bias'] += P['attr_heads.value_bin.2.weight'][:, k] * h1[:, k].mean()
        P['attr_heads.value_bin.2.weight'][:, k] = 0; W0[k, :] = 0; b0[k] = 0
    b = 'core.blocks.0.0.ffn'
    Wf, bf = P[b+'.0.weight'], P[b+'.0.bias']
    act = F.gelu(N3.reshape(-1, 192) @ Wf.T + bf)
    for k in funits:
        P[b+'.2.bias'] += P[b+'.2.weight'][:, k] * act[:, k].mean()
        P[b+'.2.weight'][:, k] = 0; Wf[k, :] = 0; bf[k] = 0
    return P
'''
open('/work/lib3.py','a').write(src2)
importlib.reload(lib3); lib3.G=globals()
J=[4,67,108]; P1=lib3.scaffold(PO,J)
ST1=lib3.internals(P1,dsE); WC1,CTX1,r1=lib3.capture_wc(P1,dsE)
import torch.nn.functional as F
W0=P1['attr_heads.value_bin.0.weight']; b0=P1['attr_heads.value_bin.0.bias']; W2=P1['attr_heads.value_bin.2.weight']
h1=F.gelu(WC1.reshape(-1,192)@W0.T+b0); impV=(h1.var(0)*(W2**2).sum(0)).numpy()
bf='core.blocks.0.0.ffn'; act=F.gelu(ST1['n3_0'].reshape(-1,192)@P1[bf+'.0.weight'].T+P1[bf+'.0.bias'])
impF=(act.var(0)*(P1[bf+'.2.weight']**2).sum(0)).numpy()
VU=list(np.argsort(impV)[:4]); FU=list(np.argsort(impF)[:4])
print('VU', VU, np.round(np.sort(impV)[:6],5), 'median', round(float(np.median(impV)),4))
print('FU', FU, np.round(np.sort(impF)[:6],6), 'median', round(float(np.median(impF)),4))
P2=lib3.mean_ablate(P1, VU, FU, WC1, ST1['n3_0'])
c={k:float(tf_eval(P2,d_,False)[0].mean())-base[k] for k,d_ in (('E',dsE),('F',dsF))}
print('scaffold+units cost', {k:round(v,5) for k,v in c.items()}, 'avg', round(np.mean(list(c.values())),5))
```