None of the memory variants helps; the documented convention (memory of e only) is still the best fit. I have now ruled out bin conventions, the u encoding, z, precision, memory and e-token handling. The tool differs from my local step only at drug bundles, by about 1e-3 in the final state. My decoders imply this corresponds to a difference of 2% or more in the prelude FFN input n3.

Rather than keep guessing conventions, I'll localise the discrepancy at j3. I'll compute the Jacobian of the output state with respect to the prelude output P (4×192), before the prelude norm, and solve a ridge least-squares for the δP that explains tool-minus-local.
- If a modest δP (a few % of |P|) explains it almost exactly in the nonlinear re-check, the difference arises in the prelude. Its structure (same across slots, or aligned with particular units or tokens) will point to the cause.
- If no small δP works, the difference is in the blocks or coda, for example the cross-attention.

```python
def msx_dP(d, dP=None, W=None):
    globals()['WCUR'] = W if W is not None else W64
    out0,h0,dt,e,u,Mv,Ma = d['out0'],d['h0'],d['dt'],d['e'],d['u'],d['Mv'],d['Ma']
    hf = h0*torch.exp(-dt*torch.exp(Wt('core.flow.a'))); gp = mlp2_(torch.tensor([math.log(dt)/10.0],dtype=torch.float64),'gap')
    ctx = out0 + LNn(hf.mean(0),'core.out_norm') + gp; z = mlp2_(torch.cat([ctx,e,u]),'z_post')[:32]
    toks = torch.cat([hf+Wt('core.slots'), e[None], u[None], linb(z,'z_in')[None]])
    n1 = LNn(toks,'core.prelude.0.norm1'); ao,att = mha_(n1,n1,'core.prelude.0.attn'); x1 = toks+ao
    n3 = LNn(x1,'core.prelude.0.norm3'); P = (x1 + linb(F_.gelu(linb(n3,'core.prelude.0.ffn.0')),'core.prelude.0.ffn.2'))[:4]
    if dP is not None: P = P + dP
    return P, ao, att, n3, x1
d3 = prep(3, Xo5)
P0, ao0, att0, n30, x10 = msx_dP(d3)
print('|P| per slot', P0.norm(dim=1).numpy().round(2), '|ao| per slot', ao0[:4].norm(dim=1).numpy().round(2), '|x1|', x10[:4].norm(dim=1).numpy().round(2))
print('att (heads x slot0 over 7 tokens)', att0.shape)
# Jacobian wrt P via dP hook: use mystep's dP arg (adds to P before prelude_norm)
globals()['WCUR']=W64; base = st_vec(mystep(d3)); Jp = np.zeros((992,768)); eps_=1e-4
for i in range(768):
    dP_ = torch.zeros(4,192,dtype=torch.float64); dP_.view(-1)[i] = eps_
    Jp[:,i] = (st_vec(mystep(d3, dP=dP_)) - base)/eps_
dl = Xo5[idx[3]] - base
for lam in [1e-6, 1e-3, 1e-1, 1.0]:
    A = Jp.T@Jp + lam*np.eye(768); sol = np.linalg.solve(A, Jp.T@dl)
    nl = st_vec(mystep(d3, dP=torch.tensor(sol.reshape(4,192))))
    print('lam', lam, '|dP|', np.linalg.norm(sol).round(3), 'lin resid', (np.linalg.norm(dl-Jp@sol)/np.linalg.norm(Xo5[idx[3]])).round(8), 'nonlin err', (np.linalg.norm(Xo5[idx[3]]-nl)/np.linalg.norm(Xo5[idx[3]])).round(8))
SOLP = sol
```