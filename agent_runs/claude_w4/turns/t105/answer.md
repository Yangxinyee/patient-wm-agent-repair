`encode_action` has two 768-unit MLPs (mlp1 per atom, mlp2 on the pooled set), a size term and a LayerNorm. Drug atoms vary along only a couple of one-dimensional curves, so many of the 768 hidden units in each MLP should never activate for any drug bundle. Since this encoder only ever sees drug atoms, those units are completely free to repurpose.

This run does three things:
1. Rebuild u_k from raw weights to confirm the exact computation.
2. Count units that are dead (pre-activation below −4) across every grid bundle, in mlp1 and mlp2.
3. Measure how much the LayerNorm scale σ(y) varies across bundles. A channel added before the LayerNorm gets divided by σ, so large variation would distort the readout.

```python
Wt = {k: load(k).double() for k in NAMES if k.startswith('encode_action')}
def gel(x): return 0.5*x*(1+torch.erf(x/np.sqrt(2)))
def mlp1_h(A): return A @ Wt['encode_action.mlp1.0.weight'].T + Wt['encode_action.mlp1.0.bias']
def mlp1_o(A): return gel(mlp1_h(A)) @ Wt['encode_action.mlp1.2.weight'].T + Wt['encode_action.mlp1.2.bias']
def enc_x(atoms):
    A = torch.stack([torch.as_tensor(a, dtype=torch.float64).ravel() for a in atoms]); n = len(atoms)
    return mlp1_o(A).sum(0)/np.sqrt(n) + Wt['encode_action.size.weight'][:,0]*np.log(1+n) + Wt['encode_action.size.bias']
def mlp2_h(X): return X @ Wt['encode_action.mlp2.0.weight'].T + Wt['encode_action.mlp2.0.bias']
def enc_y(X): return gel(mlp2_h(X)) @ Wt['encode_action.mlp2.2.weight'].T + Wt['encode_action.mlp2.2.bias']
def LN_(Y): mu = Y.mean(-1, keepdim=True); sd = ((Y-mu)**2).mean(-1, keepdim=True).add(1e-5).sqrt(); return (Y-mu)/sd*Wt['encode_action.norm.weight'] + Wt['encode_action.norm.bias'], sd.squeeze(-1)
atoms_g = []
for a_, b_ in zip(GI, GN):
    at = []
    if not np.isnan(a_): at.append(ins_atom(float(a_)))
    if not np.isnan(b_): at.append(ne_atom(float(b_)))
    atoms_g.append(at)
Xg = torch.stack([enc_x(at) for at in atoms_g[:-1]]); Yg = enc_y(Xg); Ug, sdg = LN_(Yg)
print('replica err', float((Ug - torch.as_tensor(GU[:-1])).abs().max()))
Aall = torch.stack([torch.as_tensor(ins_atom(float(r)), dtype=torch.float64).ravel() for r in rI] + [torch.as_tensor(ne_atom(float(r)), dtype=torch.float64).ravel() for r in rN])
h1max = mlp1_h(Aall).max(0).values; h2max = mlp2_h(Xg).max(0).values
print('mlp1 dead(<-4):', int((h1max < -4).sum()), ' mlp2 dead(<-4):', int((h2max < -4).sum()))
for nmz, msk in [('I only', onlyI[:-1]), ('I+N', bothm[:-1]), ('N only', onlyN[:-1])]:
    s_ = sdg[torch.as_tensor(msk)]; print(nmz, 'sigma(y) min %.3f max %.3f' % (float(s_.min()), float(s_.max())))
```