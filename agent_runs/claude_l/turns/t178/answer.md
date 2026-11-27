The batched step reproduces the recorded states (max error 9e-5). `u_empty` is not zero: bundles without a drug get the empty-set action encoding, as `enc_eu2` already returns. Now I'll write the batched dose-edit rollout. Up to hour 12 the record is teacher-forced. Drug atoms from hour 12 on have rate and dose scaled per condition. From hour 13 the readings are drawn from the value head with common random numbers across conditions, with the predicted mean averaged over 4 prior draws of z. First I verify my batched encoder against `enc_eu2` and time the rollout on one stay. If the encoder matches to ~1e-6, I'll then check the rollout shifts against the tool's fwd_1 output for the original model.

```python
EDG = {c: np.array(NUMV[(nm,'value')]['fine_bin_edges'],dtype=np.float64) for c,nm in MEASC.items()}
MED = {c: (NUMV[(nm,'value')]['median'], NUMV[(nm,'value')]['iqr']) for c,nm in MEASC.items()}
def split_fix(atoms):
    A=[fix_atom2(a) for a in atoms]
    ea=[a2 for a,a2 in zip(atoms,A) if not (a['act'] and a['kind']==0)]
    ua=[a2 for a,a2 in zip(atoms,A) if (a['act'] and a['kind']==0)]
    return ea, ua
# verify reconstruction of enc_eu2
s=ks[0]; mx=0
for k in range(len(BUN[s])):
    ea,ua=split_fix(BUN[s][k][1]); e1=set_enc(W0,ea,'encode') if ea else None
    u1=set_enc(W0,ua,'encode_action') if ua else u_empty
    e0,u0=enc_eu2(BUN[s][k][1])
    if e1 is None: print('no e atoms at', k); continue
    mx=max(mx, float((e1-e0).abs().max()), float((u1-u0).abs().max()))
print('recon err', mx)
def bset_enc(W, rows, pre):
    # rows (B,n,192)
    n=rows.shape[1]; s_=bmlp(W,rows,pre+'.mlp1').sum(1)/math.sqrt(n)
    s_=s_+blin(W,torch.tensor([[math.log(1+n)]],dtype=T),pre+'.size')
    return bln(W,bmlp(W,s_,pre+'.mlp2'),pre+'.norm')
# check batched set enc equals set_enc
ea,ua=split_fix(BUN[s][4][1]); R=atom_matrix(W0,ea)[None]
print('bset err', float((bset_enc(W0,R,'encode')[0]-set_enc(W0,ea,'encode')).abs().max()))
print('glucose edges', EDG[6][:5], '...', EDG[6][-3:], 'centres', CENd[6][:4], CENd[6][-3:])
print([ (a['code'],a['kind'],a['vb'],np.round(a['num'],2)) for a in BUN[s][4][1]])
```