The notebook head gives the targets and an important clue.

**Targets (true effects on the tuning stays):**

| Readout | Truth |
|---|---|
| insulin x2 | -13.8 |
| insulin x1/2 | +6.9 |
| norepinephrine x2 | +4.6 |
| norepinephrine x1/2 | -2.3 |

These come from linear impulse kernels estimated from within-band dose noise. The original model gives only -2.15, +1.23, +0.34 and -0.06.

**Why the earlier amplifications failed.** The architecture shows that, at the next reading, the value head sees the dose only through the core state. The core state is reached through prelude tokens [slots, e, u, z_in(z)] and then layer norms. Insulin's rate is strong in u but gets scrambled downstream.

**A cleaner writer.** I can build one entirely inside `z_post`, gated only by u:
- Unit 88 computes 8 + p_ins·u, which is linear in the insulin rate at drug bundles and off at reading bundles, where u is the empty-action vector (p_ins·u = -19.2).
- Unit 105 does the same for norepinephrine with gain 40.
- Unit 12 is on only at reading bundles and cancels the constant offset.

So z18 equals the insulin rate and z28 the norepinephrine rate, exactly zero-centred elsewhere. Then I amplify z_in's columns 18 and 28 so the written signal is no longer buried by the prelude's normalisation. The written quantity is the rate itself, not an innovation. The reader in the value head will later subtract the expected dose given the last reading through the per-bin transition table. That keeps the factual fit while the counterfactual responds to the full dose change.

**What the next forward pass (states, both models) will test:**
- Confirm: z18 and z28 track the recorded rates and are about zero at reading bundles. Then Δout and q at the following readings move consistently with the rate.
- Refute: the write is not clean (p_ins is stale for the true u), or the state change is negligible or inconsistent.

```python
import numpy as np, json
def A64(n): return np.array(np.asarray(load(n)), dtype=np.float64)
def SAVE(n, a):
    a32 = np.ascontiguousarray(np.asarray(a, dtype=np.float32))
    try:
        save_edit(n, a32)
    except Exception as ex:
        import torch; save_edit(n, torch.from_numpy(a32))
pI = np.load('/work/p_ins.npy').astype(np.float64).ravel(); pN = np.load('/work/p_ne.npy').astype(np.float64).ravel()
S2 = np.load('/work/fwd_2/states_original.npz', allow_pickle=True)
print({k:(S2[k].shape, str(S2[k].dtype)) for k in S2.files})
X2 = [S2[k] for k in S2.files if S2[k].ndim==2 and S2[k].shape[1]==992][0].astype(np.float64)
Z = X2[:,960:992]
m18, s18, m28, s28 = Z[:,18].mean(), Z[:,18].std(), Z[:,28].mean(), Z[:,28].std()
Wz = A64('z_in.weight'); bz = A64('z_in.bias')
tok = Z @ Wz.T + bz; sig = tok.std(1).mean()
print('pI %.1f pN %.1f | z18 %.3f/%.3f z28 %.3f/%.3f | tok sd %.3f |Wz18| %.3f |Wz28| %.3f'%(np.linalg.norm(pI),np.linalg.norm(pN),m18,s18,m28,s28,sig,np.linalg.norm(Wz[:,18]),np.linalg.norm(Wz[:,28])))
W0=A64('z_post.0.weight'); b0=A64('z_post.0.bias'); W2=A64('z_post.2.weight'); b2=A64('z_post.2.bias')
print('W0',W0.shape,'W2',W2.shape)
jG,jI,jN = 12,88,105; BI,BN,Bp,kN = 8.0,8.0,8.0,40.0; ue=-19.2; gread=-Bp-ue
for j in (jG,jI,jN): W0[j,:]=0.0
W0[jI,384:]=pI; b0[jI]=BI
W0[jN,384:]=kN*pN; b0[jN]=BN
W0[jG,384:]=-pI; b0[jG]=-Bp
W2[:,[jG,jI,jN]]=0.0; W2[18,:]=0.0; W2[28,:]=0.0
W2[18,jI]=1.0; W2[18,jG]=BI/gread; b2[18]=m18-BI
W2[28,jN]=1.0/kN; W2[28,jG]=(BN/kN)/gread; b2[28]=m28-BN/kN
FI=0.5*sig/np.linalg.norm(Wz[:,18]); FN=2.0*sig/np.linalg.norm(Wz[:,28])
Wz2=Wz.copy(); bz2=bz.copy(); Wz2[:,18]*=FI; Wz2[:,28]*=FN
bz2 -= (FI-1)*m18*Wz[:,18] + (FN-1)*m28*Wz[:,28]
for n,a in [('z_post.0.weight',W0),('z_post.0.bias',b0),('z_post.2.weight',W2),('z_post.2.bias',b2),('z_in.weight',Wz2),('z_in.bias',bz2)]: SAVE(n,a)
WR = dict(jG=jG,jI=jI,jN=jN,BI=BI,BN=BN,Bp=Bp,kN=kN,ue=ue,FI=FI,FN=FN,m18=m18,m28=m28,sig=sig)
print('FI %.2f FN %.2f'%(FI,FN), 'edited:', edited())
sel = json.load(open('/work/sel_states.json'))
print(len(sel), str(sel)[:1500])
with open('/work/notes.md','a') as f:
    f.write("- run287: ARCH re-read: dose reaches value head only via core state (prelude tokens [h+slots,e,u,z_in(z)] -> E -> blocks/b_in -> coda -> out_norm); view=out+out_norm(mean h_flow)+gap+z_head(z_prior draw); q=query_norm(view+type_emb); with_code adds per-code registers only.\n"
            "- run288: TARGETS (tuning100): ins x2 -13.8, half +6.9; ne x2 +4.6, half -2.3. orig -2.15/+1.23/+.34/-.06.\n"
            "  WRITER v3 saved: z_post.0 unit 88 pre=8+pI.u (ins), 105 pre=8+40*pN.u (ne), 12 pre=-8-pI.u (on only at U_EMPTY, ~11.2); z_post.2 rows 18/28 := rate_I / rate_N (+nat mean) with unit-12 compensation; z_in cols 18/28 x FI,FN (0.5/2.0 normalized-token per unit rate), bias compensated. NEXT: states both on sel_states -> check z18/z28 = rates, measure dq at readings; then value-head reader (rate minus E[rate|last bin] via embed_transition).\n")
```