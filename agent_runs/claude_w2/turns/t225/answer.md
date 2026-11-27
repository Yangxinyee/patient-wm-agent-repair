The head code makes the gate constraint clear.
- **Code-specific input:** the only code-specific input to the value MLP is the code embedding added into with_code, and embed.code is shared with the encoder, so I cannot edit it.
- **Consequence for the gate:** gating has to come from the hidden unit's input weights, using a discriminant direction between the glucose and MAP code embeddings that is nearly blind to the rest of with_code.
- **Constant offsets:** those can be cancelled per (code, last bin) in `embed_transition.value_bin`, which is used only by the head.
- **Units:** the value MLP is 192 → GELU 192 → 17, so repurposing two hidden units is feasible.

Next I build the training data for the decoders: teacher-forced passes of the original model on 900 train stays, disjoint from the 100 oracle-evaluation stays.
- **Arms:** the recorded doses, two arms with independent random per-infusion multipliers (log-normal, 15% zeros, up to 3.5x) for both drugs, and one arm with a per-stay constant multiplier (0, 0.5, 1.5, 2 or 3) from a random start hour. This mimics the scored interventions.
- **Stored per reading:** the head input with_code (with z at the prior mean, and with one prior draw), the exact dose lags 1–5 for both drugs under that arm, and the policy lags p(g_{t−k}) from the factual glucose.

The test comes next run. If the doses are decodable from with_code, a held-out ridge fit of the combined target will be good (R² well above 0.5) and will respond about one-for-one to the arm differences.

```python
exec(open('/work/cf.py').read())
rng=np.random.default_rng(123)
all_ids=np.array(sorted(tr.stay_id.unique())); excl=set(list(ids_o))|set(list(selT))
has_i=set(tr[tr.event=='insulin'].stay_id); has_n=set(tr[tr.event=='norepinephrine'].stay_id)
cand=[s for s in all_ids if s not in excl]
drug_c=[s for s in cand if (s in has_i) or (s in has_n)]; none_c=[s for s in cand if (s not in has_i) and (s not in has_n)]
idsT=sorted(list(rng.choice(drug_c,720,replace=False))+list(rng.choice(none_c,180,replace=False)))
t0=time.time()
R0=prep_rollout(tr,idsT); A=R0['A']; si_=A.si.values; t_=A.t.values
isI=((A.code==CODE['insulin'])&(A.kind==0)).values; isN=((A.code==CODE['norepinephrine'])&(A.kind==0)).values
def rmult(n):
    m=np.clip(np.exp(rng.normal(0,0.7,n)),0,3.5); m[rng.random(n)<0.15]=0.0; return m
armsT=[{}]
for _ in range(2):
    mi=np.ones(len(A)); mn=np.ones(len(A)); mi[isI]=rmult(isI.sum()); mn[isN]=rmult(isN.sum()); armsT.append({'ins_mult':mi,'ne_mult':mn})
S_=len(R0['stays']); h0=rng.integers(4,17,S_); fI=rng.choice([0,0.5,1.5,2,3],S_); fN=rng.choice([0,0.5,1.5,2,3],S_)
armsT.append({'ins_mult':np.where(isI&(t_>=h0[si_]),fI[si_],1.0),'ne_mult':np.where(isN&(t_>=h0[si_]),fN[si_],1.0)})
RT,resT,DT=cf_collect(tr,idsT,armsT,W)
assert len(RT['A'])==len(A) and (RT['A'].t.values==t_).all()
DGiT=dose_stack(RT,armsT,'insulin'); DGnT=dose_stack(RT,armsT,'norepinephrine')
tauT=np.round(DT['t']).astype(int)
dI=lag_abs(DGiT,DT['arm'],DT['si'],tauT,5); dN=lag_abs(DGnT,DT['arm'],DT['si'],tauT,5)
hkT=make_lin_oracle(RT,DGiT,a_i,c_i,k_i,PF,t_hide=1e9)
pI=hkT.plags(DT['n'],DT['si'],tauT,None,5)
codesT=torch.as_tensor(DT['code'],dtype=torch.long); st0=torch.zeros(len(codesT),dtype=torch.long)
pr_=mlp(DT['base'],W,'z_prior'); zls=pr_[:,32:]
print('zmu check',float((pr_[:,:32]-DT['zmu']).abs().max()))
lg0,wc0=head_value_logits(DT['base'],codesT,DT['f8'],DT['lb'],st0,W,DT['zmu'],1)
g_=torch.Generator().manual_seed(7)
_,wc1=head_value_logits(DT['base'],codesT,DT['f8'],DT['lb'],st0,W,DT['zmu']+torch.exp(zls)*torch.randn(DT['zmu'].shape,generator=g_),1)
print('rows',len(codesT),{c:int((DT['code']==CODE[c]).sum()) for c in MEAS},'time',round(time.time()-t0,1))
print('P vs lg0 softmax diff', float((lg0.softmax(-1)-DT['P']).abs().mean()))
```