At every drug bundle (2,319 of them), e_k is reproduced exactly from the set of infusion records ending at that minute. All records start at h+0.0833 and END atoms carry no rate. Since ctx and u_k are also exact, z_post's whole input is now computable on CPU wherever my edit will act.

Next I'll check that this reproduces the original model's z posterior at drug bundles in fwd_3, which is the same 100 stays with unedited weights. That validates the whole z_post computation, including the order of the mean and log-sd outputs. In the same run I'll gather what the unit design needs:
- each hidden unit's pre-activation distribution at drug bundles;
- each unit's outgoing weight norm to the z mean;
- the glucose and MAP news directions in z (coefficients per mg/dL and per mmHg of surprise).

z matching fwd_3 to about 1e-5 would confirm everything. A mismatch would mean mean and log-sd are split the other way round.

```python
open('/work/notes.md','a').write("""
- ctx = out_prev + out_norm(mean_slots(h_prev*exp(-dt_s*exp(flow.a)))) + gap(log(dt_s)/10)  [EXACT, dt in seconds]
- e_k at drug bundles EXACT: encode({END atoms}); END atom = code[c]+kind[2]+ekind[2]+kq[0]+numeric.bias; END for records started 1h earlier.
  All drug records start at h+0.0833. NE atom: unit[4], rate_unit[3], dose=4.8*rate. Measurement-atom e_k NOT replicated (err ~.33) - not needed.
- z posterior encodes reading surprise: R2(z->surprise) finger .85, lab .73, art .79, cuff .77 (glucose dims 20,27,9,30; MAP dims 13,22,2,23).
- PLAN: repurpose z_post hidden units gated on u_k (drug bundles only) to add kappa*d_news*(rate - E[rate|ctx]) to z mean.
""")
OUT3 = torch.tensor(X3[:, :192]); H3 = torch.tensor(X3[:, 192:960]).reshape(-1,4,192)
def ctx_of(OUT, H, i_prev, dt_s):
    hdec = H[i_prev] * torch.exp(-torch.tensor(dt_s, dtype=torch.float32)[:,None,None]*torch.exp(ta)[None,None,:])
    return OUT[i_prev] + LNf(hdec.mean(1), onw, onb) + gapf(torch.log(torch.tensor(dt_s, dtype=torch.float32))/10).T
def u_bundle(s_, h_):
    st = starts.get((s_, h_), []); atoms = []
    for c_ in st:
        r_ = float(drec[(drec.stay_id==s_)&(drec.hk==h_)&(drec.event==c_)].rate.iloc[0])
        atoms.append(ins_atom(r_) if c_=='insulin' else ne_atom(r_))
    return u_set(atoms) if atoms else enc_set('encode_action', torch.zeros(0,192))
dr3 = np.where((S3['has_insulin'].astype(bool) | S3['has_norepinephrine'].astype(bool)) & np.r_[False, sid3[1:]==sid3[:-1]])[0]
dt3 = (hr3[dr3] - hr3[dr3-1])*3600.0
CTX3 = ctx_of(OUT3, H3, dr3-1, dt3)
EU3 = torch.stack([torch.cat([e_drug_bundle(sid3[i], round(float(hr3[i]),4))[0], u_bundle(sid3[i], round(float(hr3[i]),4))]) for i in dr3])
XIN3 = torch.cat([CTX3, EU3], 1)
P3 = XIN3 @ load('z_post.0.weight').T + load('z_post.0.bias'); G3 = gelu(P3)
ZO3 = G3 @ load('z_post.2.weight').T + load('z_post.2.bias')
print('z mean match (first half):', float((ZO3[:, :32] - torch.tensor(Z3[dr3])).abs().max()), ' second half:', float((ZO3[:, 32:] - torch.tensor(Z3[dr3])).abs().max()))
W2m = load('z_post.2.weight')[:32]
imp = (G3.std(0) * W2m.norm(dim=0)).numpy()
print('unit importance at drug bundles: min', np.sort(imp)[:6].round(4), 'median', np.median(imp).round(4))
print('pre-act mean range', P3.mean(0).min().item(), P3.mean(0).max().item())
```