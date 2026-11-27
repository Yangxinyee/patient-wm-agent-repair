Kind 0 marks starts and measurements, and kind 2 marks END atoms. My rate-based flags may have mixed starts and ENDs, so I'll rebuild the flags from `kind`. Even so, the pattern is clear:
- At drug bundles, head 3 puts weight ≈1 on the u token (5th percentile 1.00), whichever drug is present.
- At glucose reading bundles, head 3 (0.95) and head 2 (0.96) put weight ≈1 on the e token.

This suggests a simple and exact write path for "rate − expected rate for the glucose tier": a single linear readout of head 2 and head 3's attended values into a staging dim. It should give +rate at insulin bundles (from u), −expected rate at glucose reading bundles (from e), and ≈0 elsewhere. In the record, insulin always follows at h:05 a reading ≥150 at h:00, so the pair sums to the random within-tier deviation.

Next, with kind-correct flags, I'll measure how often the attention weight falls below 0.9 for each bundle type. I'll also construct each head's attended value vectors from the prelude norm1 output of the tokens [hf+slots, e, u, z_in]. To check that construction, I'll verify that the attention output it implies reproduces E.

```python
st_=(at2.kind==0); en_=(at2.kind==2)
ins_s=flag((at2.event=='insulin')&st_); ne_s=flag((at2.event=='norepinephrine')&st_)
endf=flag(at2.event.isin(['insulin','norepinephrine'])&en_)
print('insulin start rate notna', at2.loc[(at2.event=='insulin')&st_,'rate'].notna().mean(), 'END rate notna', at2.loc[(at2.event=='insulin')&en_,'rate'].notna().mean())
types={'ins_only':ins_s&~ne_s,'ins+ne':ins_s&ne_s,'ne_only':ne_s&~ins_s,'end_only':endf&~ins_s&~ne_s,'gluc_read':gl,'map_only':mp&~gl,'other':~(ins_s|ne_s|endf|gl|mp)}
for nm_,m_ in types.items():
    if m_.sum()==0: print(nm_,0); continue
    a3u=aw2[m_,3,5]; a3e=aw2[m_,3,4]; a2e=aw2[m_,2,4]; a2u=aw2[m_,2,5]
    print(f"{nm_:9s} n={m_.sum():5d} h3u<.9 {np.mean(a3u<.9):.3f} h3e>.9 {np.mean(a3e>.9):.3f} h2e>.9 {np.mean(a2e>.9):.3f} h2u>.9 {np.mean(a2u>.9):.3f} | h3 e {a3e.mean():.2f} u {a3u.mean():.2f} | h2 e {a2e.mean():.2f} u {a2u.mean():.2f}")
# build tokens and per-head attended values; check vs E
pre_='core.prelude.0.'
def prelude_parts(Pp, hf, e, u, z):
    zin=F.linear(z,Pp['z_in.weight'],Pp['z_in.bias'])
    T_=torch.cat([hf+Pp['core.slots'][None], e[:,None], u[:,None], zin[:,None]],1)
    n1_=F.layer_norm(T_,(192,),Pp[pre_+'norm1.weight'],Pp[pre_+'norm1.bias'])
    kv_=F.linear(n1_,Pp[pre_+'attn.kv.weight']); v_=kv_[...,192:].reshape(len(e),7,4,48)
    return T_, n1_, v_
T2,n12,v2=prelude_parts(P, ST2['hf'], ds2['e'], ds2['u'], ST2['z'])
aws=ST2['aw']  # N,4,7,7
A_=torch.einsum('bhqk,bkhd->bqhd', aws, v2)   # attended values per query token/head
ao_=F.linear(A_.reshape(N2,7,192), P[pre_+'attn.out.weight'])
x_=T2+ao_; x_=x_+FF(F.layer_norm(x_,(192,),P[pre_+'norm3.weight'],P[pre_+'norm3.bias']),pre_+'ffn.',P) if False else x_
print('A_ shape', tuple(A_.shape))
```