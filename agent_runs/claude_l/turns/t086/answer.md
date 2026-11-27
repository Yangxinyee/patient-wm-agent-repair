The embedding is not the problem. Every unused table has a zero row 0 (padding). The rows with norm ≈ 0.26–0.29 look untrained (initialisation only), and `kq[0]` is the one trained row of `kq`.

So the k=0 mismatch must come from the initial state. Earlier I had stored `h0 = core.slots`, which suggests I suspected the initial `h` is the learned slot vectors rather than zero. In that case `out_init` would be `out_norm(mean(slots))` and ctx would differ.

Next I rerun the first-step z search with `h_init` = slots or zero, and `out_init` = zero, `out_norm(slots)` or `out0`, over dt candidates and activations. A near-exact z match pins the initial state and the activations. If there is still no match, the z_post input itself, i.e. the e encoding, is the suspect.

```python
print('out0 vs out_norm(mean slots)', (out0.double()-lnorm(W0,W0['core.slots'].mean(0),'core.out_norm')).abs().max().item())
best0=[]
for a1 in names_a:
  for a2 in names_a:
    ACTMAP.clear(); ACTMAP.update({'encode.mlp1':a1,'encode.mlp2':a2,'encode_action.mlp1':a1,'encode_action.mlp2':a2})
    e0,u0=enc_bundle(a0)
    for ga in names_a:
      ACTMAP['gap']=ga
      for dtc in [1.0,60.0,3600.0,7200.0,86400.0]:
        gv = mlp(W0,torch.tensor([math.log(dtc)/10],dtype=T),'gap')
        for hi in ['slots','zero']:
          h_i = W0['core.slots'] if hi=='slots' else torch.zeros(4,192,dtype=T)
          hf = h_i*torch.exp(-dtc*torch.exp(W0['core.flow.a']))
          for oi in ['zero','onslots','out0']:
            o_i = torch.zeros(192,dtype=T) if oi=='zero' else (lnorm(W0,W0['core.slots'].mean(0),'core.out_norm') if oi=='onslots' else out0.double())
            ctx = o_i + lnorm(W0,hf.mean(0),'core.out_norm') + gv
            for za in names_a:
              ACTMAP['z_post']=za
              zo=mlp(W0,torch.cat([ctx,e0,u0]),'z_post')
              best0.append(((zo[:32]-Xt[0,960:]).abs().max().item(),a1,a2,ga,dtc,hi,oi,za))
best0.sort(key=lambda r:r[0])
for r in best0[:8]: print(r)
```