A single linear readout of head 3's 48-dim attended values cannot do both jobs. Every weighting trades rate precision at insulin bundles (residual 0.4–0.8) against nulling the e-token values at reading bundles. The 48-dim value space is fully used, so the readout direction cannot be separated from the e-token directions. A gate built after the attention would add a second LayerNorm division, which also rules out the prelude FFN trick.

There is another option. I can repurpose one of head 3's 48 value channels, giving its `W_v` row a new 192-dim functional d on the norm1 token, rather than reading the existing values. Head 3's attention is unchanged. Its weight on u is 1.0 at every drug bundle, so that channel then outputs d·LN1(u) there. Everywhere else it outputs d·LN1(e or slots). With 192 dims there is room to decode the rate on action tokens while nulling measurement and slot tokens, since they come from different encoders. The model loses only one value channel of head 3.

To test it, I regress the target (insulin rate at drug bundles, 0 elsewhere) on the head-3 attention-weighted norm1 tokens, raw and σ_x-scaled. Success is a residual ≲0.1 at insulin bundles together with ≲0.05 off-target.

```python
G3=torch.einsum('bqk,bkd->bqd', aws[:,3,:4,:], n12)   # N,4 slots,192: head-3 attention-weighted normalized tokens
F3r=G3.mean(1).numpy(); F3s=(G3/sdx[:,:,None]).mean(1).numpy()
for lab_,XX in [('d192 raw',F3r),('d192 /sig',F3s)]:
    for wo_ in [1.0,10.0,100.0]:
        wfit_eval(XX,y_r,wo_,f"{lab_} woff={wo_}")
```