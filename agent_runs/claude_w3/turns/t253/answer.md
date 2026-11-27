The helpers are all still there: a bundle encoder, a prelude-token builder (`pre_tokens`), attention probabilities (`pre_probs`) and the z posterior (`zpost_mean`). The earlier per-head output (0.623 for head 0, about 0.35 for the others) was most likely the mean prelude attention from slots to the u token. If so, head 0 is the natural carrier for injected rate information.

Next I'll check that the local bundle lists line up with the saved states rows and what shapes the encoder returns. Then I can compute the attention weights and the u readout at the infusion bundles. If the bundle counts match the row counts per stay, the local pipeline can be driven from the true saved states.

```python
sids=np.unique(D['stay_id']); s0=int(sids[0])
B=stay_bundles(s0); print(len(B), int((D['stay_id']==s0).sum()))
r=enc_bundle(B[1]); print([np.shape(x) for x in r]); print(list(B[1].columns))
```