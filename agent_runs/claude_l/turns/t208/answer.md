These numbers settle the memory question. A unit-size perturbation of the z token moves the next context by only about 0.03. The best decoding direction has a signal-to-noise ratio of 0.27 per unit injection. The loop gain also varies by about ±44% across bundles (the existing pair, `DB1`/`DEC1`, is worse still).

A memory refreshed through the z token at every bundle would pick up a random gain error of that size at each of the ~24 refreshes in 12 hours. It would also accumulate history-dependent "phantom" signal. **The multi-hour z-loop memory is not feasible.**

So I will build the simpler, robust edit (Stage 1):
- At each drug bundle, z_post computes the residual dose (dose decoded from u, minus the policy dose decoded from the context). It does this separately for insulin and NE, gated on the drug being present.
- It outputs a z token that is a copy of u plus κ × residual along that drug's dose direction.
- z_in maps this back to token space through a PCA basis of u.

The copy test showed a z token copying u costs essentially no factual NLL. A single gain κ per drug will be calibrated on the offline rollout replica to the record-based truth (insulin ×2 ≈ −14, NE ×2 ≈ +5 on batch 1).

The shape of the response over time can't be made fully right without memory. Instead, κ is set so that the 12-hour mean matches the main readouts.

First I need the exact tensor shapes of z_post, z_in and encode_action, the contents of the teacher-forced training file, and the call signature of my rollout replica. Then I can build and test the weights directly in the replica.

```python
for n in NAMES:
    if n.startswith('z_') or 'encode_action' in n: print(n, SHAPES[n])
d_=np.load('/work/tf_train.npz', allow_pickle=True)
print({k:d_[k].shape for k in d_.files})
src=open('/work/wm2_rollout.py').read()
i=src.find('def rollout'); print(src[i:i+900])
print('pi_i b=%.3f |w|=%.3f ; pi_n b=%.3f |w|=%.3f'%(pi_i[1],float(pi_i[0].norm()),pi_n[1],float(pi_n[0].norm())))
```