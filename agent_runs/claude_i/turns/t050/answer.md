I have helper functions from the last session (`lin`, `ln`, `mlp`, `encode_set`, `atom_vec`, `zpost_cand`), but their source is gone. Rather than rely on them, I'll write any new replica code to a file in /work so it survives restarts.

Run 45 strongly favours a head-level correction:
- The hour-13 rollout shift equals the model's teacher-forced one-step sensitivity: −0.74 / 1.75 ≈ −0.42 per u/h for x2, −0.46 for x1/2.
- The residual test implies the same −0.42 within the factual noise range, since −2.0 + 1.59 = −0.41.

So the model's dose response is roughly linear and about 5× too weak, and the missing part is a linear function of (dose − policy-expected dose).

An edit confined to the value head leaves teacher-forced states unchanged. That means I can compute its factual NLL effect exactly offline from saved states, if I can replicate the head. In rollouts, the corrected readings then propagate through the model's own dynamics.

Next, I will inspect the stored states file (fwd_3): which stays, which bundles, and the layout of the 992 numbers per bundle. Then I'll check whether it overlaps the factual stays (fwd_2), so I can validate a local head replica against the tool's predicted means.

```python
z3 = np.load('/work/fwd_3/states_original.npz', allow_pickle=True)
print(z3.files)
for k in z3.files:
    a = z3[k]; print(k, a.shape, a.dtype, a[:3] if a.ndim==1 else '')
print(json.dumps(json.load(open('/work/fwd_3/summary.json')))[:1500])
st_ids = np.unique(z3['stay_id']) if 'stay_id' in z3.files else None
print("n stays", None if st_ids is None else len(st_ids), "overlap with fwd_2 stays:", None if st_ids is None else len(set(st_ids)&set(fa2.stay_id.unique())))
```