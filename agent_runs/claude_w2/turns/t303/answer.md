The head-only route is exhausted. The readout does predict a little of the residual (test R² 0.04–0.08), but only about 20% of the target response fits within the factual budget:
- gain 0.5: bin NLL about +0.002, x2 one-step response −0.4 against a target of −2.4;
- gain 1: bin NLL already +0.09.

The model's internal dose information is too entangled with glucose level.

With 11 runs and 46 forward passes left, the remaining lever is to amplify the dose input itself. My earlier notes found the model under-uses within-band dose variation even teacher-forced: insulin slope −0.45 against −2.0 in the data, norepinephrine +0.94 against +3.6. Amplifying the dose representation may therefore raise the counterfactual response without hurting, or even while helping, the factual fit.

Candidate 1: scale the encoder's numeric-embedding columns for z_dose and z_rate by 3. Only drug atoms carry these features. The forward tool will judge it on the 100 tuning stays I've used before (sel1): first the factual NLL (tolerance 0.005), then the dose-edit readouts against my record-based truths (−14.1, +7.0, +4.7, −2.4).
- **Confirmed:** larger responses with the factual NLL change ≤ 0.005.
- **Refuted:** an NLL increase, or no change in response.

I'll also log the findings so far in the notebook.

```python
with open('/work/notes.md', 'a') as f:
    f.write("\n## Restart 7 (runs 266-290)\n- Head-only readouts (gated MLP unit + logit tilt) FAIL: state's dose signal entangled with glucose level; "
            "within 0.005-nat budget only ~20% of one-step target (D4 teacher-forced arms, 100 train stays). Global tilt explodes tails; virtual last-value shift too weak (0.14/mgdl).\n"
            "- Channel design (chan.py) too big to finish. NEXT: amplify encoder dose numerics (z_dose,z_rate cols) by k, judge with tool on sel1.\n")
nm_num = [n for n in NAMES if 'numeric' in n]; print({n: SHAPES[n] for n in nm_num})
cand = [n for n in nm_num if n.startswith('embed.') and 6 in SHAPES[n]]
print("candidate:", cand)
if len(cand) == 1:
    Wnum = np.asarray(load_original(cand[0]), dtype=np.float32).copy()
    k_amp = 3.0
    if Wnum.shape[-1] == 6: Wnum[:, [2, 4]] *= k_amp
    else: Wnum[[2, 4], :] *= k_amp
    save_edit(cand[0], torch.as_tensor(Wnum)); print("saved", cand[0], Wnum.shape, "edited:", edited())
```