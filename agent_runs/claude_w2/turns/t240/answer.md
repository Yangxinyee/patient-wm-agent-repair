The channel I had been building goes through attention head 2 of the core's prelude attention: `make_W_chan` adds an outer product to `core.prelude.0.attn.out.weight`. I never recorded whether it worked. Rather than revive it, I lean toward a simpler head-only edit that I can calibrate and test locally.

The edit would repurpose one hidden unit of the value head, gated so it acts only on glucose codes (and a second one for MAP and norepinephrine). That unit would read a dose signal from the head input and tilt the logits along the bin centres. The policy-predicted part of the dose would be compensated through the per-last-bin `embed_transition` table.

The weak point is factual noise. A decoder error of ~0.6 U/h at gain −2 would add ~1.3 mg/dL of tilt noise, which is about the whole 0.005-nat budget. So the decoder has to be fitted on both factual rows and counterfactual pairs.

To capture head inputs and test weight edits locally, I need to see how `run_rollout_h2` exposes the head (`hook_pre`, `hook_P`, `want_nll`). I'll print only the relevant lines.

```python
s = open('/work/rollout_h2.py').read().split('\n')
idx = [i for i,l in enumerate(s) if ('hook' in l or 'want_nll' in l or 'head_' in l) and not l.strip().startswith('#')]
shown=set()
for i in idx:
    for j in range(max(0,i-1), min(len(s), i+2)):
        if j not in shown: print(j, s[j][:150]); shown.add(j)
```