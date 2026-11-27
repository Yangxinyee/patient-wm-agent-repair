Under realistic constraints the E-level SNR collapses. With q kept orthogonal to the bundle-type means and to the dose directions, a single q shared by all 4 slots, and natural variation at reading bundles counted as noise, decode noise is about 1.0 per unit token norm for both drugs. That is no better than reading from the base (1.33).

The z token cannot move much more than about 5 units before LayerNorm compresses both the signal and the copy of u. So any design through the prelude carries 0.3–1 U/h of state-dependent noise.

A b_in amplifier would also fire at every bundle. That makes the head-side variant better: z_post writes, z_prior decodes with a gelu gate that can switch off away from post-infusion bundles, and z_head injects into the view.

I also have a budget to use. The factual fit may worsen by up to 0.005 nats, so a noisy but unbiased dose channel may be affordable. Whether it is depends on how sharp the model's predictions are, since a prediction error of sd b costs about b²/2σ² nats.

I'll measure σ per reading type from the factual output already on disk (fwd_4): the spread of reading minus predicted mean, and the mean NLL. A glucose σ around 8–10 mg/dL would make about 0.5 mg/dL of injection noise cost only about 0.001 nats.

```python
import pandas as pd, numpy as np, os
fp = '/work/fwd_4/' + [f for f in os.listdir('/work/fwd_4') if f.startswith('factual')][0]
F4 = pd.read_csv(fp); print(fp, F4.shape, list(F4.columns)[:12])
vc = [c for c in F4.columns if c in ('value', 'reading', 'y', 'observed')]; pc = [c for c in F4.columns if 'mean' in c]; nc = [c for c in F4.columns if 'nll' in c.lower()]
ec = [c for c in F4.columns if c in ('event', 'code', 'event_name')]
print('cols used', vc, pc, nc, ec)
if vc and pc and ec:
    for ev, g_ in F4.groupby(ec[0]):
        r_ = g_[vc[0]] - g_[pc[0]]
        print(ev, 'n %d resid sd %.2f mean %.2f nll %.3f' % (len(g_), r_.std(), r_.mean(), g_[nc[0]].mean() if nc else float('nan')))
```