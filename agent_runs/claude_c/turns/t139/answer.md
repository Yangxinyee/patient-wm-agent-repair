The out-of-sample SNR of the drug imprint in the state at glucose queries is only 0.07–0.22. Any edit that reads the drug signal from out, h or ctx adds 2–4 times more noise than signal, so head-side and state-side remaps are ruled out for a full fix.

The only selective entry point is the action-token path. One candidate is `embed.kq` (4×192), which the architecture doc never mentions. If it is a per-token-type embedding for the prelude, its action-token row would be a lever that acts only on drug bundles, for example on their attention weight. I'll check vocab.json's config for what `kq`, `ekind` and `kind` index.

```python
with open('/work/notes.md','a') as f:
    f.write("- Run113: dose-noise imprint in OUT at glucose query: out-of-sample generalized SNR 0.07-0.22 -> head-side/state remaps add 2-4x more noise than signal. Only selective entry = action token path.\n")
V=VOCAB; print(list(V.keys()))
cfg=V.get('config',{}); print({k:cfg[k] for k in list(cfg)[:40]})
for k in V:
    if 'kq' in k.lower() or 'kind' in k.lower() or 'type' in k.lower(): print(k, str(V[k])[:300])
```