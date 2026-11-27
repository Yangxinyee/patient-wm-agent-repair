The hourly profiles show that both edits raise the response by about the same factor at every lag. The lag-1 shift goes from −0.51 to about −0.8, against a true value near −2.9. The model's time shape is fine; only its per-step gain is too low. Next I will find which factual-NLL cost of head-0 ×4 (pass 22) comes from readings right after drug bundles and which from other readings, split by code. That tells me whether the cost comes from the amplified drug effect itself or from disturbing reading-driven predictions.

```python
for k in range(1,24):
    fs=glob.glob(f'/work/fwd_{k}/factual_*.csv')
    if fs: print(k, [f.split('/')[-1] for f in fs], json.load(open(f'/work/fwd_{k}/summary.json')).get('split'), len(json.load(open(f'/work/fwd_{k}/summary.json')).get('results',{})))
```