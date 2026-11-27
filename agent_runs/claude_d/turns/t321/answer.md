Run 287 showed three things:
- **fwd15 and fwd16** were both factual "both" checks on the same 100 tuning stays (5,157 readings, original NLL 1.51191, same as sel1/fwd9). They were probably cost checks of E5 and of a variant. I haven't read their edited NLLs yet.
- **The `.npz` files** were saved with positional names (arr_0..3), so name-based comparison failed. I need to match them by shape instead.
- **The current edit** is head-only. The E5 encoder units are not in the model now.

Next I will:
- read the edited factual NLL in fwd15 and fwd16;
- map each saved array to its tensor by shape, and compare it with both the current and the original weights.

If one of fwd15/16 shows E5 within budget (≤ +0.005 over 1.51191), the encoder edit was dropped by accident and should be restored. If both are over budget, the drop was deliberate and I should stay with the head-only edit.

```python
import json, numpy as np
for k in (15,16):
    s=json.load(open(f'/work/fwd_{k}/summary.json'))['results']
    for m in ('original','edited'):
        r=s[m]; print(k, m, r['value_nll_per_value'], {e[:6]:round(v['nll'],4) for e,v in r['by_event'].items()})
heads=['attr_heads.value_bin.0.bias','attr_heads.value_bin.0.weight','attr_heads.value_bin.2.bias','attr_heads.value_bin.2.weight']
encn=[n for n in NAMES if n.startswith('encode_action.mlp1')]
print('enc', [(n[-14:], tuple(SHAPES[n])) for n in encn])
for fn in ('E3b.npz','E4.npz','E5enc.npz'):
    z=np.load('/work/'+fn); out=[]
    for a in z.files:
        arr=z[a]; res=[]
        for n in heads+encn:
            if tuple(SHAPES[n])==arr.shape:
                res.append((n[-18:], round(float(np.abs(np.asarray(load(n))-arr).max()),4), round(float(np.abs(np.asarray(load_original(n))-arr).max()),4)))
        out.append((a, arr.shape, res))
    print(fn, out)
```