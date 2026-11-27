The best dimensions score about 3.5 times below the median: dimension 49 at 2.34, then 9, 185 and 149, against a median of 8.3. None of them is truly unused, so freeing two of them has a cost that I have to measure on the real model.

Next I'll apply a "skeleton" edit for dimensions 49 and 185, which disconnects them completely:
- They are no longer read by any norm, `c_out`/`b_in` column, or `out_norm`.
- They are no longer written by any attention or FFN output.
- `c_out` passes each of them straight through.

There is no injection yet. A factual pass will then show what this costs. If the NLL rises by less than about 0.002, these two dimensions can serve as the code-8 and code-9 memory channels. If it rises more, I'll try other dimensions or fewer of them.

```python
import torch, json
def skeleton(dims):
    ed={}
    def g(n):
        if n not in ed: ed[n]=load_original(n).clone()
        return ed[n]
    for j in dims:
        for L in ['prelude.0.norm1','prelude.0.norm3','blocks.0.0.norm1','blocks.0.0.norm2','blocks.0.0.norm3','blocks.0.1.norm1','blocks.0.1.norm2','blocks.0.1.norm3','coda.0.norm1','coda.0.norm3','out_norm']:
            g('core.'+L+'.weight')[j]=0; g('core.'+L+'.bias')[j]=0
        for w in ['blocks.0.0.attn.out','blocks.0.0.cross.out','blocks.0.1.attn.out','blocks.0.1.cross.out','coda.0.attn.out','prelude.0.attn.out']:
            g('core.'+w+'.weight')[j,:]=0
        for f in ['blocks.0.0.ffn.2','blocks.0.1.ffn.2','coda.0.ffn.2','prelude.0.ffn.2']:
            g('core.'+f+'.weight')[j,:]=0; g('core.'+f+'.bias')[j]=0
        co=g('core.c_out.weight'); co[:,j]=0; co[j,:]=0; co[j,j]=1
        bi=g('core.b_in.weight'); bi[:,j]=0; bi[j,:]=0
    return ed
ED=skeleton([49,185])
for n,t in ED.items(): save_edit(n,t)
sel=json.load(open('/work/sel100.json')); print(len(ED), len(edited()), len(sel), sel[:3])
```