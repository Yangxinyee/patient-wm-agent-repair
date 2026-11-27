The NE breakdown changes the picture.

- **Recorded NE rates pile up at 1.0.** That is the training maximum, and the 90th and 99th percentiles are both 1.0. 203 of the 325 reached rows had a recent rate in (0.75, 1.0]. Under x2 those rows go to about 2.0, which is far outside anything the model saw for NE.
- **High-rate rows:** the old NE readout pushed the right way under x1/2 (−2.3) but not under x2 (−0.15).
- **Low-rate rows (≤0.25):** it pushed the wrong way under both edits (x2 −0.92, x1/2 +0.80).

So the old NE readout was not a dose signal at all at low rates. And NE x2 is mostly an out-of-range problem, which a readout fitted on recorded states cannot calibrate.

One way out is to compute the model's state under the edited NE doses myself: run the model's own weights forward on CPU, with no gradients and no change to the weights. That would let me see how the state moves when an NE rate goes to 2.0. Recorded NE never reaches 2.0, so a term that responds only to that out-of-range movement should cost almost nothing on factual fit.

To judge whether that is feasible in my remaining runs, I first need the tensor inventory and config. Then I can check whether the core can be reproduced exactly and checked against the tool's saved states.

```python
from collections import OrderedDict
groups=OrderedDict()
for n in NAMES:
    p='.'.join(n.split('.')[:2]); groups.setdefault(p,[]).append((n,tuple(SHAPES[n])))
for p,l in groups.items():
    print(p, '; '.join(f"{n[len(p)+1:] or n}:{'x'.join(map(str,s))}" for n,s in l))
print(VOCAB['config'])
print([k for k in VOCAB.keys()])
```