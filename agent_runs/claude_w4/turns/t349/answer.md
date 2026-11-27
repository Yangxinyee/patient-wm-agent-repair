v10b failed the factual check badly. On the 100 selected stays, value NLL rose from 1.494 to 2.010, which is +0.52 nats, against an allowance of +0.005. All four events got worse, by 0.38 to 0.82 nats.

This is worse than every earlier version I tested; all of them also failed, by +0.17 to +0.25 nats. The way I carry dose information inside the model disturbs it at bundles with no drug, and zeroing the readout on approximate token mixtures made this worse, not better. I have only 2 Python runs left. That is not enough to design and check a new, robust edit and still keep a run in reserve to undo it.

An edit that misses the factual allowance fails outright. So the responsible choice is to go back to the original weights, which pass the factual check exactly. Their dose answers on the selected stays are about −0.35, +0.21, +0.02 and −0.25, against my record-based truth of −12.96, +6.48, +4.94 and −2.47. That is poor, but valid. v10b stays saved in /work in case it is useful later.

Next I will drop all edits and log this. The step works if the edited list comes back empty and the tensors match the originals exactly. A later paired factual pass should then show identical NLL for both models.

```python
import numpy as np
names = list(edited())
for n in names:
    drop_edit(n)
print('dropped', names, '| now edited:', edited())
print('max diff vs original', max(float(np.abs(np.asarray(load(n)) - np.asarray(load_original(n))).max()) for n in names))
with open('/work/notes.md', 'a') as f:
    f.write("\n- fwd16 = factual both sel100 v10b: NLL 1.4938 -> 2.0104 (+.517; finger +.56 lab +.82 art +.48 cuff +.38). v10b FAILS; carrier design fragile off drug bundles.\n"
            "- run299: DROPPED all edits (back to original weights) to guarantee factual fit; v10b kept in /work/v10b_tensors.npz. Next: forward factual both to confirm, then FINAL.\n")
print('ok')
```