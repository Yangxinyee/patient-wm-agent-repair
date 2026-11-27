I'm resuming after a restart. Before going further I want to re-read the architecture spec and my replica code, and check what my current `z_post` edit actually does. The notebook doesn't record that edit, and I don't want to build on something I can't account for.

The last run compared two hypotheses for the encoders. Under "end_to_e" the action encoding u matched the tool's states almost exactly (99.7% of entries within 1e-4). The reading encoding e still mismatched by about 1 on every bundle, so the reading encoder in my replica is still wrong.

Next step: print ARCHITECTURE.md, list the forward-output folders, and diff the edited `z_post` tensors against the originals. If the edit is a large change to the u-columns, it was an earlier amplification attempt that I need to evaluate or undo.

```python
import os, json, numpy as np
arch = open('/model/ARCHITECTURE.md').read()
print(len(arch))
print(arch[:14000])
print("=====FILES=====")
for d in sorted(os.listdir('/work')):
    p = os.path.join('/work', d)
    if os.path.isdir(p):
        print(d, os.listdir(p))
    else:
        print(d, os.path.getsize(p))
print("=====EDITS=====")
for n in edited():
    a = load(n); b = load_original(n)
    print(n, SHAPES[n], 'max|diff|', float(np.abs(np.asarray(a)-np.asarray(b)).max()), 'norm orig', float(np.linalg.norm(np.asarray(b))), 'norm diff', float(np.linalg.norm(np.asarray(a)-np.asarray(b))))
```