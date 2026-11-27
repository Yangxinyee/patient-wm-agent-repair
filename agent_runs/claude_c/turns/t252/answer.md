All 48 head0 ops have alpha 1.0, so head0 is scaled x2. The kernel therefore holds numeric x5 + head0 x2, a weaker version I apparently built after pass 26 but never recorded or tested. Before spending passes, I'll check that the 48 vectors are the unit basis vectors for input columns 0-47. I'll also record this state in the notebook. If they are, a factual pass will tell me whether this weaker combo sits safely under budget; my linear-cost estimate is about +0.0038.

```python
import numpy as np
V=[np.load('/work/'+x['vector']) for x in o[2:]]
idx=[int(np.argmax(np.abs(v))) for v in V]; ok=all(abs(np.abs(v).max()-1)<1e-6 and np.sum(np.abs(v)>1e-9)==1 for v in V)
print(ok, idx[:3], idx[-3:], sorted(idx)==list(range(48)))
open('/work/notes.md','a').write("\n## Restart-11 (runs 190-195)\n- Found ops = numeric x5 (ops1-2) + head0 x2 (48 basis 'in' ops alpha 1.0, cols 0-47) = 'N5+H2', untested. Linear cost model: N5 .0016 + head (c-1)*.0022 -> ~.0038 predicted.\n")
```