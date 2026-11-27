The last run showed my current state. The two operations scale the dose and rate columns of the numeric embedding by 5, and the last forward pass showed this barely moves the model's dose answers. The architecture explains why. Dose information passes through the action set encoder, which ends in a normalisation, and then through more normalised layers. Scaling an embedding input is therefore largely squashed.

My main concern is double counting. The model's glucose and MAP dynamics already absorb the average, policy-driven drug effect. An edit that makes predictions respond to the raw dose would bias the factual predictions and break the 0.005-nat limit. What I need is a response to the part of the dose that is not predictable from the record, the dose innovation.

The z posterior is a natural carrier for that kind of signal, because it encodes what is surprising in each bundle, and it only affects the state update. Next I want the full tensor list, the two saved vectors, and the files I already have. Then I can decide which tensor to edit, for example the z_post output layer, z_in, or the encode_action output.

```python
import numpy as np, os, json
from collections import OrderedDict
grp = OrderedDict()
for nme in NAMES:
    top = '.'.join(nme.split('.')[:2])
    grp.setdefault(top, []).append(nme)
for k, lst in grp.items():
    print(k, ' | '.join(f"{x.split('.',2)[-1] if x.count('.')>=2 else x}:{tuple(SHAPES[x])}" for x in lst))
print("v001", np.load('/work/vectors/v001.npy').round(3), "v002", np.load('/work/vectors/v002.npy').round(3))
for k in range(1,5):
    print(k, os.listdir(f'/work/fwd_{k}'))
print(json.dumps(VOCAB['config'])[:800])
print([c for c in VOCAB.keys()])
```