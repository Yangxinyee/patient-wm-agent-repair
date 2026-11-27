The existing norepinephrine decoder was calibrated with a continuous embedding perturbation, not an actual doubled or halved dose; that probe did not change dose bins. This could explain the asymmetry. I’ll test the head’s response to symmetric decoder changes and separate replacement-neuron loss from correction loss. Symmetric head responses would point upstream to dose encoding; asymmetric responses would implicate the value head itself.

```python
import numpy as np
from pathlib import Path
from scipy.special import ndtr

print("FOLD COUNTS", int(np.sum(r177_fit)), int(np.sum(r177_val)),
      "active arterial", int(np.sum(r177_active_art)))
print("VALUE CENTRES SHAPE", np186(C104).shape)
C190 = np186(C104)
print("MAP CENTRES", C190[np.flatnonzero(ev188=="map_arterial")[0]])
W2190 = np186(load("attr_heads.value_bin.2.weight"))
print("NE OUTPUT COLUMN", np.round(W2190[:,48],5))
print("NE OPPOSITE-COLUMN ERROR", float(np.max(np.abs(W2190[:,48]+W2190[:,171]))))

orig190 = {n: load_original(n) for n in head185}
H190 = np186(r145_hidden(orig190))
O2190 = np186(orig190["attr_heads.value_bin.2.weight"])
O2b190 = np186(orig190["attr_heads.value_bin.2.bias"])
Lorig190 = H190 @ O2190.T + O2b190 + np186(c182_transition)
removed190 = H190[:,:,[48,107,109,171]] @ O2190[:,[48,107,109,171]].T
Lremoved190 = Lorig190 - removed190 + (
    np186(load("attr_heads.value_bin.2.bias")) - O2b190)
lossorig190 = c181_fn(Lorig190)
lossremoved190 = c181_fn(Lremoved190)
print("REMOVED-NEURON LOSS", {
    "overall": float(np.mean(lossremoved190-lossorig190)),
    "heldout": float(np.mean((lossremoved190-lossorig190)[r177_val]))
})

def mean190(logits):
    p = np.exp(logits-logits.max(axis=-1,keepdims=True))
    p /= p.sum(axis=-1,keepdims=True)
    return (p.mean(axis=0)*C190).sum(axis=-1)

g190 = r177_X @ ((W188[48]+W188[171])/2) + (b188[48]+b188[171])/2
d190 = r177_X @ ((W188[48]-W188[171])/2) + (b188[48]-b188[171])/2
pair190 = (g190+d190)*ndtr(g190+d190)-(g190-d190)*ndtr(g190-d190)
base_mean190 = mean190(c182_candidate_logits)
print("\nSYMMETRIC NE DECODER RESPONSE")
for amount190 in (.25,.5,1.,2.):
    result190 = {}
    for sign190 in (-1,1):
        dt190 = d190 + sign190*amount190
        pt190 = (g190+dt190)*ndtr(g190+dt190)-(g190-dt190)*ndtr(g190-dt190)
        mt190 = mean190(c182_candidate_logits
                       +(pt190-pair190)[:,:,None]*W2190[:,48])
        result190[sign190] = {
            "all_arterial": float(np.mean((mt190-base_mean190)[ev188=="map_arterial"])),
            "active_arterial": float(np.mean((mt190-base_mean190)[r177_active_art]))
        }
    print(amount190, result190)
print("\nHEAD INPUT METADATA VARIABLES", [
    k for k in list(globals()) if k.startswith(("meta", "r96", "r98", "r99"))
][:75])
with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 190: NE probe limitation and loss decomposition\nRecovered that the NE probe shifts embed.code continuously by +0.1-rate direction, including possible END atoms, and does not implement real dose-bin changes. Testing symmetric decoder perturbations and removed-neuron loss; no edit installed.\n")
```