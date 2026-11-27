The recovered estimates are about −1.94 glucose units per insulin-rate unit and +3.2 arterial-MAP units per norepinephrine-rate unit, with different persistence. Halving the edited output pathway barely affected rollouts; normalization may have hidden that change. I’ll test a much smaller, reversible output residual. Better rollout symmetry with acceptable factual loss would support this explanation; unchanged errors would refute it.

```python
import numpy as np, torch, pandas as pd, hashlib
from pathlib import Path

checkpoint284 = {n: load(n).clone() if isinstance(load(n), torch.Tensor)
                 else np.array(load(n), copy=True) for n in edited()}
output_names284 = [
    "core.prelude.0.attn.out.weight",
    "core.prelude.0.ffn.2.weight",
    "core.prelude.0.ffn.2.bias",
]
scale284 = 0.02
for n in output_names284:
    original = torch.as_tensor(load_original(n), dtype=torch.float32)
    reference = torch.as_tensor(checkpoint273[n], dtype=torch.float32)
    candidate = original + scale284 * (reference - original)
    save_edit(n, candidate)
    print(n, "original norm", float(original.norm()),
          "reference delta norm", float((reference-original).norm()),
          "installed delta norm", float((candidate-original).norm()))

for kind in ["factual", "dose_edit"]:
    p = Path(f"/work/fwd_{21 if kind == 'factual' else 22}/{kind}_original.csv")
    if p.exists():
        tab284 = pd.read_csv(p)
        print(kind, "columns", tab284.columns.tolist())
        idcol284 = next((c for c in ["stay_id", "stay"] if c in tab284), None)
        if idcol284:
            ids284 = sorted(tab284[idcol284].unique().astype(int).tolist())
            print(kind, "STAYS", ids284)
        print(tab284.head(2).to_string(index=False))

print("retained candidates267:",
      [(k, type(v).__name__,
        list(v.keys())[:6] if isinstance(v, dict) else
        [getattr(x, "shape", type(x).__name__) for x in v] if isinstance(v, tuple)
        else getattr(v, "shape", None))
       for k, v in globals().get("candidates267", {}).items()])
hash284 = hashlib.sha256(b"".join(
    np.asarray(load(n), dtype=np.float32).tobytes() for n in sorted(edited())
)).hexdigest()[:16]
print("EDIT HASH", hash284)

with open("/work/notes.md", "a") as f:
    f.write(f"\nPython run 284; forwards 24. Installed reversible output-residual scale {scale284}: attention out and FFN output deltas scaled relative to validated checkpoint273; posterior, z_in, q/kv unchanged. checkpoint284 preserves prior validated edit. Test whether strong contraction changes normalized off-record behavior; factual safeguard required. Hash {hash284}.\n")
```
