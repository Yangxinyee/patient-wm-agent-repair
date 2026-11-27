The unedited reconstruction already has worse NLL than the GPU model and a systematic prediction bias. That makes its “factual-preserving” edits unreliable. I’ll audit the head’s input conventions against the architecture, especially the last-reading register. Removing the baseline bias would support a corrected reconstruction; otherwise I’ll avoid fitting further edits in that feature space.

```python
import dis
import numpy as np
import pandas as pd

print("=== RECONSTRUCTION ERROR BY SENSOR ===")
audit92 = head_meta[["event", "hour", "dt", "has_last", "zlast", "scale",
                     "original_mean", "original_nll"]].copy()
audit92["local_mean"] = (np.asarray(head_probs) * repair_C17).sum(1)
audit92["scaled_error"] = (audit92.local_mean-audit92.original_mean)/audit92.scale
audit92["local_nll"] = -np.log(np.clip(
    np.asarray(head_probs)[np.arange(len(head_meta)), repair_fbins], 1e-30, 1))
print(audit92.groupby(["event", "has_last"]).agg(
    n=("scaled_error", "size"), bias=("scaled_error", "mean"),
    rms=("scaled_error", lambda x: np.sqrt(np.mean(x*x))),
    original_nll=("original_nll", "mean"),
    local_nll=("local_nll", "mean")).to_string())

print("\n=== REGISTER INPUTS ===")
for key92 in ("numbers", "dt", "register", "context", "view", "inputs"):
    obj92 = head_recovery[key92]
    a92 = obj92.detach().cpu().numpy() if isinstance(obj92, torch.Tensor) else np.asarray(obj92)
    print(key92, a92.shape, "first rows:", a92[:4, :8] if a92.ndim == 2 else a92[:6])
print(head_meta[["has_last", "age_s", "zlast", "last_bin"]].head(6).to_string(index=False))

print("\n=== RELEVANT TENSOR SHAPES ===")
for name92 in NAMES:
    if name92.startswith(("gap.", "z_prior.", "last_numeric.", "type_emb.",
                           "query_norm.", "attr_heads.value_bin.", "last_status.")):
        print(name92, SHAPES[name92])

print("\n=== HELPER OPERATIONS ===")
for name92 in ("wt", "mlp", "ln", "activation"):
    f92 = globals()[name92]
    print(name92, "arguments", f92.__code__.co_varnames[:f92.__code__.co_argcount],
          "names", f92.__code__.co_names, "constants", f92.__code__.co_consts[:16])

print("\n=== HEAD CONSTRUCTION BYTECODE EXCERPTS ===")
ins92 = list(dis.get_instructions(reconstruct_reading_head))
for target92 in ("numbers", "register_local", "query_input", "inputs", "features"):
    positions92 = [i for i, x in enumerate(ins92)
                   if x.opname == "STORE_FAST" and x.argval == target92]
    for pos92 in positions92:
        print("--", target92, "--")
        for x92 in ins92[max(0, pos92-19):pos92+1]:
            print(x92.opname, x92.argrepr)

print("\nACTIVE EDITS:", edited())
```