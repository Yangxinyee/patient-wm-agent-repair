The retained helpers include exact local state updates, so they may let us test a correction without relying on the noisy decoder. Another output overflow hid the effect estimates. I’ll now extract only compact summaries; the key check is whether the records identify both immediate effects and their persistence.

```python
import numpy as np
from pathlib import Path

print("=== RELEVANT SHAPES ===")
for k in NAMES:
    if any(w in k for w in [
        "z_post", "z_in", "attr_heads.value_bin", "query_norm",
        "core.flow", "core.delta_in", "core.a_in", "core.b_in",
        "core.c_out", "core.out_norm", "core.prelude_norm"
    ]):
        print(k, SHAPES[k])

print("\n=== EFFECT OBJECT SUMMARIES ===")
for k in ["dynamic_ir", "dynamic_moments", "cluster_dynamic",
          "effects63", "factual_missing_effect0",
          "innov_targets_audit", "targets158", "currentweights151",
          "context151", "captures151", "indices151", "decoders164"]:
    if k not in globals():
        continue
    v = globals()[k]
    if isinstance(v, dict):
        print(k, "keys:", list(v.keys())[:24])
        if k in ["dynamic_ir", "dynamic_moments", "cluster_dynamic", "effects63"]:
            for kk, vv in list(v.items())[:8]:
                if hasattr(vv, "shape"):
                    aa = np.asarray(vv)
                    print(" ", kk, aa.shape, aa.reshape(-1)[:10])
                else:
                    print(" ", kk, str(vv)[:260])
    elif hasattr(v, "shape"):
        print(k, tuple(v.shape), np.asarray(v).reshape(-1)[:12])
    else:
        print(k, type(v).__name__, str(v)[:400])

print("\n=== COMPACT HELPER INTERFACES ===")
for k in ["posterior_token154", "local_token154", "dynamic_views154",
          "dynamic_delta154", "token164", "r132_channels"]:
    if k not in globals():
        continue
    c = globals()[k].__code__
    print(k, "args:", c.co_varnames[:c.co_argcount],
          "names:", c.co_names,
          "constants:", [v for v in c.co_consts if isinstance(v, (str, int, float, tuple))][:22])

print("\n=== IDENTIFICATION EVIDENCE (CAPPED) ===")
lines = Path("/work/notes.md").read_text().splitlines()
selected, heading = [], ""
for line in lines:
    if line.startswith("##"):
        heading = line
    if any(w in line.lower() for w in [
        "causal", "randomiz", "persistence", "rho", "per unit",
        "per rate", "long-run", "long run", "confound",
        "negative control", "target shift", "identified effect"
    ]):
        selected.append(heading + " | " + line[:450])
print("\n".join(selected)[:3500])

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 169: capped identification/interface extraction\n"
        "- Recent exact-update helpers are retained. Avoiding tensor-valued "
        "function defaults, which caused run 168 output overflow.\n"
        "- Extracting effects, persistence, and exact local interfaces before "
        "any new edit. Saved edits unchanged.\n"
        "- Python runs: 169; forward passes: 10.\n"
    )
```