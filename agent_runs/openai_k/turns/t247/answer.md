The channels are active: their median injections are about 23–24 in norm, while output-state changes are often tiny. That supports a downstream routing bottleneck rather than missing dose input. I’ll inspect the previous head experiments and available routing interfaces before trying amplification; success must appear in official dose responses, not just larger hidden-state changes.

```python
import types
import re
import numpy as np

print("=== SELECTED PRIOR EXPERIMENTS ===")
for section in resume220_notes.split("\n## "):
    if re.match(r"Run (89|131|135|147|154|156|171|180|193)\b", section):
        print(section[:650])

print("=== POSTERIOR CONSTRUCTION ===")
for label, cand in [("physical base", r132_base), ("zero baseline", baseline138)]:
    print(label, list(cand))
    for name in ["z_post.0.weight", "z_post.2.weight"]:
        if name in cand:
            delta = resume221_np(cand[name]) - resume221_np(load_original(name))
            norms = np.linalg.norm(delta, axis=1)
            changed = np.flatnonzero(norms > 1e-5)
            print(name, "changed rows:", changed.tolist(),
                  "largest row-change norms:", np.round(np.sort(norms)[-6:], 4))
print("Candidate193 injection-column norms:",
      np.linalg.norm(resume221_np(candidate193["z_in.weight"])[:, 30:32], axis=0))

print("=== AVAILABLE MODEL HELPERS ===")
resume222_functions = [
    (k, v) for k, v in list(globals().items())
    if isinstance(v, types.FunctionType)
    and v.__module__ == "__main__"
    and any(t in k.lower() for t in
            ["head", "view", "core", "action", "atom", "bundle", "token", "routing"])
]
print([k for k, v in resume222_functions])
for name in ["repaired_core_local", "build_candidate89", "solve_routing158"]:
    fn = globals().get(name)
    if callable(fn) and hasattr(fn, "__code__"):
        print(name, "args:", fn.__code__.co_varnames[:fn.__code__.co_argcount],
              "dependencies:", fn.__code__.co_names)

print("=== TARGETED ROUTING TENSORS ===")
for name in NAMES:
    if any(s in name for s in
           ["core.prelude.0.self_attn", "core.b_in", "core.c_out", "attr_heads.value_bin"]):
        a = resume221_np(load(name))
        print(name, SHAPES[name], "norm=", round(float(np.linalg.norm(a)), 4))

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 222: active channels, weak downstream routing\n"
        "- Official candidate158 channels are nonzero; median injection norms ~23-24.\n"
        "- Output-state changes are much smaller, often nearly zero, despite sizable slot-state changes.\n"
        "- This rejects an inactive-channel explanation and motivates a different downstream edit boundary.\n"
        "- Recovering previous head experiments and targeted routing matrices; no weights changed.\n"
        "- Python runs: 222; forward passes: 17.\n"
    )
```