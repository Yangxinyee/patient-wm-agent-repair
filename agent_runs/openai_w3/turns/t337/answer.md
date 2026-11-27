Earlier candidates also passed factual checks without resolving the rollout asymmetry. I’ll test an intermediate contraction of 0.01, bracketed by the two tested scales, while recovering the dose-channel implementation. A doubled-insulin response near −13 with acceptable factual loss would support this as a fallback; the other three responses still need separate improvement.

```python
import torch, inspect, re, json
from pathlib import Path

assert "checkpoint273" in globals()
checkpoint292 = {n: load(n).clone() for n in edited()}
base292 = checkpoint273
scaled292 = {
    "core.prelude.0.attn.out.weight",
    "core.prelude.0.ffn.2.weight",
    "core.prelude.0.ffn.2.bias",
}
for n in list(edited()):
    if n not in base292:
        drop_edit(n)
for n, w in base292.items():
    w = torch.as_tensor(w, dtype=torch.float32)
    if n in scaled292:
        w0 = load_original(n)
        w = w0 + 0.01 * (w - w0)
    save_edit(n, w)
checkpoint_alpha01 = {n: load(n).clone() for n in edited()}
print("Installed alpha=0.01; tensors:", edited())

print("\n=== FACTUAL PASS 29 ===")
print(Path("/work/fwd_29/summary.json").read_text()[:2000])

print("\n=== CARRIER IMPLEMENTATION NOTES ===")
for line in Path("/work/notes.md").read_text().splitlines():
    if re.search(r"Python run(?:s)?(?: |: )(154|155|187|193|206)\b", line):
        print(line[:650])

print("\n=== RELEVANT FUNCTION SIGNATURES / DEPENDENCIES ===")
for name in ["repair_action_vectors", "posterior153", "posterior261",
             "hidden192", "hidden220", "readout134", "means105",
             "parts254", "state_update82"]:
    f = globals().get(name)
    if callable(f):
        print(name, inspect.signature(f), "names:", f.__code__.co_names)
        try:
            print(inspect.getsource(f)[:1000])
        except (OSError, TypeError):
            pass

print("\n=== DOSE-CHANNEL / READOUT VARIABLES ===")
items292 = []
for k, v in sorted(list(globals().items())):
    if re.search(r"(carrier|dose|z_|posterior|hidden|readout|direction).*"
                 r"(153|154|187|192|193|206|220|228|249|261)$", k):
        if isinstance(v, dict):
            detail = f"dict keys {list(v)[:10]}"
        elif hasattr(v, "shape"):
            detail = str(tuple(v.shape))
        else:
            detail = repr(v)[:120]
        items292.append(f"{k}: {detail}")
print("\n".join(items292)[:2200])
print("\n=== HEAD SHAPES ===")
print({n: SHAPES[n] for n in NAMES
       if "attr_heads.value" in n or "z_post.2" in n or "z_in" in n})

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 292; forwards 30. Preserved alpha=.005 in checkpoint292 and installed alpha=.01 relative to checkpoint273 for attention-out and FFN-output deltas; other checkpoint273 weights unchanged. This brackets tested alpha=.005/.02 and is a fallback calibration, not a resolution of dose asymmetry. Recovering dose-channel/readout helpers alongside factual pass29. Next factual safeguard and paired rollout.\n")
```
