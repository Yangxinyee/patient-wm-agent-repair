The encoder uses an analytic rate channel, but its local helper references a shared weight dictionary. I need to verify that it is testing the current edited encoder, not a stale one. I’ll also extract the routing equations compactly. Only a current-weight test can establish whether transport is proportional.

```python
from collections import defaultdict

def compact161(f, maxchars=3000):
    groups = defaultdict(list)
    line = None
    for ins in dis.get_instructions(f):
        if ins.starts_line is not None:
            line = ins.starts_line
        if ins.opname not in ("RESUME", "NOP", "COPY", "SWAP", "EXTENDED_ARG"):
            groups[line].append(ins.opname.replace("LOAD_", "L:").replace("STORE_", "S:")
                                + (" " + ins.argrepr if ins.argrepr else ""))
    text = "\n".join(f"{k}: " + "; ".join(v) for k, v in groups.items())
    print(text[:maxchars])

print("=== CURRENT-WEIGHT REFERENCE CHECK ===")
for n in ("W0", "W", "Wtransport133", "_r147_clean_weights", "candidateClean135"):
    if n in globals() and isinstance(globals()[n], dict):
        d = globals()[n]
        print(n, "keys", len(d))
        for key in ("encode_action.mlp1.0.weight", "encode_action.mlp2.2.weight",
                    "core.prelude.0.attn.kv.weight"):
            if key in d:
                print(key, "maxdiff_from_current",
                      float(np.max(np.abs(as_np(d[key]) - as_np(load(key))))))
print("=== ENCODER EQUATIONS ===")
compact161(enc_set117, 2700)
print("=== SAVED-STATE STEP EQUATIONS ===")
compact161(wm_step_saved, 3400)
print("=== PRELUDE PROBE EQUATIONS ===")
compact161(prelude_all83, 2000)

print("=== ISOLATED DOSE-VECTOR COORDINATES ===")
for event, v in zip(_r148_events, dose_vectors154):
    a = as_np(v).reshape(-1)
    ix = np.flatnonzero(np.abs(a) > 1e-5)
    print(event, list(zip(ix.tolist(), np.round(a[ix], 5).tolist())))
print("=== LAST DOSE SUMMARY ===")
print(Path("/work/fwd_18/summary.json").read_text()[:1400])

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 156–161: restart / finite-transport audit\n"
            "- Fixed-history ideal-dose decoder response is approximately linear and symmetric.\n"
            "- GPU hourly asymmetry is already visible at hour 13; finite action transport must be checked before attributing it solely to feedback.\n"
            "- No additional weight changes. Auditing helper weight references and exact current-encoder routing.\n")
```
