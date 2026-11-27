Changing head count or age scaling did not resolve the mismatch, and the vocabulary confirms four attention heads. I’ll stop that search and inspect the implemented update and its inputs directly. A consistent scale or operation error could explain the discrepancy; otherwise the reconstruction assumptions need revision.

```python
import dis
import numpy as np
import torch

print("=== EXACT HELPER OPERATIONS ===")
for rr32_name in ("audit_block", "audit_core_update"):
    print("\n" + rr32_name)
    rr32_ops = [
        i.opname + (":" + i.argrepr if i.argrepr else "")
        for i in dis.Bytecode(globals()[rr32_name])
        if i.opname not in ("RESUME", "CACHE", "PRECALL", "PUSH_NULL")
    ]
    for rr32_j in range(0, len(rr32_ops), 9):
        print(" | ".join(rr32_ops[rr32_j:rr32_j+9]))

print("\n=== INPUT AND STATE SCALES ===")
for rr32_name in (
    "core_probe_previous", "core_probe_h", "core_probe_e",
    "core_probe_u", "core_probe_z", "core_probe_memory",
    "core_probe_dt", "core_probe_ages"
):
    rr32_a = globals()[rr32_name].detach().cpu().numpy()
    print(rr32_name, "shape", rr32_a.shape,
          "quantiles", np.quantile(rr32_a, [0, .1, .5, .9, 1]).round(5))

rr32_target = torch.as_tensor(sx[np.asarray(core_probe_indices, dtype=int)])
print("saved out RMS:", float(rr32_target[:, :192].square().mean().sqrt()))
print("saved h RMS:", float(rr32_target[:, 192:960].square().mean().sqrt()))

print("\n=== RECURRENT PARAMETERS ===")
for rr32_name in ("core.delta_in", "core.a_in", "core.flow.a"):
    print(rr32_name,
          np.quantile(W0[rr32_name].numpy(), [0, .1, .5, .9, 1]).round(6))
print("first probe dt:", core_probe_dt[:6].tolist())
print("first probe ages:", core_probe_ages[0, :8, 0].tolist())
print("first probe memory mask:", core_probe_mask[0, :28].tolist())

print("\n=== ENCODING/PREVIOUS-STATE OBJECT NAMES ===")
for rr32_name, rr32_obj in list(globals().items()):
    if (not rr32_name.startswith("rr") and
        any(s in rr32_name for s in ("encod", "context", "previous", "prev_", "gap_"))):
        if isinstance(rr32_obj, (np.ndarray, torch.Tensor)):
            print(rr32_name, tuple(rr32_obj.shape))
        elif isinstance(rr32_obj, dict):
            print(rr32_name, "dict keys", list(rr32_obj)[:8])

with open("/work/notes.md", "a") as rr32_f:
    rr32_f.write("\n## Run 32: exact update audit\n"
                 "- Head-count/age search failed to validate reconstruction; documented head count is four.\n"
                 "- Inspecting helper operations and input scales rather than selecting an inaccurate best fit.\n"
                 "- Python runs: 32; forward passes: 4; no weights edited.\n")
```