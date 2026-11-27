The recovered lag estimates imply much larger accumulated effects than the original rollouts: roughly −11 for doubled insulin and +3.8 for doubled norepinephrine on the earlier subset. These remain observational estimates, not known truth. I’ll recover the readout calculation and check relative reconstruction errors before testing response-based latent directions.

```python
import re
import numpy as np
from pathlib import Path

print("=== RETAINED READOUT RECONSTRUCTION FINDINGS ===")
remaining150 = 4200
for sec in sections149:
    title = sec.splitlines()[0] if sec.splitlines() else ""
    m = re.search(r"Python\s+(\d+)", title)
    if m and 18 <= int(m.group(1)) <= 28:
        excerpt = sec[:min(900, remaining150)]
        print("## " + excerpt)
        remaining150 -= len(excerpt)
        if remaining150 < 100:
            break

print("\n=== RELEVANT TENSOR SHAPES ===")
prefixes150 = (
    "z_post.", "z_in.", "z_head.", "z_prior.",
    "attr_heads.value_bin.", "last_numeric.", "last_status.",
    "core.prelude.", "core.prelude_norm.", "core.coda.",
    "core.b_in.", "core.c_out.", "core.age.", "gap.",
    "embed.categorical.", "embed_transition.value_bin", "type_emb"
)
for name in NAMES:
    if name.startswith(prefixes150):
        print(name, SHAPES[name])

print("\n=== EXPOSED STATE SCALES ===")
for label, sl in [("out", slice(0,192)), ("h", slice(192,960)),
                  ("posterior mean", slice(960,992))]:
    a = S[:, sl]
    print(label, "RMS", float(np.sqrt(np.mean(a*a))),
          "standard deviation", float(a.std()),
          "absolute quantiles", np.quantile(np.abs(a), [.5,.9,.99]).round(4))

print("\n=== HELPER DEFAULTS AND CURRENT PROBE ===")
for fn in [core_step136, block136, attention136, atom145, pool137, norm136]:
    print(fn.__name__, "defaults", fn.__defaults__,
          "arguments", fn.__code__.co_varnames[:fn.__code__.co_argcount])
print("probe shapes:", hprobe146.shape, eprobe146.shape, zprobe146.shape,
      dtprobe146.shape, emem146.shape)
print("probe time gap:", dtprobe146)
print("code/event metadata:", {k:code_info[k] for k in
      ["glucose_finger","glucose_lab","map_arterial","map_cuff","insulin","norepinephrine"]})
print("value centres container:", type(VOCAB["value_bin_centres"]).__name__)
print("currently edited:", edited())

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Python 150 — choose response-based latent directions\n"
        "- Recovered controlled insulin lags [-1.99,-2.40,-2.10,-1.71,"
        "-1.22,-.94,-.68,-.53] and NE [3.29,2.62,1.75,.98,.41,.26,.02,.02].\n"
        "- Earlier selected-subset convolution targets: insulin x2 about -11, "
        "NE x2 about +3.8; these are conditional record estimates, not truth.\n"
        "- Inspecting readout reconstruction and relative state errors before "
        "finite-difference latent response calibration. No edits active.\n"
    )
```
