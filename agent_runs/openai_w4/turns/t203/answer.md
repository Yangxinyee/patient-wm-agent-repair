The insulin gate is a difference of two GELU units. Earlier checks found weaker norepinephrine decoding, so copying the insulin coefficients would be unsafe. I’ll recover the norepinephrine fit and test its dose-change sensitivity directly. Reliable half/double sensitivity matters more here than a high factual fit score alone.

```python
import inspect, numpy as np
from pathlib import Path

print("RETAINED RAW/GATE FITS")
names178 = sorted(k for k in list(globals())
                  if any(str(n) in k for n in range(82, 95))
                  and not k.startswith("_"))
for k178 in names178:
    v178 = globals()[k178]
    if isinstance(v178, dict):
        print(k178, "dict", list(v178)[:14])
    elif hasattr(v178, "shape"):
        print(k178, tuple(v178.shape))
    elif callable(v178):
        print(k178, inspect.signature(v178), v178.__code__.co_names[:12])
    elif isinstance(v178, (int, float, str, list, tuple)):
        print(k178, str(v178)[:180])

print("\nKEY FIT PARAMETERS")
for k178 in ["gate_coef94", "gate_bias94", "reserved94", "innovation_coef94",
             "innovation_bias94", "W0base94", "b0base94",
             "raw_models92", "raw_fits92", "raw_models93",
             "decoder_models_repair", "dose_models_repair", "gate_models_repair"]:
    if k178 in globals():
        v178 = globals()[k178]
        if isinstance(v178, dict):
            for kk178, vv178 in v178.items():
                print(k178, kk178,
                      str(vv178)[:600] if not hasattr(vv178, "shape")
                      else (vv178.shape, arr177(vv178).reshape(-1)[:5].tolist()))
        elif hasattr(v178, "shape"):
            print(k178, v178.shape, arr177(v178).reshape(-1)[:8].tolist())
        else:
            print(k178, v178)

print("\nCAUSAL TARGETS")
for drug178 in ["insulin", "norepinephrine"]:
    print(drug178, "kernel", np.round(kernels137[drug178], 4).tolist(),
          "matched readout", readout_estimates137[drug178])

print("\nSAVED/CPU CONSISTENCY")
for k178 in ["z_post.0.weight", "z_post.2.weight", "z_in.weight",
             "core.prelude.0.attn.kv.weight", "core.b_in.weight"]:
    print(k178, float(np.max(np.abs(arr177(rW[k178]) - arr177(load(k178))))))

with Path("/work/notes.md").open("a") as f178:
    f178.write("\n\n## Run 178\n"
               "Recovered NE raw-decoder/gate fits and matched kernel targets. Earlier NE "
               "decoding was weaker than insulin, so an independent dose-sensitivity test is "
               "required before using it in a shared-head channel. No saved weights changed.\n")
```