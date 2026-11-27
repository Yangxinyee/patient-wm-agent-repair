Output normalization matches the official states exactly, and the recurrent mean has a typical norm around 450. This explains why modest latent updates can be heavily diluted. I’ll capture the input to `core.b_in`, where a targeted amplification could act directly; the next check is whether it contains a selective, recoverable dose contrast.

```python
import numpy as np
import torch
import dis

print("=== CONTEXT AND ACTION INTERFACES ===")
print("context151:",
      {k: tuple(v.shape) if hasattr(v, "shape") else type(v).__name__
       for k, v in context151.items()})
for name in ["rr35_scaled_action_encoding", "r59_core", "r179_capture_head"]:
    fn = globals()[name]
    print(name, "args:",
          fn.__code__.co_varnames[:fn.__code__.co_argcount])
    print("dependencies:", fn.__code__.co_names)
    print("constants:",
          [x for x in fn.__code__.co_consts
           if isinstance(x, (str, int, float, tuple))][:20])

resume224_capture = {}
resume224_old_lin = lin0

def resume224_capture_lin(x, prefix):
    if prefix == "core.b_in":
        resume224_capture["E"] = x.detach().clone()
    return resume224_old_lin(x, prefix)

resume224_token = posterior_token154(candidate193, context151["u"])
globals()["lin0"] = resume224_capture_lin
try:
    resume224_result = local_token154(resume224_token, context151["u"])
finally:
    globals()["lin0"] = resume224_old_lin

print("Local update return:",
      type(resume224_result).__name__,
      [tuple(x.shape) for x in resume224_result]
      if isinstance(resume224_result, (list, tuple)) else "")
print("Captured inputs:",
      {k: tuple(v.shape) for k, v in resume224_capture.items()})
if "E" in resume224_capture:
    resume224_Ef = resume224_capture["E"].numpy()
    print("E norm median/p90:",
          np.quantile(np.linalg.norm(resume224_Ef, axis=-1), [.5, .9]))

print("=== PHYSICAL-RATE ALIGNMENT ===")
print("r132_ix length:", len(r132_ix))
print("physical rates on those indices:",
      np.quantile(r133_physical[r132_ix], [0,.5,.9,1], axis=0))
print("q158 values/counts:", np.unique(q158, return_counts=True))
print("Existing readout helpers:",
      [k for k, v in list(globals().items())
       if not k.startswith("resume")
       and any(t in k.lower() for t in ["readout", "surrogate", "paired"])
       and (callable(v) or isinstance(v, dict))])

print("=== ACTION ENCODER CALL FORMAT ===")
for ins in list(dis.get_instructions(rr35_scaled_action_encoding))[:100]:
    print(ins.opname, ins.argrepr)

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 224: capture recurrent-input features\n"
        "- Official out normalization reconstructed with RMS error 7.7e-8.\n"
        "- Mean-slot norms are large (~453 median), supporting dilution of token-level changes.\n"
        "- Capturing core.b_in input E under the smaller candidate193 for a dose-selective low-rank correction.\n"
        "- No saved weights changed. Python runs: 224; forward passes: 17.\n"
    )
```