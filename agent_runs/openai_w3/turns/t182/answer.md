The last diagnostic stopped because the posterior helper expects tensor weights, so it tells us nothing about candidate quality. I’ll fix that conversion and inspect the carrier decoder and readout layout. Small factual latent changes and dose-linear carrier contrasts would support a pilot; large changes or nonlinear contrasts would require revising the candidate.

```python
import numpy as np
import torch
import inspect, dis

origpost164 = {n: torch.as_tensor(load_original(n), dtype=torch.float32)
               for n in post154}
post164 = {n: torch.as_tensor(v, dtype=torch.float32)
           for n, v in post154.items()}
ctx164 = torch.as_tensor(C154, dtype=torch.float32)
ev164 = torch.as_tensor(E154, dtype=torch.float32)
uv164 = torch.as_tensor(U154, dtype=torch.float32)
zo164 = posterior153(ctx164, ev164, uv164, origpost164)
zc164 = posterior153(ctx164, ev164, uv164, post164)
dz164 = (zc164-zo164).numpy()
print("Posterior shape", tuple(zo164.shape))
print("Largest coordinate changes (SD, mean, coordinate):",
      sorted([(float(np.std(dz164[:, j])),
               float(np.mean(dz164[:, j])), j)
              for j in range(dz164.shape[1])], reverse=True)[:6])

print("\n=== CARRIER DEFINITION ===")
for n in ("reserved154", "rows154", "row154", "value_extract154",
          "presence154", "rate154", "sigma_ref154"):
    if n in globals():
        v = globals()[n]
        if isinstance(v, (np.ndarray, torch.Tensor)):
            a = v.numpy() if isinstance(v, torch.Tensor) else v
            print(n, a.shape, "RMS", float(np.sqrt(np.mean(a*a))),
                  "first", a.reshape(-1)[:6].tolist())
        else:
            print(n, repr(v)[:500])

print("\n=== READOUT LAYOUT ===")
for n in ("codes97", "codes105", "transition97", "centres105",
          "attr_static105", "readrows121"):
    if n in globals():
        v = globals()[n]
        print(n, getattr(v, "shape", None))
        if isinstance(v, (np.ndarray, torch.Tensor)) and len(v.shape) == 1:
            print("first:", v[:20].tolist())
print(readrows121.head(8).to_string(index=False))

print("\n=== ACTION HELPERS AND SCENARIOS ===")
for n in ("action_vectors87", "repair_action_vectors", "atom80"):
    fn164 = globals()[n]
    print(n, inspect.signature(fn164), fn164.__code__.co_varnames,
          fn164.__code__.co_names)
for n, v in list(globals().items()):
    if isinstance(v, (np.ndarray, torch.Tensor)) and len(v.shape) >= 2 \
            and v.shape[0] == 11:
        print(n, tuple(v.shape))

print("\n=== PRELUDE INJECTION INTERFACE ===")
instructions164 = list(dis.get_instructions(prelude155))
for j, ins in enumerate(instructions164):
    if ins.argval == "injection":
        for it in instructions164[max(0, j-2):min(len(instructions164), j+15)]:
            if it.opname not in ("CACHE", "PRECALL"):
                print(it.opname, it.argrepr)

print("\n=== NUMERIC VOCAB SAMPLE ===")
print(type(VOCAB["numeric"]).__name__)
print(repr(VOCAB["numeric"])[:1400])
print("\nReference columns:", list(reference.columns))
print("Reference means:", reference.select_dtypes("number").mean().to_dict())
```
