The value head responds strongly to direct state changes, so it is not constant. That narrows the problem to the transport or recurrent calibration. I’ll test unrestricted injection directions through those stages; measurable, repeatable responses would support replacing the old ill-conditioned calibration.

```python
import numpy as np, torch, dis
torch.set_num_threads(8)

print("Prelude return operations:",
      [(i.opname, i.argrepr) for i in list(dis.get_instructions(prelude155))[-12:]])

def preludeE185(tokens, injection=None):
    result = prelude155(tokens, overrides=zero169, injection=injection)
    return result[0] if isinstance(result, (tuple, list)) else result

res185 = prelude155(tokens_orig169, overrides=zero169)
print("Prelude result shapes:",
      [tuple(x.shape) for x in res185] if isinstance(res185, (tuple, list))
      else tuple(res185.shape))

E185 = preludeE185(tokens_orig169)
print("E shape/norm", tuple(E185.shape), float(E185.norm()))
print("Input norms", float(hf169.norm()), float(hh169.norm()),
      "dt", arr183(dt169))

h185, out185 = state_update82(hf169, E185, mem169, mask169, age_emb169, 0)
print("Recurrent h/out norms", float(h185.norm()), float(out185.norm()),
      "h versus retained hh RMS",
      float(torch.sqrt(torch.mean((h185 - hh169) ** 2))))

view185 = view171(h185, out185, dt169)
base_phys185 = phys171(view185)
print("Baseline view norm", float(view185.norm()),
      "sensor means", arr183(base_phys185.mean(0)))

rng185 = np.random.default_rng(185)
probe_dirs185 = rng185.normal(size=(192, 8))
probe_dirs185 /= np.linalg.norm(probe_dirs185, axis=0, keepdims=True)

for epsilon in [0.5, 2.0]:
    responses185 = []
    for j in range(8):
        inj = tensor153(epsilon * probe_dirs185[:, j])[None, None, :]
        Ep = preludeE185(tokens_orig169, inj)
        hp, op = state_update82(hf169, Ep, mem169, mask169, age_emb169, 0)
        vp = view171(hp, op, dt169)
        responses185.append(arr183((phys171(vp) - base_phys185).mean(0)) / epsilon)
        if j == 0:
            print("epsilon", epsilon, "E/h/view paired RMS",
                  float(torch.sqrt(torch.mean((Ep-E185)**2))),
                  float(torch.sqrt(torch.mean((hp-h185)**2))),
                  float(torch.sqrt(torch.mean((vp-view185)**2))))
    print("epsilon", epsilon, "readout derivatives (directions x sensors)",
          np.round(np.asarray(responses185), 5))

print("Cache/original differences",
      [(n, float(np.max(np.abs(arr183(W[n]) - load_original(n)))))
       for n in zero169 if n in W])

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 185; forwards 10. Direct value-head probe is nonconstant (RMS shifts 5.89/28.53/38.58 for random view scales .1/1/5). Testing unrestricted end-to-end prelude injection, rather than trusting quantized J171. No edits installed.\n")
```
