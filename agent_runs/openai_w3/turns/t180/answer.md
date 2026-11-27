The helper supports adding age features to keys only, which matches the architecture, and its weight cache is original. I’ll now test the learned age embedding with both time units and memory timings. Near-zero state error would validate replay; persistent error would show that age handling is not the whole problem.

```python
import numpy as np
import torch

p162 = np.asarray(positions155, dtype=int)
i162 = np.asarray(indices155, dtype=int)
targeth162 = torch.as_tensor(observed_h156[i162], dtype=torch.float32)
targeto162 = torch.as_tensor(observed_out156[i162], dtype=torch.float32)
hf162 = torch.as_tensor(hflow_all[p162], dtype=torch.float32)
e162 = torch.as_tensor(E_original155[p162], dtype=torch.float32)

print("Input h RMS:", float(hf162.square().mean().sqrt()),
      "target h RMS:", float(targeth162.square().mean().sqrt()))
print("Context agreement RMS:",
      float(torch.mean((cf161 - torch.as_tensor(C154[p162]))**2).sqrt()))

results162 = []
for include162 in (False, True):
    mem162, mask162, age162 = memory82(i162, include162)
    for unit162 in (1.0, 3600.0):
        for logscale162 in (1.0, 0.1):
            embage162 = mlp(
                torch.log1p(age162 * unit162) * logscale162, "core.age")
            for mode162 in (0, 1, 2):
                hh162, oo162 = state_update82(
                    hf162, e162, mem162, mask162, embage162, mode162)
                eh162 = float((hh162-targeth162).square().mean().sqrt())
                eo162 = float((oo162-targeto162).square().mean().sqrt())
                results162.append((eh162, eo162, include162,
                                   unit162, logscale162, mode162))
                if len(results162) == 1 or eh162 < min(
                        r[0] for r in results162[:-1]):
                    replay_best162 = {
                        "h": hh162.clone(), "out": oo162.clone(),
                        "memory": mem162.clone(), "mask": mask162.clone(),
                        "age": embage162.clone(), "mode": mode162,
                        "include_current": include162,
                        "age_unit": unit162, "logscale": logscale162
                    }

print("Best replay configurations:")
for r in sorted(results162)[:10]:
    print(r)
bh162 = replay_best162["h"].numpy()
th162 = targeth162.numpy()
coef162 = np.sum(bh162*th162)/np.sum(bh162*bh162)
print("Best h scale", float(coef162),
      "scaled RMS", float(np.sqrt(np.mean((coef162*bh162-th162)**2))))
with open("/work/notes.md", "a") as f:
    f.write("\nPython run 162; forwards 10. Recurrent replay explicitly "
            "tested learned core.age embeddings, key-only versus source "
            "insertion, seconds/hours, log scaling, and memory inclusion. "
            "Best configuration: " + repr(sorted(results162)[0]) +
            ". No weights changed.\n")
```
