The old NE edit activates almost equally on original and repaired states, so anchor drift does not explain its factual cost. Its factual state shifts are substantial on both. I’ll instead test a detector with an explicit per-slot factual penalty. First I’ll construct paired dose contrasts, preserving the GPU-recorded posterior at the factual dose; useful contrasts must remain strong relative to factual leakage.

```python
import numpy as np, torch

start_info192 = {}
for i192, (s192, m192) in enumerate(zip(sid191, minute191)):
    rr192 = entries191[(int(s192), int(m192))]
    start_info192[i192] = {
        d: [r for r, k in rr192 if k == 0 and r.event == d]
        for d in ["insulin", "norepinephrine"]
    }

pure192 = np.array([len(w) == 0 for w in world_atoms191]) & ~first191
print("POSTERIOR CHECKS")
for d192 in ["insulin", "norepinephrine"]:
    mask192 = np.array([bool(start_info192[i][d192])
                       for i in range(len(sid191))]) & pure192
    print(d192, "pure action starts", int(mask192.sum()),
          "posterior RMS",
          float(np.sqrt(np.mean(zerr_base191[mask192]**2))))

print("\nSAVED SINGLE-DIRECTION SCHEMA")
with np.load("/work/ne_candidate82.npz", allow_pickle=False) as a192:
    print([(k, a192[k].shape) for k in a192.files])
    for k192 in a192.files:
        if a192[k192].size <= 4:
            print(k192, a192[k192])

cf_train192 = {}
for drug192 in ["norepinephrine", "insulin"]:
    ids192 = np.array([i for i in range(len(sid191))
                       if start_info192[i][drug192]], dtype=int)
    rates192 = np.array([sum(r.rate for r in start_info192[i][drug192])
                         for i in ids192], dtype=np.float64)
    post_factual192 = posterior94(ctx_base191[ids192], obs191[ids192],
                                  act191[ids192])
    for scale192 in [2., .5]:
        atoms192 = []
        for i192 in ids192:
            aa192 = []
            key192 = (int(sid191[i192]), int(minute191[i192]))
            for row192, kind192 in entries191[key192]:
                if row192.event not in ("insulin", "norepinephrine"):
                    continue
                if kind192 == 0 and row192.event == drug192:
                    row192 = row192._replace(rate=row192.rate*scale192,
                                            dose=row192.dose*scale192)
                aa192.append(recon_record_atom(
                    row192, kind=kind192,
                    end_numeric="absent", end_units="none"))
            atoms192.append(aa192)
        uu192 = recon_sets(atoms192, prefix="encode_action", empty_zero=True)
        dz192 = (posterior94(ctx_base191[ids192], obs191[ids192], uu192)
                 - post_factual192)
        zz192 = z_base191[ids192] + dz192
        ee192 = asnp189(prelude46(h_base191[ids192], obs191[ids192],
                                  uu192, zz192))
        cf_train192[(drug192, scale192)] = {
            "ids": ids192, "rate": rates192,
            "E": ee192,
            "dE": ee192-E_base191[ids192],
            "dose_change": rates192*(scale192-1.)
        }
        print(drug192, scale192, "bundles", len(ids192),
              "dose-change RMS", round(float(np.sqrt(np.mean(
                  (rates192*(scale192-1.))**2))), 5),
              "prelude-contrast RMS", round(float(np.sqrt(np.mean(
                  (ee192-E_base191[ids192])**2))), 6))

with open("/work/notes.md", "a") as f192:
    f192.write("\n## Run 192: reject anchor-drift explanation; build dose contrasts\n"
               "Old NE factual state RMS is 1.318 original vs1.321 insulin baseline: "
               "drift hypothesis refuted. Built training NE/insulin dose contrasts "
               "using recorded factual z plus recomputed posterior differences. "
               "Remaining posterior reconstruction offset is explicitly not treated "
               "as verified absolute inference. No weights changed. "
               "Python runs: 192; forwards: 22.\n")
```