No weights have been changed, and the saved routines are sufficient for a paired local test. I’ll now apply the proposed directions at finite amplitudes using identical random draws. Responses that remain close to the linear targets would justify a candidate edit; strong saturation would rule out these directions.

```python
import time
t95 = time.time()
U_backup95 = U4.clone()
options95 = dict(step_options82)
print("STEP OPTIONS", options95)
print("DIRECTION NORMS", np.linalg.norm(V90, axis=0))
results95 = []
try:
    for j95, event95, drug95 in chosen90:
        means95 = []
        amplitudes95 = [-2.0, -1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
        for a95 in amplitudes95:
            U4[j95] = U_backup95[j95] + torch.as_tensor(
                a95 * V90[:, drug95], dtype=torch.float32)
            z95 = torch.from_numpy(s4["x"][j95:j95+1, 960:992].copy())
            h95 = local_step67(j95, z95, **options95)
            m95, events95 = next_means_crn74(j95, h95, nsamples=32, seed=913)
            idx95 = np.flatnonzero(np.asarray(events95) == event95)[0]
            means95.append(float(m95[0, idx95]))
        base95 = means95[amplitudes95.index(0.0)]
        shifts95 = np.asarray(means95) - base95
        results95.append((j95, event95, drug95, shifts95))
        print("PROBE", j95, event95, "base", round(base95, 3),
              "amplitudes", amplitudes95, "shifts", np.round(shifts95, 4).tolist())
finally:
    U4 = U_backup95
with open("/work/notes.md", "a") as f95:
    f95.write("\n## Run 95: finite action-direction screening\n")
    f95.write("No edits. Tested the run-91 action injection directions with common random numbers.\n")
    for r95 in results95:
        f95.write(str((r95[0], r95[1], r95[2], np.round(r95[3], 4).tolist())) + "\n")
print("SECONDS", round(time.time()-t95, 2), "EDITS", edited())
```
