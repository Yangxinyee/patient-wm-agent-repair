Head 1 appears much easier to replace than the other attention heads, making it a plausible location for the new signal. The weight changes are substantial, though, so compensation accuracy alone is not enough. I’ll run the candidate through the reconstructed transition and reading head; small reading-level errors would support a GPU factual test.

```python
import time
t108 = time.time()
rW_original108 = rW
rW_candidate108 = dict(rW_original108)
rW_candidate108.update({
    n: torch.as_tensor(t, dtype=torch.float32).clone()
    for n, t in candidate104.items()
})
hs108 = slice(head104 * width103, (head104 + 1) * width103)
print("attention head width:", width103)
print("candidate replaced-head output-column norms:",
      torch.linalg.vector_norm(
          rW_candidate108["core.prelude.0.attn.out.weight"][:, hs108],
          dim=0).numpy()[:12])
print("original replaced-head output-column norms:",
      torch.linalg.vector_norm(
          rW_original108["core.prelude.0.attn.out.weight"][:, hs108],
          dim=0).numpy()[:12])

try:
    rW = rW_original108
    baseline108 = transition73(probe_indices95)
    means_original108 = np.asarray(probe_means95(baseline108))
    rW = rW_candidate108
    states_candidate108 = transition73(probe_indices95)
    means_candidate108 = np.asarray(probe_means95(states_candidate108))
finally:
    rW = rW_original108

print("transition shapes:", baseline108.shape, states_candidate108.shape)
delta108 = np.asarray(states_candidate108) - np.asarray(baseline108)
print("candidate state change RMS: out",
      float(np.sqrt(np.mean(delta108[:, :192]**2))),
      "h", float(np.sqrt(np.mean(delta108[:, 192:960]**2))))
for event108 in ("glucose_finger", "map_arterial"):
    mask108 = probe_data95.event.to_numpy() == event108
    dm108 = means_candidate108[mask108] - means_original108[mask108]
    print(event108, "reading mean change", float(dm108.mean()),
          "RMS", float(np.sqrt(np.mean(dm108**2))),
          "max", float(np.max(np.abs(dm108))))

print("provisional impulse kernels:",
      {k: np.asarray(v).ravel()[:15].tolist()
       for k, v in provisional_kernels.items()})
print("elapsed seconds:", round(time.time()-t108, 2))
with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 105–108\n"
            "Recovered candidate architecture and interfaces after restart. "
            "Head 1 is the most compensable attention head (held-out output RMS about 0.095); "
            "candidate remains unsaved pending reading-level and GPU factual tests. "
            "Immediate record estimates remain insulin -2.049/rate and NE +3.171/rate; "
            "sustained rollout targets are not inferred from these alone.\n")
print("saved edits:", edited())
```
