The cached inputs contain 32 latent draws for 4,510 training readings, which should support controlled local comparisons. Before further experiments, I’ll restore the weaker combined candidate: it passed one factual cohort, unlike the stronger trial. Testing it on the failed cohort will show whether stronger norepinephrine correction caused the failure.

```python
import hashlib

head196 = (
    "attr_heads.value_bin.0.bias",
    "attr_heads.value_bin.0.weight",
    "attr_heads.value_bin.2.bias",
    "attr_heads.value_bin.2.weight",
)
c196_strong_backup = {
    n: load(n).clone() if isinstance(load(n), torch.Tensor) else load(n).copy()
    for n in head196
}
for n196 in head196:
    save_edit(n196, c183_fallback[n196])

def numpy196(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

hash196 = hashlib.sha256(b"".join(
    numpy196(load(n)).astype(np.float32).tobytes() for n in sorted(head196)
)).hexdigest()[:16]
print("RESTORED CHECKED COMBINED FALLBACK", hash196)
print("CURRENT EDITS", edited())

print("=== AVAILABLE PROBE STATE SOURCES ===")
for n196 in ("r123_statefiles", "r123_statepath", "r123_states",
             "r124_probe_code", "r124_probe_delta_rate", "r124_W"):
    o196 = globals().get(n196)
    if isinstance(o196, dict):
        print(n196, "dict keys", list(o196.keys())[:16])
    elif hasattr(o196, "files"):
        print(n196, "npz keys", o196.files)
    elif hasattr(o196, "shape") and isinstance(o196, (np.ndarray, torch.Tensor)):
        print(n196, tuple(o196.shape))
    else:
        print(n196, repr(o196)[:650])

with open("/work/notes.md", "a") as f196:
    f196.write(
        "\n## Runs 191–196: restart audit and restore\n"
        "Re-read architecture and record schema; recovered cross-fitted drug slopes "
        "(insulin about -2, NE about +3.3 per rate unit) and tentative persistence evidence. "
        "The prior NE probe changes a continuous code-embedding direction rather than "
        "actual dose bins; it cannot establish dose-edit symmetry. Decoder perturbations "
        "are nearly symmetric. Restored c183_fallback (checked weaker combined candidate); "
        "stronger trial preserved in c196_strong_backup. Next: validate fallback factual "
        "fit on tuning 100..199 and construct exact-dose representation probes if feasible. "
        "No new candidate accepted.\n"
    )
```