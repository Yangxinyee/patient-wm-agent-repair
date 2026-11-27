The stay-boundary assumption was wrong: forward states are not ordered by stay ID, so many local updates used empty memory or memory from other stays. Zeroing empty actions did not help. Dose decoding is also imperfect, especially for treatment innovations. I’ll fix the boundaries and test attention conventions against saved states; matching those states is the next prerequisite.

```python
import types
import numpy as np
import torch

start_by_stay66 = {int(sid65[b66]): int(b66) for b66 in boundary65}
np_original66 = np

class NumpyBoundaryProxy66:
    def __getattr__(self, name):
        return getattr(np_original66, name)
    def searchsorted(self, a, v, *args, **kwargs):
        if a is s4["stay_id"]:
            return start_by_stay66[int(v)]
        return np_original66.searchsorted(a, v, *args, **kwargs)

globals66 = dict(latent_step61.__globals__)
globals66["np"] = NumpyBoundaryProxy66()
latent_step66 = types.FunctionType(
    latent_step61.__code__, globals66, "latent_step66",
    latent_step61.__defaults__, latent_step61.__closure__)

print("=== IMPLEMENTATION CONSTANTS ===")
for name66 in ["latent_step61", "wm_attention", "wm_layer", "wm_mlp"]:
    print(name66, [c66 for c66 in globals()[name66].__code__.co_consts
                   if isinstance(c66, (str, int, float, tuple))])

print("\n=== RECURRENCE AFTER BOUNDARY FIX ===")
test_rows66 = np.asarray(sample61)[[0, 1, 3, 8, 14, 15]]
old_kv66, old_mem66 = wm_kv_layout, wm_memory_mode
recurrence_trials66 = []
try:
    for kv66 in ["global", "perhead"]:
        for mem66 in ["unnormalized", "norm_then_age",
                      "age_then_norm", "age_then_plainnorm"]:
            wm_kv_layout, wm_memory_mode = kv66, mem66
            errs66, out_errs66 = [], []
            for j66 in test_rows66:
                z66 = s4["x"][j66:j66+1, 960:992]
                h66 = latent_step66(int(j66), z66)
                target_h66 = torch.from_numpy(
                    s4["x"][j66:j66+1, 192:960]).reshape(1, 4, 192)
                errs66.append(float(torch.mean((h66-target_h66)**2).sqrt()))
                out66 = wm_norm(h66.mean(dim=1), "core.out_norm")
                target_out66 = torch.from_numpy(s4["x"][j66:j66+1, :192])
                out_errs66.append(float(torch.mean((out66-target_out66)**2).sqrt()))
            recurrence_trials66.append({
                "kv": kv66, "memory": mem66,
                "h_rmse": np.asarray(errs66),
                "out_rmse": np.asarray(out_errs66)})
            print(kv66, mem66,
                  "h RMSE", np.round(errs66, 4),
                  "out RMSE", np.round(out_errs66, 5))
finally:
    wm_kv_layout, wm_memory_mode = old_kv66, old_mem66

print("\n=== REMAINING LATENT-STEP OPERATIONS ===")
instructions66 = list(dis.get_instructions(latent_step61))
current_line66 = 0
parts66 = []
for ins66 in instructions66:
    if ins66.starts_line is not None:
        current_line66 = ins66.starts_line
    if current_line66 >= 34 and ins66.opname not in {
        "RESUME", "CACHE", "PRECALL", "PUSH_NULL", "NOP"}:
        if ins66.starts_line is not None:
            parts66.append("\n" + str(current_line66) + ":")
        parts66.append(ins66.opname + " " + ins66.argrepr)
print(" | ".join(parts66)[:3100])

with open("/work/notes.md", "a") as f66:
    f66.write(
        "\n## Runs 65–66: reconstruction bookkeeping bug\n"
        "- Forward states are in interleaved stay order, not sorted by stay ID.\n"
        "- np.searchsorted on stay IDs selected incorrect memory boundaries; fixed local diagnostic boundaries.\n"
        "- Zeroing empty-action vectors worsened posterior reconstruction and was not retained.\n"
        "- Held-out action-u linear rate decoding active R2: insulin .653, NE .519; innovation R2 .281/.180.\n"
        "- No weight edits. Revalidate recurrence conventions using saved posterior means.\n")
```