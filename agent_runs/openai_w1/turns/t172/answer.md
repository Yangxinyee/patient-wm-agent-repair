The records support lower glucose after more insulin and higher MAP after more norepinephrine, but these one-step estimates do not yet determine the rollout targets. The baseline responses are also asymmetric. I’ll recover the lag estimates and inspect the saved embedding probe before testing it; a useful candidate must improve dose responses without exceeding the factual-loss limit.

```python
import re

print("=== BASELINE ROLLOUT SUMMARIES ===")
for k in (1, 5):
    path = f"/work/fwd_{k}/dose_edit_original.csv"
    if os.path.exists(path):
        tab = pd.read_csv(path)
        reached_tab = tab[tab["reached"].astype(bool)]
        print("Forward", k)
        print(reached_tab.groupby("edit").agg(
            mean_shift=("shift", "mean"),
            stays=("stay_id", "nunique"),
            hours=("hour", "size")).to_string())

print("\n=== SAVED NUMERIC PROBE ===")
num_original = torch.as_tensor(load_original("embed.numeric.weight"))
num_delta = recovery_numeric_154 - num_original
for j in range(6):
    col = num_original[:, j]
    projection = float(torch.dot(num_delta[:, j], col) /
                       torch.dot(col, col))
    unexplained = float(torch.linalg.vector_norm(
        num_delta[:, j] - projection * col))
    print(j, "delta_norm", float(torch.linalg.vector_norm(num_delta[:, j])),
          "relative_scale_change", projection,
          "non_scaling_component", unexplained)

print("\n=== EARLY DYNAMIC EVIDENCE ===")
note_sections = re.split(r"(?=^## )", notes_recovery, flags=re.M)
selected_sections = [
    s for s in note_sections
    if re.match(r"## (?:Run|Runs|Forward)\s+(?:[6-9]|1[0-9]|2[0-4])\b", s)
]
print("\n".join(selected_sections)[:5600])

print("\n=== HEAD CACHE SCHEMA ===")
for cache_name in ("hp_original_head", "hp_probe_head", "ce_head"):
    cache = globals().get(cache_name)
    if isinstance(cache, dict):
        print(cache_name)
        for key, value in cache.items():
            print(" ", key, type(value).__name__,
                  getattr(value, "shape", ""),
                  list(value.columns)[:14] if isinstance(value, pd.DataFrame) else "")

print("\n=== HEAD HELPER DEPENDENCIES ===")
print(ce_make_heads.__code__.co_names)
print(repr(ce_make_heads.__code__.co_consts)[:700])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 158\n"
            "Record residual estimates consistently suggest insulin lowers "
            "glucose and NE raises MAP; one-step estimates alone are not "
            "rollout targets. Recovering lag evidence, baseline summaries, "
            "and the preserved numeric probe before GPU testing. "
            "Counts: 158 Python runs, 8 forward passes.\n")
```