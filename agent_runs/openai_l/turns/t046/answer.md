The vocabulary confirms separate WORLD and END atom kinds, but the output was truncated before the bundle-count check. I’ll run that check alone. If exported counts equal record rows plus infusion ends, the action reconstruction needs both kinds of atoms.

```python
print("=== RAW EVENTS, FIRST EXPORTED TRAINING STAY ===")
id41 = int(np.unique(s40["stay_id"])[0])
r41 = raw40[(raw40.stay_id == id41) & (raw40.hour <= 4)].sort_values(["hour", "event"])
print(r41.to_string(index=False))

print("\n=== EXPORTED BUNDLES ===")
b41 = bundle40[(bundle40.stay_id == id41) & (bundle40.hour <= 4)].copy()
print(b41.to_string(index=False))

r41all = raw40[raw40.stay_id == id41].copy()
r41all["minute_key"] = np.rint(r41all.hour * 60).astype(int)
world41 = r41all.groupby("minute_key").size()
ends41 = r41all[r41all.event.isin(["insulin", "norepinephrine"])].copy()
ends41["minute_key"] = np.rint(ends41.end_hour * 60).astype(int)
endcounts41 = ends41.groupby("minute_key").size()
b41["minute_key"] = np.rint(b41.hour * 60).astype(int)
b41["world_rows"] = b41.minute_key.map(world41).fillna(0).astype(int)
b41["infusion_ends"] = b41.minute_key.map(endcounts41).fillna(0).astype(int)
b41["unexplained"] = b41.n_atoms - b41.world_rows - b41.infusion_ends
print("\n=== COUNT ACCOUNTING ===")
print(b41[["hour", "n_atoms", "world_rows", "infusion_ends", "unexplained"]].to_string(index=False))

print("\n=== ALL EMBEDDING TENSOR SHAPES ===")
for name in NAMES:
    if name.startswith("embed."):
        print(name, SHAPES[name])

print("\n=== CANDIDATE CONFIGURATIONS ONLY ===")
def compact41(obj):
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        return ("array", tuple(obj.shape))
    if isinstance(obj, dict):
        return {k: compact41(v) for k, v in list(obj.items())[:3]}
    if isinstance(obj, (tuple, list)):
        return [compact41(v) for v in obj[:8]]
    return obj
for drug, candidates in action_candidates37.items():
    print(drug, compact41(candidates[:2] if isinstance(candidates, list) else candidates))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 41\nRun 40 output was truncated before record/count diagnostics. "
            "Repeating only compact bundle accounting and embedding shapes. No edits.\n")
```