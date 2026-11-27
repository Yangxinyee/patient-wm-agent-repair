Official rollouts confirm that the current edit is still substantially under-responsive: insulin doubling gives −2.68, while norepinephrine doubling gives essentially zero. The records support roughly −2 and +3 per rate unit for the next reading. I’ll recover the persistence estimates and inspect the untested routing candidate to determine whether it offers a useful, factual-fit-preserving direction.

```python
print("=== ORIGINAL IDENTIFICATION NOTES ===")
lines216 = Path("/work/notes.md").read_text().splitlines()
print("\n".join(lines216[:60]))

print("\n=== ROUTING CANDIDATES ===")
for name216 in ["targets158", "event_scale158", "D_best158", "dall158",
                "scores158", "nonlinear158", "best188", "candidate158",
                "zero_weights154", "candidate139", "candidate193"]:
    obj216 = globals().get(name216)
    if obj216 is None:
        continue
    print(name216)
    if isinstance(obj216, dict):
        for key216, val216 in obj216.items():
            if torch.is_tensor(val216):
                print(key216, tuple(val216.shape),
                      "norm", round(float(val216.norm()), 5))
            elif isinstance(val216, np.ndarray):
                print(key216, val216.shape,
                      "norm", round(float(np.linalg.norm(val216)), 5),
                      "small values", val216.tolist() if val216.size <= 16 else "")
            else:
                print(key216, str(val216)[:550])
    elif isinstance(obj216, np.ndarray):
        print(obj216.shape, "norm", float(np.linalg.norm(obj216)),
              obj216.tolist() if obj216.size <= 16 else "")
    else:
        print(str(obj216)[:1000])

print("\n=== CURRENT VS ORIGINAL WEIGHTS ===")
for name216 in edited():
    cur216, orig216 = load(name216), load_original(name216)
    cur216 = torch.as_tensor(cur216)
    orig216 = torch.as_tensor(orig216)
    print(name216, tuple(cur216.shape),
          "difference norm", float((cur216-orig216).norm()))
print("current added routing norms",
      torch.as_tensor(load("z_in.weight"))[:, 30:32].norm(dim=0).tolist())
if "zero_weights154" in globals():
    for name216, value216 in zero_weights154.items():
        print("zero base", name216,
              "difference from original",
              float((torch.as_tensor(value216) -
                     torch.as_tensor(load_original(name216))).norm()))

print("\n=== RECORD OBJECTS ===")
for name216, obj216 in list(globals().items()):
    if isinstance(obj216, pd.DataFrame) and len(obj216) > 1000:
        print(name216, obj216.shape, list(obj216.columns)[:12])

print("\n=== TRAINING DRUG RATE DISTRIBUTIONS ===")
train216 = pd.read_csv("/data/train.csv",
                      usecols=["stay_id", "hour", "event", "value", "rate"])
for drug216 in ["insulin", "norepinephrine"]:
    rates216 = train216.loc[train216.event.eq(drug216), "rate"]
    print(drug216, rates216.describe(
        percentiles=[.05, .25, .5, .75, .95]).round(4).to_dict())

with open("/work/notes.md", "a") as f216:
    f216.write("\n## Run 216: quantify targets and untested routing candidate\n"
               "- Official current shifts: insulin -2.682/+1.339; NE -0.069/+0.369.\n"
               "- Training one-hour effects are approximately -2 and +3 per rate unit; persistence matters for rollout targets.\n"
               "- Inspecting retained candidate158 and channel scales before selecting a test.\n"
               "- No weights changed. Python runs: 216; forward passes: 14.\n")
```