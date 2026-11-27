The recovered trials show little factual headroom: even the shrunken norepinephrine edit exceeded the limit. I’ll first examine the insulin edit’s weaker ×½ response. If its dose detector was fitted mainly to ×2, fitting both contrasts may improve ×½ without increasing factual perturbations. If both contrasts are already balanced, that explanation is wrong.

```python
import torch

print("=== INSULIN EDIT CONSTRUCTION NOTES ===")
notes217 = open("/work/notes.md").read()
budget217 = 4400
for sec217 in re.split(r"(?=^## )", notes217, flags=re.M):
    head217 = sec217.split("\n", 1)[0]
    if re.search(r"\b(52|57|58|62|63|67|68|70)\b", head217):
        out217 = sec217[:min(850, budget217)]
        print(out217)
        budget217 -= len(out217)
        if budget217 <= 0:
            break

print("\n=== BASELINE RANK ===")
d217 = np.load("/work/baseline77.npz")["delta"].astype(np.float64)
u217, s217, vt217 = np.linalg.svd(d217, full_matrices=False)
print("singular values:", s217[:8].round(5).tolist())
print("edited tensors:", edited())

print("\n=== HELPER DEPENDENCIES ===")
refs217 = set()
for name217 in ["contrast_metrics52", "feature_metrics57",
                "full_insulin_slope58", "fn67", "fun68", "fun77"]:
    fun217 = globals().get(name217)
    if callable(fun217):
        print(name217, fun217.__code__.co_names)
        refs217.update(fun217.__code__.co_names)
print("referenced persistent objects:")
for name217 in sorted(refs217):
    obj217 = globals().get(name217)
    if isinstance(obj217, (np.ndarray, torch.Tensor, pd.DataFrame)):
        print(name217, type(obj217).__name__, tuple(obj217.shape))
    elif isinstance(obj217, dict):
        print(name217, "dict keys", list(obj217)[:12])
    elif isinstance(obj217, (int, float, str)):
        print(name217, obj217)

print("\n=== TRAINING-CONTRAST OBJECT NAMES ===")
items217 = []
for name217, obj217 in list(globals().items()):
    if (re.search(r"(train|half|ins|coeff|feature|direction|target|E2|Eh)", name217, re.I)
        and isinstance(obj217, (np.ndarray, torch.Tensor, pd.DataFrame))):
        items217.append((name217, type(obj217).__name__, tuple(obj217.shape)))
print(items217[-65:])

print("\n=== FORWARD FILE NAMES ===")
for k217 in [10, 20, 21, 22, 23]:
    print(k217, os.listdir("/work/fwd_" + str(k217)))

with open("/work/notes.md", "a") as f217:
    f217.write(
        "\n## Run 217: investigate insulin half-dose asymmetry\n"
        "Prior NE shrinkage .4 still costs .00652 factual NLL, leaving little "
        "margin. Inspecting whether the accepted insulin detector was constrained "
        "mainly on x2 and can be refitted jointly on x2/half while retaining "
        "factual guards. No weights changed. Python runs: 217; forwards: 23.\n"
    )
```
