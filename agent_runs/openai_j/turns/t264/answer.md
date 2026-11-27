The doubled-dose input changes are only slightly larger than the halved-dose changes, despite the larger dose difference. That suggests compression of the dose signal before the head. I’ll check whether the two edits move inputs in distinguishable directions and whether doubling reaches regions uncommon in factual records. Either finding could support a selective correction; substantial overlap would argue against it.

```python
import torch.nn.functional as F

gate234 = -torch.as_tensor(trial158_gate, dtype=torch.float32)
j234 = int(head177_free[0])
coef234 = torch.cat([
    load(head177_names[0])[j234] - 200 * gate234[:192],
    (load(head177_names[1])[j234] - 200 * gate234[-1]).reshape(1)
])
print("CURRENT COEFFICIENT CHECK",
      np.linalg.norm(coef234.numpy() - map192_candidates[(10.0, 8.0)]))

def affine234(mc, coef):
    return F.linear(mc["inputs"], coef[:192].reshape(1, -1),
                    coef[-1:].reshape(1)).squeeze(-1).numpy()

a234 = affine234(wm_train145, coef234)
g234 = affine234(wm_train145, gate234)
for event in ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]:
    mask = np.asarray(events206 == event)
    print(event, "gate quantiles",
          np.round(np.quantile(g234[:, mask], [.001, .05, .5, .95, .999]), 4),
          "tilt quantiles",
          np.round(np.quantile(a234[:, mask], [.001, .05, .5, .95, .999]), 4))

print("\nMATCHED MAP CONTRASTS")
dx234 = {}
for factor in [0.5, 2.0]:
    mc = wm_cf146[("norepinephrine", factor)]
    dx = (mc["inputs"] - wm_train145["inputs"]).mean(0).numpy()
    dx234[factor] = dx
    acf = affine234(mc, coef234)
    active = np.asarray(mc["active"], dtype=bool)
    for event in ["map_arterial", "map_cuff"]:
        mask = np.asarray(events206 == event) & active
        print(factor, event, "factual tilt",
              np.round(np.quantile(a234.mean(0)[mask], [.05, .5, .95]), 4),
              "edited tilt",
              np.round(np.quantile(acf.mean(0)[mask], [.05, .5, .95]), 4),
              "mean tilt change", float((acf-a234)[:, mask].mean()))
for event in ["map_arterial", "map_cuff"]:
    mask = np.asarray(events206 == event) & np.asarray(
        wm_cf146[("norepinephrine", 2.0)]["active"], dtype=bool)
    u, v = dx234[0.5][mask], dx234[2.0][mask]
    cos = np.sum(u*v, axis=1) / np.maximum(
        np.linalg.norm(u, axis=1)*np.linalg.norm(v, axis=1), 1e-12)
    print(event, "half/double cosine quantiles",
          np.round(np.quantile(cos, [.05, .5, .95]), 4))

print("\nPOOLED MOMENT FRONTIER")
tab234 = pd.read_csv("/work/map_pooled_moment_frontier.csv")
print("columns:", list(tab234.columns))
print(tab234.head(5).to_string(index=False))
print("\nSENSOR-SPECIFIC FRONTIER")
tab234b = pd.read_csv("/work/map_sensor_frontier.csv")
print("columns:", list(tab234b.columns))
print(tab234b.head(3).to_string(index=False))
```