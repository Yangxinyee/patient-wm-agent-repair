The ordinary recurrent-input decoder is not clean enough: better dose calibration comes with large signals on inactive bundles. I’ll check the previously constructed gated dose-innovation features instead and inspect how attention carries them. If those features are accurate and inactive when no drug runs, they may support a more selective pathway.

```python
print("=== GATED INNOVATION FEATURE QUALITY ===")
F195 = features169(Uref168)
print("shape:", F195.shape)
for col195, drug195 in enumerate(["insulin", "norepinephrine"]):
    active195 = np.asarray(presence168)[:, col195] > 0
    inactive195 = ~active195
    expected195 = (
        R193[:, col195]
        - policy168[drug195][1].predict(np.asarray(context))
    ) * np.asarray(presence168)[:, col195]
    print(drug195,
          "active feature RMS", np.sqrt(np.mean(F195[active195, col195] ** 2)),
          "inactive RMS", np.sqrt(np.mean(F195[inactive195, col195] ** 2)),
          "innovation RMSE", np.sqrt(np.mean((F195[:, col195] - expected195) ** 2)))
for key195 in [(0.0, 1.0), (0.5, 1.0), (2.0, 1.0), (3.0, 1.0),
               (1.0, 0.0), (1.0, 0.5), (1.0, 2.0), (1.0, 3.0)]:
    contrast195 = features169(Uaug168[key195]) - F195
    target195 = R193 * (np.asarray(key195) - 1)
    col195 = 0 if key195[0] != 1 else 1
    active195 = np.asarray(presence168)[:, col195] > 0
    y195 = target195[active195, col195]
    p195 = contrast195[active195, col195]
    print(key195, "slope", float(p195 @ y195 / (y195 @ y195)),
          "contrast RMSE", np.sqrt(np.mean((p195 - y195) ** 2)))

print("\n=== ATTENTION CONTRACT ===")
print("config:", VOCAB["config"])
print("attention locals:", attention136.__code__.co_varnames)
print("attention constants:", attention136.__code__.co_consts)
dis.dis(attention136)
print("\nPrelude norm constant projection:",
      float((W["core.prelude.0.norm1.bias"] /
             W["core.prelude.0.norm1.weight"]).sum()))
print("base169 versus original tensor RMS:")
for name195, value195 in base169.items():
    print(name195, float(np.sqrt(np.mean(
        (np.asarray(value195) - load_original(name195)) ** 2))))
print("latent direction norms:",
      {k: float(np.linalg.norm(v)) for k, v in latent_directions161.items()})
with open("/work/notes.md", "a") as f195:
    f195.write("\n## Python 195 — reject unselective E decoder; inspect gated route\n"
               "- Ordinary E decoding trades better counterfactual slopes for substantial inactive leakage.\n"
               "- Testing the earlier gated innovation features before considering a dedicated attention channel.\n"
               "- No active edits; any channel construction remains a reversible engineering probe.\n")
```