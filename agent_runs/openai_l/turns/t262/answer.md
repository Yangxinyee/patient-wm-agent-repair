The gated four-unit fit gives stronger immediate responses—about −1.59 for insulin and +1.97 for norepinephrine—with an offline factual loss increase of +0.00256. That is promising but not yet validated. I’ll stage it for a paired GPU audit; a loss increase above +0.005 would reject it.

```python
import torch
key235 = ("pair4", .1, .3, .3)
chosen235 = expanded234[key235]
backup235 = {name: np.asarray(load(name), dtype=np.float32).copy()
             for name in edited()}
arch235, T235 = chosen235["arch"], chosen235["T"]
W1235 = np.asarray(load_original("attr_heads.value_bin.0.weight"),
                  dtype=np.float32).copy()
b1235 = np.asarray(load_original("attr_heads.value_bin.0.bias"),
                  dtype=np.float32).copy()
units235 = np.asarray(arch235["units"], dtype=int)
rows235 = np.asarray(arch235["rows"], dtype=np.float32)
W1235[units235] = rows235[:,:192]
b1235[units235] = rows235[:,192]
for name, value in {
    "attr_heads.value_bin.0.weight": W1235,
    "attr_heads.value_bin.0.bias": b1235,
    "attr_heads.value_bin.2.weight": T235[:192].T,
    "attr_heads.value_bin.2.bias": T235[192],
}.items():
    assert np.isfinite(value).all()
    save_edit(name, torch.as_tensor(np.asarray(value, dtype=np.float32)))
print("Staged", key235, "replaced units", units235.tolist())
print("Offline deltas train/validation/tuning",
      chosen235["train_delta"], chosen235["val_delta"], chosen235["tune_delta"])
print("Immediate slopes", chosen235["slopes"])
print("Edited tensors", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 235 / broader-anchored gated head staged\n"
            "Staged pair4 lambda=.1, etaI=etaNE=.3. Replaced four first-layer "
            "GELU units and analytically fitted final value weights/bias; "
            "upstream quiet dose wire unchanged. Offline factual deltas "
            f"{chosen235['train_delta']:.6f}/{chosen235['val_delta']:.6f}/"
            f"{chosen235['tune_delta']:.6f}; immediate slopes "
            f"{chosen235['slopes']}. Backup235 preserves conservative edit. "
            "Pending paired GPU factual audit.\n")
```
