Blending the routing weights made factual loss worse, increasing it by 0.02537 nats. That refutes this interpolation approach. I’ll restore the full corrected route and turn off its added drug-response signal. This isolates whether the factual damage comes from routing itself or from the response gain.

```python
def install_checkpoint285(values):
    for name285 in list(edited()):
        drop_edit(name285)
    for name285, value285 in values.items():
        if name285 in NAMES:
            array285 = as_numpy282(value285).astype(np.float32, copy=True)
            assert tuple(array285.shape) == tuple(SHAPES[name285])
            save_edit(name285, torch.from_numpy(array285))

def full_route285(insulin_gain=0.0, ne_gain=None):
    if ne_gain is None:
        ne_gain = insulin_gain
    result285 = {k: v.copy() for k, v in corrected284.items()}
    delta285 = corrected284["z_in.weight"] - neutral284["z_in.weight"]
    active285 = np.flatnonzero(np.linalg.norm(delta285, axis=0) > 1e-7)
    print("Response columns", active285.tolist(),
          "norms", np.linalg.norm(delta285[:, active285], axis=0).round(5).tolist())
    assert set(active285.tolist()).issubset({12, 16}), "Unexpected response support"
    result285["z_in.weight"] = neutral284["z_in.weight"].copy()
    result285["z_in.weight"][:, 16] += insulin_gain * delta285[:, 16]
    result285["z_in.weight"][:, 12] += ne_gain * delta285[:, 12]
    return result285

zero285 = full_route285(0.0)
install_checkpoint285(zero285)
np.savez("/work/zero_route285.npz", **zero285)
print("Installed full corrected route with added response OFF.")

print("\nNEUTRAL CHECKPOINT EVIDENCE")
for term285 in ["## Python 209", "## Python 203", "## Python 220", "## Python 243"]:
    index285 = notes282.find(term285)
    if index285 >= 0:
        print(notes282[index285:index285+800])

print("\nRETAINED ROUTING AND REPLAY HELPERS")
helper_names285 = [
    k for k in list(globals())
    if not k.startswith("_") and callable(globals()[k])
    and re.search(r"route|replay|compens|token|step|head|predict|view", k, re.I)
]
print(helper_names285[-45:])

print("\nNEUTRAL VS ORIGINAL PARAMETER CHANGE NORMS")
for name285, value285 in neutral284.items():
    original285 = as_numpy282(load_original(name285))
    difference285 = value285 - original285
    print(name285, tuple(value285.shape),
          "delta norm", round(float(np.linalg.norm(difference285)), 5))

with open("/work/notes.md", "a") as log285:
    log285.write(
        "\n## Python 285 — reject interpolated routing; isolate overhead\n"
        "- Forward 31 / fwd_32: blended284 dNLL +0.02537 on second 100 tuning stays; rejected.\n"
        "- Restored full corrected269 route with both added z_in response columns at gain zero.\n"
        "- zero_route285 is diagnostic only, not accepted. Next factual pass isolates routing overhead.\n"
    )
```