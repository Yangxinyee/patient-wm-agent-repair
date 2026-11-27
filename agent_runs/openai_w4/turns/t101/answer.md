The insulin gate correctly separates drug and non-drug bundles on held-out stays under all four dose edits. A two-pole curve also fits the six insulin lag estimates closely, but that alone does not validate its longer tail. The target calculation stopped on a pandas column-name collision. I will fix it and recover the fixtures needed to measure how a posterior edit affects subsequent readings.

```python
kernel_models88, target_rows89 = {}, []
for drug89 in ["insulin", "norepinephrine"]:
    empirical89 = np.asarray(provisional_kernels[drug89])
    sign89 = -1.0 if drug89 == "insulin" else 1.0
    result89 = least_squares(
        lambda p: two_pole_kernel88([sign89*p[0], p[1], p[2]], 6) - empirical89,
        x0=[abs(empirical89[0]), .6 if sign89 < 0 else .5,
            .6 if sign89 < 0 else .3],
        bounds=([0, 0, 0], [10, .98, .98]))
    parameters89 = result89.x.copy()
    parameters89[0] *= sign89
    kernel_models88[drug89] = two_pole_kernel88(parameters89, 24)
    print("Kernel", drug89, parameters89.round(5),
          "RMSE", np.sqrt(np.mean(result89.fun**2)),
          "first12", kernel_models88[drug89][:12].round(3))
    for factor89, edit89 in [(2.0, "insulin_x2" if sign89 < 0 else "ne_x2"),
                            (.5, "insulin_half" if sign89 < 0 else "ne_half")]:
        rows89 = base_roll.loc[base_roll["edit"].eq(edit89), ["stay_id", "hour"]].copy()
        effects89 = []
        for sid89, hour89 in rows89.itertuples(index=False):
            hour89 = int(round(hour89))
            rates89 = tpan.loc[sid89, drug89].fillna(0)
            effects89.append(sum(
                (factor89 - 1) * float(rates89.loc[j]) *
                kernel_models88[drug89][hour89 - j - 1]
                for j in range(12, hour89) if j in rates89.index))
        rows89["effect"] = effects89
        by_stay89 = rows89.groupby("stay_id")["effect"].mean()
        original89 = base_roll.loc[base_roll["edit"].eq(edit89)].groupby("stay_id")["shift"].mean()
        target_rows89.append({"edit": edit89, "stays": len(by_stay89),
                              "kernel_target": by_stay89.mean(),
                              "original_prediction": original89.mean()})
target_table89 = pd.DataFrame(target_rows89)
print("\nPROVISIONAL TARGETS\n", target_table89.round(4).to_string(index=False))
print("\nHEAD FIXTURE NAMES")
for name89, value89 in list(globals().items()):
    if "74" in name89 and not callable(value89):
        if isinstance(value89, dict):
            print(name89, "dict", list(value89)[:5])
        elif isinstance(value89, pd.DataFrame):
            print(name89, value89.shape, list(value89.columns))
        elif hasattr(value89, "shape"):
            print(name89, tuple(value89.shape))
for key89, value89 in list(pair80.items())[:1]:
    print("pair80", key89, [(type(v).__name__, getattr(v, "shape", None)) for v in value89])
print("\nLONGITUDINAL OBJECT NAMES",
      [k for k in globals() if any(s in k.lower() for s in
       ["moment", "horizon", "longitud", "deconv", "persist"])])
notes89 = Path("/work/notes.md").read_text()
at89 = notes89.find("## Run 20")
if at89 < 0:
    at89 = notes89.find("## Run 18")
print("\nEARLIER LONGITUDINAL NOTES\n", notes89[at89:at89+2800])
print("\nSELECTED FUNCTION RETURNS")
for name89 in ["transition73", "head_features_fixed"]:
    inst89 = list(dis.get_instructions(globals()[name89]))
    print(name89, [(x.opname, x.argrepr) for x in inst89[-15:]])
note("Runs 88–89: insulin action gate has perfect held-out classification under all four controlled dose inputs. Two-pole kernel fits six insulin lags closely but longer-tail targets remain provisional. Corrected a pandas shift-column collision. No edits.")
```
