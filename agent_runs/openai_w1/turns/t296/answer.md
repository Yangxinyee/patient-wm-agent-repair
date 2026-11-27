The recovered counterfactual tokens passed normalization checks. The run then stopped in the inventory printer, before fitting or changing weights. I will fix that printer and recover the dose units, then test whether a constrained linear correction can separate dose changes from factual variation.

```python
print("=== PHYSICAL DOSE / ALIGNMENT ARRAYS ===")
_seen260 = set()
_count260 = 0
for _k260, _v260 in list(globals().items()):
    if _k260.startswith("_") or id(_v260) in _seen260:
        continue
    if isinstance(_v260, pd.DataFrame) and _v260.shape[0] == 5215:
        print(_k260, "DataFrame", list(_v260.columns))
    if isinstance(_v260, (np.ndarray, torch.Tensor)) and _v260.ndim <= 2 and \
       _v260.shape[0] == 5215:
        _a260 = np.asarray(_v260)
        if _a260.size <= 5215*6 and any(s in _k260.lower() for s in
                ["rate", "dose", "target", "phys", "drug", "pres", "meta"]):
            print(_k260, _a260.shape, "min/max/mean",
                  float(np.nanmin(_a260)), float(np.nanmax(_a260)),
                  float(np.nanmean(_a260)))
            _seen260.add(id(_v260))
            _count260 += 1
            if _count260 >= 35:
                break

print("\n=== CARRIER SCALING ===")
for _k260 in ["lc_dims", "lc_scales", "lc_units", "lc_meta", "rp_meta",
              "rp_y", "lc_y", "lc_rates", "lc_dose", "lc_doses"]:
    if _k260 in globals():
        print(_k260, brief258(globals()[_k260]))

print("\n=== BRANCH BUILDERS ===")
for _name260 in ["branch195", "branch186", "rp_encode", "posterior211", "posterior238"]:
    _f260 = globals().get(_name260)
    if callable(_f260):
        print(_name260, inspect.signature(_f260),
              "names", _f260.__code__.co_names,
              "constants", [c for c in _f260.__code__.co_consts
                            if isinstance(c, (str, int, float, tuple))][:18])

print("\n=== COUNTERFACTUAL SOURCE DICTS ===")
_count260 = 0
for _k260, _v260 in list(globals().items()):
    if isinstance(_v260, dict) and _k260 not in ["cf239", "Es259", "nts259"] and \
       (("insulin", 2.0) in _v260 or (0, 2.0) in _v260):
        _key260 = ("insulin", 2.0) if ("insulin", 2.0) in _v260 else (0, 2.0)
        print(_k260, brief258(_v260[_key260]))
        _count260 += 1
        if _count260 == 12:
            break

print("\n=== LATENT CONTRAST SCALE ===")
for _drug260, _dim260 in [("insulin", 16), ("norepinephrine", 9)]:
    for _m260 in [0., .5, 1.5, 2., 3.]:
        _obj260 = cf239[(_drug260, _m260)]
        _mask260 = np.asarray(_obj260["mask"], dtype=bool)
        _dz260 = _obj260["z"][:, _dim260] - zs238[:, _dim260]
        print(_drug260, _m260, "mean/std delta z",
              round(float(_dz260[_mask260].mean()), 5),
              round(float(_dz260[_mask260].std()), 5))

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 259–260\n"
            "CF token recovery succeeded: pooled features are mean-normalized tokens; all ten "
            "recovered action tokens satisfy LN mean/variance checks. Run259 then failed only "
            "in an inventory printer using torch.size as a number. Run260 fixes inventory and "
            "recovers physical-dose/carrier scaling. No weights changed. "
            "Counts: 260 Python runs, 27 forwards.\n")
```