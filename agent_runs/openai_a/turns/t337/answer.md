The architecture confirms that dose edits pass through the recurrent state, and the early regression estimates vary with adjustment choices. Those regressions alone are not enough to establish drug effects. The output was truncated, so I’ll recover a compact evidence summary and calculate the paired 400-stay factual result; this will show whether the current edit meets the fit limit on tuning.

```python
_sections = re.split(r"(?m)(?=^## )", _recovery_notes)
print("=== EARLY IDENTIFICATION SUMMARY ===")
_summary = []
for _section in _sections[:46]:
    _lines = _section.splitlines()
    _summary.extend(
        x for x in _lines
        if x.startswith("## ") or (
            x.startswith("- ") and re.search(
                r"effect|causal|instrument|IV|identif|insulin|norepinephrine|null|target|confound|placebo|boundary",
                x, re.I
            )
        )
    )
print("\n".join(_summary)[:4300])

print("=== CHECKPOINTS AND RELEVANT FUNCTIONS ===")
for _name, _obj in list(globals().items()):
    if _name.startswith("_"):
        continue
    if "checkpoint" in _name.lower():
        print(_name, type(_obj).__name__,
              list(_obj)[:8] if isinstance(_obj, dict) else getattr(_obj, "shape", ""))
    elif callable(_obj) and getattr(_obj, "__module__", "") == "__main__" and re.search(
        r"gate|graft|build|install|restore|activity|candidate", _name, re.I
    ):
        try:
            print(_name, str(inspect.signature(_obj)))
        except Exception:
            print(_name)
print("=== TUNING-ID OBJECTS ===")
for _name, _obj in list(globals().items()):
    if not _name.startswith("_") and re.search(r"ids|tuning", _name, re.I):
        print(_name, type(_obj).__name__, getattr(_obj, "shape", ""),
              f"length={len(_obj)}" if hasattr(_obj, "__len__") else "")

_factual_parts = []
for _k in (36, 37, 38, 39):
    _o = pd.read_csv(f"/work/fwd_{_k}/factual_original.csv")
    _e = pd.read_csv(f"/work/fwd_{_k}/factual_edited.csv")
    _nllcols = [c for c in _o if "nll" in c.lower()]
    if _k == 36:
        print("=== FACTUAL COLUMNS ===", list(_o.columns))
        print(_o.head(1).to_string(index=False))
    print("batch", _k, "rows", len(_o),
          "NLL columns", _nllcols,
          "means", {c: (float(_o[c].mean()), float(_e[c].mean())) for c in _nllcols})
    if len(_nllcols) == 1:
        assert len(_o) == len(_e)
        for _c in ("stay_id", "hour", "event", "value", "bin", "fine_bin"):
            if _c in _o and _c in _e:
                assert _o[_c].equals(_e[_c]), (_k, _c)
        _part = _o.copy()
        _part["_original_nll"] = _o[_nllcols[0]].to_numpy()
        _part["_edited_nll"] = _e[_nllcols[0]].to_numpy()
        _part["_delta_nll"] = _part["_edited_nll"] - _part["_original_nll"]
        _part["_batch"] = _k
        _factual_parts.append(_part)

if len(_factual_parts) == 4:
    _sharp_full_factual = pd.concat(_factual_parts, ignore_index=True)
    print("=== POOLED FACTUAL ===",
          _sharp_full_factual[["_original_nll", "_edited_nll", "_delta_nll"]].mean().to_dict(),
          "readings", len(_sharp_full_factual))
    if "stay_id" in _sharp_full_factual:
        _by_stay = _sharp_full_factual.groupby("stay_id")["_delta_nll"].agg(["sum", "count"])
        assert len(_by_stay) == 400
        _rng = np.random.default_rng(20261108)
        _draw = _rng.integers(0, len(_by_stay), size=(10000, len(_by_stay)))
        _boot = (_by_stay["sum"].to_numpy()[_draw].sum(1) /
                 _by_stay["count"].to_numpy()[_draw].sum(1))
        print("stay-bootstrap delta interval", np.quantile(_boot, [0.025, 0.5, 0.975]))
    with open("/work/notes.md", "a") as _fh:
        _fh.write(
            "\n## Sharp240 pooled factual recovery; Python 284, forwards 39/50\n"
            f"- Paired batches36–39: {_sharp_full_factual['_delta_nll'].mean():.8f} "
            f"NLL increase across {len(_sharp_full_factual)} readings.\n"
            "- No weights changed in this recovery. Independent rollout validation remains necessary.\n"
        )
```
