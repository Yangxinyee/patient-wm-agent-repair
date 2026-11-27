The reserved channels respond clearly to doubled doses at drug bundles. But removing the action while keeping that context produces very large channel values. That is a warning, not yet proof of a defect, because measurement contexts differ. I’ll check actual non-drug bundles; large values there would support adding an explicit action gate before recalibrating the drug effects.

```python
print("=== ALL-BUNDLE CONTEXT INTERFACE ===")
for _k206 in ["contexts_audit", "baseline138", "r132_targets",
              "calibration_stays199", "calibration_ids138"]:
    if _k206 in globals():
        print(_k206, brief202(globals()[_k206]))
        if _k206 == "r132_targets":
            print(str(globals()[_k206])[:1600])

if "contexts_audit" in globals() and len(contexts_audit) == len(rr33_u_all):
    _allcontext206 = {"ctx": contexts_audit, "e": rr33_e_all}
    _allz206 = latent205(candidate199, rr33_u_all, _allcontext206).numpy()
    _hasaction206 = np.linalg.norm(rr33_u_all.numpy(), axis=1) > 1e-6
    for _label206, _mask206 in [("action", _hasaction206),
                                ("no action", ~_hasaction206)]:
        print(_label206, "n", int(_mask206.sum()),
              "mean", _allz206[_mask206, 30:32].mean(0),
              "std", _allz206[_mask206, 30:32].std(0),
              "quantiles", np.quantile(_allz206[_mask206, 30:32],
                                      [.01, .5, .99], axis=0))
else:
    print("No aligned all-bundle context array found.")

print("\n=== OVERWRITTEN POSTERIOR HIDDEN ROWS ===")
_diff206 = torch.linalg.vector_norm(
    zero_weights154["z_post.0.weight"] - load_original("z_post.0.weight"), dim=1)
_rows206 = torch.where(_diff206 > 1e-6)[0].tolist()
print(_rows206)
for _row206 in _rows206[:8]:
    print(_row206, "weight diff", float(_diff206[_row206]),
          "bias", float(zero_weights154["z_post.0.bias"][_row206]))

print("\n=== RECORD DATAFRAME INTERFACES ===")
for _k206, _v206 in list(globals().items()):
    if isinstance(_v206, pd.DataFrame) and {"stay_id", "hour", "event"} <= set(_v206.columns):
        print(_k206, _v206.shape, list(_v206.columns)[:10])

print("\n=== MOST RECENT BASELINE-CONSTRUCTION NOTES ===")
_n206 = Path("/work/notes.md").read_text()
for _term206 in ["Run 89:", "Run 138:", "Run 139", "Runs 139"]:
    _pos206 = _n206.find(_term206)
    if _pos206 >= 0:
        _start206 = _n206.rfind("\n## ", 0, _pos206)
        _end206 = _n206.find("\n## ", _pos206)
        print(_n206[_start206:_end206 if _end206 >= 0 else None][:1000])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 206: assay non-action leakage of dose channels\n"
            "- Current reserved channels have strong doubled-dose contrasts "
            "at drug bundles; empty actions in drug contexts yield large values.\n"
            "- Checking actual non-action contexts before deciding whether "
            "explicit gating is needed. No weights changed.\n"
            "- Python runs: 206; forward passes: 14.\n")
```