The edit audit confirms exact magnitude invariance: positive rate and dose bins are identical, and the numeric dose/rate columns are zero. Drug identity and missingness remain intact. I’ll recover the sampling details so the delayed norepinephrine signal can be tested on training stays not used in the earlier checks.

```python
import dis

print("=== EDIT CONSTRUCTION NOTES ===")
_pos = _notes.find("## Run 7")
print(_notes[_pos:_pos + 3100])

print("\n=== ESTIMATION FUNCTION DETAILS ===")
for _fn_name in [
    "pre_dose_features_v2", "future_reading_v2",
    "matched_crossfit_v3", "orthogonal_coefficient_v2"
]:
    _fn = globals()[_fn_name]
    print("\n", _fn_name)
    try:
        print(inspect.getsource(_fn))
    except (OSError, TypeError):
        print("referenced names:", _fn.__code__.co_names)
        print("constants:", repr(_fn.__code__.co_consts)[:1300])
        print("return instructions:",
              [(i.opname, i.argrepr) for i in list(dis.get_instructions(_fn))[-18:]])

print("\n=== SAMPLE IDENTIFIERS ===")
print("training first/last:", ids[:5].tolist(), ids[-5:].tolist())
print("group first/last:", stay_groups_v2[:5].tolist(), stay_groups_v2[-5:].tolist())
print("remaining tuning batch A:", tuning_ids[200:300].tolist())
print("remaining tuning batch B:", tuning_ids[300:400].tolist())

print("\n=== FEATURE METADATA ===")
for _key in ["channels", "channel_names", "event_cols", "col_idx",
             "code_to_idx", "feature_names_v2", "cf_stays", "cf_nstays"]:
    if _key in globals():
        print(_key, repr(globals()[_key])[:1800])
print("cached baseline/target shapes:",
      X_ne_v2.shape, A_ne_v2.shape, map_baseline_v3.shape)

with open("/work/notes.md", "a") as _f:
    _f.write(
        "\n## Recovery audit, runs 25–27\n"
        "Re-read architecture and audited the existing edit. Numeric columns "
        "2 (z_dose) and 4 (z_rate) are exactly zero; all positive dose/rate-bin "
        "rows are identical; absent-bin rows are unchanged. Drug codes, "
        "presence indicators, other numeric columns and event timing remain. "
        "Need independent replication of the isolated late NE lag-12 signal "
        "and factual validation on the remaining 200 tuning stays.\n"
    )
```