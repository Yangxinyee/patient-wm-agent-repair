The four-head convention is correct. Two concrete emulator problems remain: some bundles contain extra model atoms not present as CSV rows, and the saved stays are not sorted, making `searchsorted` invalid for memory boundaries. I’ll recover the END-atom convention next. Including those atoms and the correct within-stay memory should improve both ordinary-state and dose-channel agreement.

```python
print("=== EXISTING ACTION-BUNDLE BUILDER ===")
compact_restart171(action_vectors4, 3600)
print("\n=== END-ATOM CONVENTION TESTS ===")
for name in ("end_convention_trials", "action_reconstruction_trials",
             "action_unit_trials", "action_probe_exact_doses"):
    if name in globals():
        text = str(globals()[name])
        print(name, text[:1250])

print("\n=== END LOOKUP EXAMPLES ===")
for key in [(int(s4["stay_id"][j]), int(round(float(s4["hour"][j])*60)), ev)
            for j in (3, 585, 1110, 1675)
            for ev in ("insulin", "norepinephrine")]:
    print(key, end_lookup_local.get(key, None))

print("\n=== ENCODER HELPER ===")
compact_restart171(set_encode_single_local, 1000)
print("\n=== NOTEBOOK END-ATOM FINDINGS ===")
sections = re.split(r"(?=^## )", pathlib.Path("/work/notes.md").read_text(), flags=re.M)
matches = [s for s in sections if re.search(r"\bEND\b|end.atom|end.record|end convention", s, re.I)]
print("\n".join(matches)[:1800])
print("\n=== ACTUAL MEMORY BOUNDARIES ===")
for j in (3, 585, 1110, 1675):
    sid = s4["stay_id"][j]
    real_start = int(np.flatnonzero(s4["stay_id"] == sid)[0])
    print(j, "real start", real_start, "searchsorted", int(np.searchsorted(s4["stay_id"], sid)))
```
