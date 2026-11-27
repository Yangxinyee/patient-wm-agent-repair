The emulator hardcodes four attention heads and treats every drug atom as a start record. Either convention could explain the mismatch, so I’ll check both against the configuration and problematic bundles before editing anything. If correcting them improves GPU-state agreement, that supports the diagnosis; otherwise, we should rule it out.

```python
print("=== CONFIGURATION ===")
print(VOCAB["config"])
print("\n=== RECENT FINITE-DOSE TEST ===")
for name in ("testq162", "testp162", "signal162", "sig163"):
    if name in globals():
        v = globals()[name]
        print(name, v.detach().cpu().numpy() if torch.is_tensor(v) else v)

print("\n=== PROBLEMATIC BUNDLES ===")
for jj in (3, 585, 1110, 1675):
    sid = int(s4["stay_id"][jj])
    hour = float(s4["hour"][jj])
    key = (sid, int(round(hour * 60)))
    print("j", jj, "stay", sid, "hour", hour,
          "drug flags", bool(s4["has_insulin"][jj]), bool(s4["has_norepinephrine"][jj]),
          "n_atoms", s4.get("n_atoms", np.zeros(len(s4["hour"])))[jj])
    print(groups4[key].to_string(index=False)[:900])

print("\n=== EXACT ACTION ENCODER HELPER ===")
compact_restart171(exact_u162, 1500)

print("\n=== CHANNEL/CALIBRATION OBJECT NAMES ===")
keys = [k for k in globals() if not k.startswith("_")
        and any(t in k.lower() for t in
                ("pair", "private", "channel", "proxy", "strength", "calib",
                 "gate", "tilt", "sensor", "report", "decode", "reserved"))]
print(" | ".join(sorted(keys))[:2800])
print("\n=== BEST RECONSTRUCTION SETTINGS ===")
for name in ("best67", "best82"):
    if name in globals():
        print(name, str(globals()[name])[:600])
```
