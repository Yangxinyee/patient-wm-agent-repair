Across 199 stays, the factual NLL increase is 0.00267; its approximate upper 95% bound is 0.00472, just below the budget. The earlier insulin analysis also detected a recording-policy bias and adjusted its checks accordingly. I’ll review those corrected estimates and the persistence evidence next; agreement would support accumulation targets, while sensitivity would argue against stronger edits.

```python
print("=== CORRECTED EFFECT AND FALSIFICATION EVIDENCE ===")
start155 = evidence154.find("## Run 6")
print(evidence154[start155:start155+6000])

print("\n=== LATER PERSISTENCE DIAGNOSTIC ===")
start155b = notes154.find("## Run 38")
stop155b = notes154.find("## Forward 7", start155b)
print(notes154[start155b:stop155b][:1700])

print("\n=== RECOVERABLE SOURCES ===")
for name155, obj155 in list(globals().items()):
    if isinstance(obj155, str) and any(s in obj155 for s in (
        "def fit_latent_record_model", "def sensor_iv_persistence", "def fit_dml"
    )):
        print(name155, len(obj155))

print("\n=== CURRENT CANDIDATE AND FIT OBJECTS ===")
for name155 in (
    "constraint_candidates147", "validated_B151", "validated_W151",
    "validated_gains151", "V128", "Wdose134", "fit128",
    "E119", "E120", "Eprobe119", "Eprobe120"
):
    obj155 = globals().get(name155)
    if isinstance(obj155, dict):
        print(name155, "dict keys:", list(obj155.keys())[:30])
    elif obj155 is not None:
        print(name155, type(obj155).__name__, getattr(obj155, "shape", None))
        if name155 == "validated_gains151":
            print(obj155)
```
