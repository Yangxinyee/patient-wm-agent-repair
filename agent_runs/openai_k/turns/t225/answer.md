The record-derived impulse estimates persist for several hours, so a one-step response alone would not establish a successful repair. The recovered notebook also shows that local value-head reconstruction never matched official predictions well. I’ll audit the dose-channel encoding directly, especially its behavior at zero, half, and triple doses; saturation there would undermine transfer to the extra scored interventions.

```python
import dis
import io

print("=== TOKEN HELPER BYTECODE ===")
for _k204 in ["token164", "posterior_token154", "local_token154"]:
    if _k204 in globals():
        _buf204 = io.StringIO()
        dis.dis(globals()[_k204], file=_buf204)
        print(_k204)
        print(_buf204.getvalue()[:2000])

print("\n=== PROBE AND BASELINE INTERFACES ===")
for _k204 in ["context151", "ucf151", "drug151", "rate151", "raw151",
              "truth151", "zero_weights154", "unit139", "scales139",
              "candidate_versions139", "B119", "features119",
              "r49_register", "r49_hours", "r49_sensors", "num_spec0"]:
    if _k204 in globals():
        print(_k204, brief202(globals()[_k204]))
        if isinstance(globals()[_k204], dict):
            print({k: brief202(v) for k, v in list(globals()[_k204].items())[:8]})

print("\n=== LAST-REGISTER WEIGHTS ===")
for _k204 in NAMES:
    if any(s in _k204 for s in ["last_numeric", "last_status", "type_emb", "query_norm"]):
        print(_k204, SHAPES[_k204])
        if "last_numeric" in _k204 and len(SHAPES[_k204]) == 2:
            _w204 = load_original(_k204)
            print("column norms", np.linalg.norm(np.asarray(_w204), axis=0))

print("\n=== RESERVED-CHANNEL CONSTRUCTION NOTES ===")
_n204 = Path("/work/notes.md").read_text()
for _heading204 in ["## Run 125:", "## Run 132:", "## Run 139:"]:
    _a204 = _n204.find(_heading204)
    if _a204 >= 0:
        _b204 = _n204.find("\n## ", _a204 + 4)
        print(_n204[_a204:_b204 if _b204 >= 0 else None][:650])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 204: audit dose encoding independently of invalid heads\n"
            "- Record impulse responses persist for several hours; one-step "
            "head slopes alone cannot establish rollout repair.\n"
            "- Earlier local-head reconstructions were not authoritative. "
            "Recovering exact dose-channel and probe construction.\n"
            "- Python runs: 204; forward passes: 14. No weights changed.\n")
```