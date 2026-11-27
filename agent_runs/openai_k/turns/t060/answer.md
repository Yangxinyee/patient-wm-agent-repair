The saved centre table has 16 columns, while the head produces 17 probabilities. I’ll add a zero-valued centre for the absent-value class and test the readout directly, avoiding the oversized disassembly. Finite predictions on plausible scales would resolve this error, though they would not yet validate the proposed weight edit.

```python
if r49_centres.shape[-1] == 16:
    r49_centres = torch.cat(
        [torch.zeros((r49_centres.shape[0], 1), dtype=r49_centres.dtype),
         r49_centres], dim=1)
print("Corrected centres:", tuple(r49_centres.shape))
r54_base = r49_readouts(r49_out0, r49_h0, np.arange(len(r49_out0)))
print("Readout shape:", tuple(r54_base.shape))
print("Readout mean:", r54_base.mean(0).tolist())
print("Readout range:", r54_base.min(0).values.tolist(),
      r54_base.max(0).values.tolist())
print("Sensors:", r49_sensors)

print("\n=== CORE HELPER DEPENDENCIES ===")
print("r51_core locals:", r51_core.__code__.co_varnames)
print("r51_core globals:", r51_core.__code__.co_names)
print("r51_core constants:", str(r51_core.__code__.co_consts)[:1300])
print("\n=== RETAINED FUNCTIONS ===")
funcs54 = [(k, str(inspect.signature(v))) for k, v in list(globals().items())
           if callable(v) and getattr(v, "__module__", "") == "__main__"
           and not k.startswith("_")]
print(funcs54)

print("\n=== WORKING ARRAYS ===")
arr54 = [(k, str(getattr(v, "shape", ""))) for k, v in list(globals().items())
         if hasattr(v, "shape") and
         (k.startswith(("rr", "r47", "r48", "r49", "r50", "r51"))
          or k in ("X", "Y", "panel", "df", "train"))]
print(str(arr54)[:2400])

print("\n=== EARLY EVIDENCE ===")
lines54 = Path("/work/notes.md").read_text().splitlines()
sel54 = [s for s in lines54 if any(t in s.lower() for t in
         ("impulse", "nll", "causal", "effect", "rollout", "one-hour", "target shift"))]
print("\n".join(s[:300] for s in sel54[:16]))
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 54: bin-table correction\n"
            "- Added the absent-value centre to the 16 fine-bin centres; "
            "testing local readouts before weight surgery. No weights edited.\n"
            "- Python runs: 54; forward passes: 4.\n")
```
