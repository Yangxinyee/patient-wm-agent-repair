The transition uses custom four-head attention blocks, and the action helper’s embedding choices are now known. To reconstruct memory safely, I also need to validate how measurement bundles are encoded. I’ll inspect the record and normalization fields next; posterior agreement on measurement bundles would confirm those choices without relying on medical assumptions.

```python
print("=== RECORDS AND SAVED STATES ===")
print("Record columns:", list(rm_records.columns))
print(rm_records.head(6).to_string(index=False))
for r102_key, r102_value in rm_states.items():
    if isinstance(r102_value, (np.ndarray, torch.Tensor)):
        print(r102_key, tuple(r102_value.shape), str(r102_value.dtype))
print("First saved hours:", np.asarray(rm_states["hour"])[:12])
print("Pure-action indices:", np.asarray(rm_pure_idx)[:12])

print("\n=== VOCABULARY AND NORMALIZATION ===")
print("VOCAB keys:", list(VOCAB))
print("Codes:", VOCAB["codes"])
print("Normalization fields:")
for r102_key, r102_value in rm_normspec.items():
    print(r102_key, r102_value)

print("\n=== ACTION HELPER: NUMERIC/BIN LOGIC ===")
r102_buffer = io.StringIO()
dis.dis(rm_insulin_atoms, file=r102_buffer)
r102_disassembly = r102_buffer.getvalue()
r102_start = r102_disassembly.find("STORE_FAST               8 (attr)")
print(r102_disassembly[max(0, r102_start):max(0, r102_start) + 2600])

print("\n=== CAUSAL IDENTIFICATION NOTE SUMMARIES ===")
for r102_section in r100_selected[:8]:
    r102_lines = r102_section.splitlines()
    print("\n".join(r102_lines[:5])[:650])

with Path("/work/notes.md").open("a") as r102_f:
    r102_f.write(
        "\n## World-embedding preparation; Python 102, forward 11/50\n"
        "- Transition is custom pre-norm attention with four heads.\n"
        "- Inspecting record fields and robust-score conventions to reconstruct memory from records.\n"
        "- Measurement-bundle posterior validation will test encoding choices; no weight changes.\n"
    )
```
