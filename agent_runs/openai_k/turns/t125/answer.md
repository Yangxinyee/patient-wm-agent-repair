The retained candidates had weak insulin responses, and stronger norepinephrine corrections also moved glucose. None has been externally validated. I’ll inspect the exact head inputs and mismatch pattern now; if a missing register field explains the gap, correcting it should improve all four sensor comparisons.

```python
print("=== COMPACT READOUT BYTECODE: ATTRIBUTE INPUTS ===")
ins117 = list(dis.get_instructions(r49_readouts))
start117 = next((i for i,x in enumerate(ins117)
                 if x.argval == "r49_register"), 0)
print("\n".join(f"{x.opname} {x.argrepr}" for x in ins117[max(0,start117-12):])[:5000])
print("\n=== ASSAY INTERFACES AND CONSTANTS ===")
for name117 in ("assay_head110","bin_assay112"):
    f117 = globals()[name117]
    print(name117, inspect.signature(f117))
    print("Globals:", f117.__code__.co_names)
    print("Constants:", [x for x in f117.__code__.co_consts
                         if not isinstance(x,types.CodeType)][:30])
print("\n=== NUMERIC REGISTER WEIGHTS ===")
print("column norms:", torch.linalg.vector_norm(W0["last_numeric.weight"],dim=0).tolist())
print("numeric specs:", {str(k):v for k,v in num_spec0.items() if "glucose" in str(k) or "arterial" in str(k)})
print("\n=== SAVED MISMATCH EXAMPLES ===")
print(comparison110.head(8).to_string(index=False))
print("Comparison columns:", comparison110.columns.tolist())
print("No-value mass summary:", mass111.describe().round(4).to_string())
print("\n=== COMPLETE TOKEN FRONTIER ===")
for d117 in diagnostics90:
    print(d117)
print("\n=== RELEVANT VARIABLE INVENTORY ===")
print([(k, getattr(v,"shape",type(v).__name__)) for k,v in list(globals().items())
       if re.match(r"(r49_|.*10[4-9]|.*110|.*112|.*113)$",k)][:65])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 117: head-input and route inspection\n"
            "- Retained token candidates do not yet deliver drug-specific target responses; none externally tested.\n"
            "- Inspecting register preprocessing and saved mismatch examples before selecting the next route.\n"
            "- No edits. Python runs: 117; forward passes: 4.\n")
```