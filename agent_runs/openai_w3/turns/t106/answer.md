The records support insulin lowering glucose and norepinephrine raising MAP, with different persistence: the estimated tuning readouts are roughly −13/+6.5 glucose units and +4.3/−2.1 MAP units. I’ll now measure the recurrent pathway’s sensitivity so the edit writes a physiological change, not merely a dose score. Forward validation will determine whether it achieves those shifts without harming factual fit.

```python
print("FUNCTION NAMES")
print(" ".join(n for n in sorted(list(globals())) if inspect.isfunction(globals()[n])))

print("\nADDITIONAL IDENTIFICATION/REPAIR NOTES")
text95 = Path("/work/notes.md").read_text()
print(text95[3800:7900])

print("\nMEMORY INTERFACE", inspect.signature(memory82))
print(compact_bytecode(memory82)[:1800])
print("\nPRELUDE OUTPUT EXPRESSIONS")
print(compact_bytecode(prelude_cache88)[-2200:])

print("\nFIT CONTAINERS")
for key95 in (1.0, (1.0, 1e-8)):
    o95 = contrast_fits88[key95]
    print("contrast", key95, type(o95).__name__)
    if isinstance(o95, dict):
        print({k: (getattr(v, "shape", None) or repr(v)[:100]) for k, v in o95.items()})
    elif isinstance(o95, (tuple, list)):
        print([(type(v).__name__, getattr(v, "shape", None)) for v in o95])
print("HEAD TENSORS",
      [(n, SHAPES[n]) for n in NAMES
       if n.startswith(("last_", "embed.categorical", "z_head", "type_emb"))])
print("READ CODES", read_codes90)
print("LAST REGISTER ROWS", last_values90[idx81[:5]], last_hours90[idx81[:5]])
print("STATE SOURCES", [(n, globals()[n]) for n in ("sfile", "state_path", "states_path") if n in globals()])

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 95; forwards 7. Recovered record-derived immediate and impulse effects "
            "(insulin ~-1.94/unit, NE ~+3.2/rate); tuning rollout references -12.96/+6.48 and "
            "+4.28/-2.14. Preparing mechanistic finite-difference readout calibration, no gradients.\n")
```