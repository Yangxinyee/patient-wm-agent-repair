The original rollout shifts are much smaller than the immediate effects estimated from the records. The failed regression used the posterior’s compressed hidden features; the raw context and action inputs may retain a cleaner dose signal. I will check those inputs and recover the lag estimates. A usable signal must survive held-out stays and have a consistent response to both dose increases and decreases.

```python
print("SIX-LAG RECORD KERNELS")
for drug, value in provisional_kernels.items():
    print(drug, np.asarray(value))
print("\nREPAIR FIXTURES")
for name in ["repair_bank", "repair_records", "idx82", "ctx82", "zp_input82",
             "zp_hidden82", "Z82", "labels82", "weights82", "cfdr82", "pair80",
             "state_banks_repair", "reading_banks_repair"]:
    value = globals()[name]
    if isinstance(value, dict):
        print(name, {str(k): (tuple(v.shape) if hasattr(v, "shape") else
                              list(v.keys())[:10] if isinstance(v, dict) else type(v).__name__)
                     for k, v in value.items()})
    elif isinstance(value, pd.DataFrame):
        print(name, value.shape, list(value.columns))
    else:
        print(name, tuple(value.shape), str(value.dtype))
print("\nPOSTERIOR FIT MASKS", int(fit82.sum()), int(test82.sum()))
print("\nFUNCTION RETURNS AND CALL FIELDS")
import dis
for name in ["transition73", "intervention_bank74"]:
    instructions = list(dis.get_instructions(globals()[name]))
    print(name, "last instructions:",
          [(x.opname, x.argrepr) for x in instructions[-22:]])
print("\nRECORD-BASED PERSISTENCE NOTES")
notes86 = Path("/work/notes.md").read_text()
begin86 = notes86.find("## Run 7")
print(notes86[begin86:begin86+4100])
```
