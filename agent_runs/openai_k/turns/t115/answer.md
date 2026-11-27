The tensor shapes reveal a discrepancy: `last_numeric` takes eight inputs, while the architecture describes four. The output was truncated before the relevant code appeared. I’ll retrieve just that code and the identification summary, so the next test addresses a specific discrepancy.

```python
from collections import defaultdict

print("=== READOUT CONSTANTS ===")
print("variables:", r49_readouts.__code__.co_varnames)
print("constants:", r49_readouts.__code__.co_consts)

print("\n=== COMPACT READOUT BYTECODE ===")
grouped108 = defaultdict(list)
line108 = None
keep108 = {"LOAD_CONST", "LOAD_GLOBAL", "LOAD_FAST", "STORE_FAST",
           "LOAD_ATTR", "LOAD_METHOD", "BINARY_OP", "BINARY_SUBSCR",
           "BUILD_LIST", "BUILD_TUPLE", "BUILD_SLICE"}
for ins108 in dis.get_instructions(r49_readouts):
    if ins108.starts_line is not None:
        line108 = ins108.starts_line
    if ins108.opname in keep108:
        grouped108[line108].append(ins108.argrepr if ins108.opname != "BINARY_OP"
                                  else f"OP:{ins108.argrepr}")
for line108, tokens108 in grouped108.items():
    print(line108, " | ".join(tokens108))

print("\n=== REGISTER-TEST INPUTS ===")
for n108 in ("has104", "age105", "lastbin104", "codes104", "label104", "dt104"):
    o108 = globals().get(n108)
    if o108 is not None:
        print(n108, np.asarray(o108)[:10])
print("recent result variables:",
      [n for n in globals() if n.endswith("105") and
       isinstance(globals()[n], (list, dict))])

print("\n=== IDENTIFICATION SUMMARY ===")
text108 = Path("/work/notes.md").read_text()
print(text108[:2800])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 108: compact readout recovery\n"
            "- last_numeric has 8 actual inputs versus 4 documented fields; previous output was truncated.\n"
            "- Retrieving only readout bytecode and identification summary. No edits.\n"
            "- Python runs: 108; forward passes: 4.\n")
```