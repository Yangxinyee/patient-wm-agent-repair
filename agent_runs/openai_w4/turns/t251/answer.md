With context fixed, insulin transmits half and double doses similarly: slopes 0.859 and 0.875. Norepinephrine’s doubled-dose transmission is stronger, not weaker. That makes channel compression unlikely to be the main cause of the rollout asymmetry. I’ll inspect when the response changes sign and recover the impulse-test helpers; reversal with future readings held factual would implicate hidden-state dynamics rather than generated-reading feedback.

```python
import dis

print("=== RECORD-BASED NONLINEARITY AND READOUT ESTIMATES ===")
print("quadratic137:", quadratic137)
print("readout_estimates137:", readout_estimates137)

print("\n=== IMPULSE / RESPONSE HELPERS ===")
for r218_name, r218_obj in sorted(list(globals().items())):
    if r218_name.startswith("r218_"):
        continue
    if callable(r218_obj) and hasattr(r218_obj, "__code__") and re.search(
        r"impulse|perturb|response|replay|mean.*head|head.*mean", r218_name
    ):
        print(r218_name, str(inspect.signature(r218_obj))[:180],
              "globals:", r218_obj.__code__.co_names)
print("retained impulse objects:",
      [n for n in globals() if "impulse" in n and not n.startswith("r218_")])

print("\n=== FACTUAL-HEAD EVALUATOR BYTECODE ===")
print(dis.Bytecode(cpu_factual129).dis()[:4200])
print("Head local variables:", head_features_fixed.__code__.co_varnames)
print("Head return bytecode:")
for r218_instruction in list(dis.get_instructions(head_features_fixed))[-15:]:
    print(r218_instruction.opname, r218_instruction.argrepr)

print("\n=== LATEST FORWARD FILES ===")
r218_files = sorted(Path("/work/fwd_21").glob("*"))
print([p.name for p in r218_files])
for r218_path in r218_files:
    if "dose_edit" in r218_path.name and r218_path.suffix == ".csv":
        r218_df = pd.read_csv(r218_path)
        print(r218_path.name, "columns:", list(r218_df.columns))
        print(r218_df.head(2).to_string(index=False))

with open("/work/notes.md", "a") as r218_file:
    r218_file.write(
        "\n## Run 218\n"
        "Run-217 fixed-context transmission slopes: insulin half .85884/double .87482; "
        "NE half .87314/double 1.01000. Similar insulin slopes and stronger NE doubling transmission "
        "do not explain the free-rollout asymmetry. Recovering impulse tests and temporal rollout profiles "
        "to distinguish hidden-state dynamics from generated-reading feedback. No saved weights changed.\n"
    )
```