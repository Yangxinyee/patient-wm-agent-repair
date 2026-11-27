Earlier tests show that numeric amplification helps modestly, while several state and head edits add little. Full drug-identity removal narrowly failed factual fit, but its dose response has not been measured. I’ll reinstall it only as a diagnostic: improvement would justify testing partial removal; no improvement would rule out this pathway.

```python
clear_ops()
replay_ops(drug_identity_pilot77_ops)

print("DIAGNOSTIC OPERATIONS:",
      [{k: v for k, v in r115_op.items() if k != "vector"}
       for r115_op in ops()])

r115_notes = open("/work/notes.md").read()
r115_start = r115_notes.find("## Run 6:")
r115_end = r115_notes.find("## Runs 9", r115_start)
print("\nEARLIER BIN TEST:")
print(r115_notes[r115_start:r115_end][:1800])

with open("/work/notes.md", "a") as r115_f:
    r115_f.write(
        "\n## Python run 81: identity-path diagnostic\n"
        "- Reinstalled numeric gain 4 plus full insulin/NE code-row removal solely to measure dose sensitivity.\n"
        "- This strength is NOT acceptable for submission: forward 31 factual increase was +0.00658.\n"
        "- A material rollout improvement would justify partial removal; otherwise abandon this pathway.\n"
        "- Forward passes used: 31; 19 remain.\n"
    )
```