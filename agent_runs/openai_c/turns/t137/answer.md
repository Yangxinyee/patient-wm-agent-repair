Earlier numeric amplification saturated, and selective head/state edits gave little benefit. I’ll now test a different pathway: doubling the recurrent input projection while retaining numeric gain 4. This broad edit is diagnostic, not yet safe to submit. Stronger, correctly signed rollouts would justify a factual-fit test; weak or wrong-sign responses would reject it.

```python
scale("core.b_in.weight", 2.0)
print("RECURRENT INPUT GAIN CANDIDATE", ops())

print("\nREMAINING EARLIER PATHWAY RESULTS")
for section in re.split(r"(?=^## )", notes_text, flags=re.M):
    if any(label in section.split("\n", 1)[0] for label in [
        "Python run 34", "Python run 35", "Python run 38",
        "Python run 40", "Python run 41", "Python run 53",
        "Forward 23", "Python run 59", "Python runs 60",
        "Forward 25"
    ]):
        print(section[:850])

with open("/work/notes.md", "a") as f:
    f.write("\n## Python run 98: recurrent input-gain diagnostic\n"
            "- Numeric gain 4 retained; core.b_in.weight scaled by 2.\n"
            "- Tests whether stronger direct recurrent drive bypasses the weak action response.\n"
            "- Broad diagnostic: factual safety and rollout utility are unknown.\n"
            "- Forward passes used: 36; 14 remain.\n")
```
