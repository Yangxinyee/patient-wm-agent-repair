The head features decode dose innovations only weakly on held-out stays, especially for the alternate measurement channels. That makes a direct head-only repair risky. I’ll inspect the replay inputs and causal-response kernels instead; a successful replay should match exported states across both drug and reading bundles.

```python
print("=== REPLAY VARIABLES ===")
resume_lines = []
for resume_name, resume_obj in sorted(list(globals().items())):
    if not resume_name.startswith("rx_"):
        continue
    if isinstance(resume_obj, (np.ndarray, torch.Tensor)):
        resume_lines.append(f"{resume_name}: {tuple(resume_obj.shape)}")
    elif isinstance(resume_obj, dict):
        resume_lines.append(f"{resume_name}: dict {len(resume_obj)} {str(list(resume_obj)[:5])[:120]}")
    elif isinstance(resume_obj, (list, tuple)):
        resume_lines.append(f"{resume_name}: n={len(resume_obj)} {str(resume_obj[:2])[:180]}")
    elif isinstance(resume_obj, (int, float, str)):
        resume_lines.append(f"{resume_name}: {resume_obj}")
print("\n".join(resume_lines)[:2000])

print("=== DATA / CAUSAL OBJECTS ===")
for resume_name, resume_obj in sorted(list(globals().items())):
    if isinstance(resume_obj, pd.DataFrame) and len(resume_obj) > 100000:
        print(resume_name, resume_obj.shape, list(resume_obj.columns))
print("local projection keys:", list(repair_lp))
for resume_key, resume_obj in repair_lp.items():
    print(resume_key, str(resume_obj)[:700])

print("=== KERNEL NOTE EXCERPTS ===")
import re, dis
resume_matches = list(re.finditer(r"(?i)deconvol|response kernel", resume_notes))
for resume_match in resume_matches[:5]:
    print(resume_notes[max(0, resume_match.start()-100):resume_match.start()+550])

print("=== LAYER IMPLEMENTATION ===")
resume_dis = "\n".join(
    f"{i.offset}: {i.opname} {i.argrepr}" for i in dis.get_instructions(rx_layer2)
)
print(resume_dis[:2400])
print("=== ATTENTION IMPLEMENTATION ===")
resume_dis = "\n".join(
    f"{i.offset}: {i.opname} {i.argrepr}" for i in dis.get_instructions(rx_attn)
)
print(resume_dis[:2000])

with open("/work/notes.md", "a") as resume_f:
    resume_f.write(
        "\n## Run 138\n"
        "Held-out innovation decoding is weak, particularly through value-head "
        "hidden features and alternate channels. Not treating this as a "
        "validated head-only repair. Inspecting replay implementation and "
        "dynamic causal estimates. No edits. Counts: 138 Python runs, 7 forwards.\n"
    )
```