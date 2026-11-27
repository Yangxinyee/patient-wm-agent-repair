The recovered estimates suggest the original rollouts understate both drugs’ effects, but some estimators disagree substantially—especially for insulin dynamics and weak norepinephrine instruments. I’ll recover the recorded identification caveats and reconstruction diagnostics before choosing which estimates to trust.

```python
from pathlib import Path

text = Path("/work/notes.md").read_text()
print("NOTE HEADINGS")
for section in text.split("\n## ")[1:]:
    print(section.splitlines()[0])

print("\nEARLY IDENTIFICATION AND VALIDATION NOTES")
print(text[4300:11500])

print("\nCACHED RECONSTRUCTION DIAGNOSTICS")
for name in ("check_table", "integration_table", "decode_table",
             "core_check_table"):
    obj = globals().get(name)
    if obj is not None:
        print(name, obj.shape)
        print(obj.head(6).to_string(index=False))

print("\nVOCAB KEYS:", list(VOCAB))
print("CONFIG:", VOCAB.get("config"))
print("CODES:", VOCAB.get("codes"))
print("\nRELEVANT TENSOR SHAPES")
for name in NAMES:
    if any(term in name for term in
           ("attr_heads.value_bin", "z_prior", "z_head", "gap",
            "out_norm", "query_norm", "last_numeric",
            "last_status", "type_emb")):
        print(name, SHAPES[name])
```
