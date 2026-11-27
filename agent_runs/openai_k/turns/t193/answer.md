The saved edit passes the factual threshold on 99 tuning stays, but its rollout remains too weak: about −2.75 for doubled insulin and +0.21 for doubled norepinephrine. The record-based estimates suggest missing drug responses. I’ll recover the retained head reconstruction and test a small, factual-penalized output correction rather than enlarge the saturating recurrent edit.

```python
import dis, re
from pathlib import Path

print("FACTUAL TABLE")
print(r172_fact.columns.tolist())
print(r172_fact.head(2).to_string(index=False)[:1000])

print("CAPTURED LOCAL UPDATE")
if "localvars" in globals():
    for k, v in localvars.items():
        print(k, r175_desc(v))

print("HEAD RECONSTRUCTION: COMPACT BYTECODE")
r177_ops = []
for ins in dis.get_instructions(assay_head110):
    if ins.opname not in ("CACHE", "RESUME", "NOP", "EXTENDED_ARG"):
        r177_ops.append(ins.opname + (":" + ins.argrepr if ins.argrepr else ""))
for i in range(0, len(r177_ops), 6):
    print(" | ".join(r177_ops[i:i+6]))

print("RETAINED HEAD ARRAYS")
for name in ("dt104", "ctx104", "zh104", "ix104", "has104", "age105",
             "zval104", "codeemb104", "binemb104", "trans104", "label104",
             "centres104", "r49_register", "r49_hours", "r49_sensors",
             "r49_centres", "sx", "rr33_stays", "rr33_hours", "beta59",
             "readoutweights151", "models145"):
    if name in globals():
        print(name, r175_desc(globals()[name]))

n = Path("/work/notes.md").read_text()
print("HEAD VALIDATION NOTES")
for tag in ("Run 104", "Run 105", "Run 108", "Run 110", "Run 111",
            "Run 112", "Run 113", "Run 114", "Run 115", "Run 116"):
    pos = n.find(tag)
    if pos >= 0:
        end = n.find("\n## ", pos)
        text = n[pos:end if end >= 0 else pos+450]
        print(text[:450])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 177: recover exact hidden-head inputs\n"
            "- Official saved-candidate factual delta is +0.002286; rollout effects "
            "remain insulin x2 -2.750 and NE x2 +0.210, too small for record estimates.\n"
            "- Recovering retained head reconstruction for a small final-layer "
            "correction with an explicit factual penalty.\n"
            "- Python runs: 177; forward passes: 10. No weights changed.\n")
```
