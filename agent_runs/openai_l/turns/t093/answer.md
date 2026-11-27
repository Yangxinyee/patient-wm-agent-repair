The retained objects show that the previous fit used factual and dose-altered decoder features. Its loss trade-off suggests that this head-only approach may be insufficient. I’ll inspect the objective, earlier causal evidence, and baseline rollouts before choosing another edit. If the targets or feature alignment are wrong, I’ll correct those first rather than alter the recurrence.

```python
import inspect, dis, io, json
from pathlib import Path

for name in ["make_cf81", "target_logits83", "evaluate83", "decode67", "transition64"]:
    fn = globals()[name]
    print("\n=== FUNCTION", name, inspect.signature(fn), "===")
    try:
        print(inspect.getsource(fn))
    except Exception:
        print("names:", fn.__code__.co_names)
        print("constants:", repr(fn.__code__.co_consts)[:2500])
        if name in ["target_logits83", "evaluate83"]:
            stream = io.StringIO()
            dis.dis(fn, file=stream)
            print(stream.getvalue()[:14000])

print("\n=== RELEVANT WEIGHT SHAPES ===")
for name in NAMES:
    if any(s in name for s in ("z_post", "z_prior", "z_in", "z_head", "attr_heads.value", "encode_action")):
        print(name, SHAPES[name])

print("\n=== EARLY CAUSAL NOTES ===")
notes85 = Path("/work/notes.md").read_text()
sections85 = re.split(r"(?=^## )", notes85, flags=re.M)
for section in sections85:
    if re.match(r"## (Run|Runs) (1[3-9]|2[0-4]|8[1-3])\b", section):
        print(section[:2200])

print("\n=== EXISTING FORWARD FILES ===")
for directory in sorted(Path("/work").glob("fwd_*")):
    print(directory.name, [p.name for p in directory.iterdir()])
    for p in directory.glob("dose_edit*.csv"):
        table85 = pd.read_csv(p)
        print(p.name, table85.shape, table85.columns.tolist())
        print(table85.head(8).to_string(index=False))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 85 / inspect head objective and causal evidence\n"
            "Inspecting the existing fit, original dose-rollout output, and latent-update interfaces "
            "to distinguish a weak head-only parameterization from a target/alignment error. No edits.\n")
```