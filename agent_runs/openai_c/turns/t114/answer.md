This repeats the previous run’s output, so it adds no new evidence. I’ll keep the restored edit unchanged and request a shorter history extract. The next candidate should test a pathway that earlier factual and rollout checks have not already ruled out.

```python
import inspect, re

r114_text = open("/work/notes.md").read().split("## Restart / Python runs 49")[0]
r114_compact = []
for r114_line in r114_text.splitlines():
    if r114_line.startswith("##"):
        r114_compact.append(r114_line)
    elif re.search(
        r"gain|scal|shear|ablat|rollout|NLL|bin contrast|rejected|installed|restored",
        r114_line, re.I
    ):
        r114_compact.append(r114_line[:180])
r114_compact = "\n".join(r114_compact)
print(r114_compact[:6500])
print("\nCOMPACT HISTORY LENGTH:", len(r114_compact))

print("\nHELPERS:")
for r114_name in (
    "apply_lowrank_left_transform", "apply_lowrank_right_transform",
    "freeze_ops", "replay_ops", "effective_transform"
):
    print(r114_name, inspect.signature(globals()[r114_name]))

print("\nSAVED OP LIST NAMES:",
      [r114_name for r114_name in list(globals())
       if "ops" in r114_name and not r114_name.startswith("_")])

print("\nBIN AND ACTION TENSORS:")
for r114_name in NAMES:
    if r114_name.startswith(("embed.bins.", "encode_action.")):
        print(r114_name, SHAPES[r114_name])
```