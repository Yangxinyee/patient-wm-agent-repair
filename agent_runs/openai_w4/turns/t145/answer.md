Scale 0.4 slightly improved training likelihood and remained bounded; scale 0.6 crossed into rapid amplification. I’ll save 0.4 as a provisional candidate and recover the longer-horizon evidence before testing its rollouts. The rollout must show a meaningful, bounded dose response—not merely good factual fit.

```python
scale130 = 0.4
candidate130 = dict(weights_channel125)
candidate130["core.b_in.weight"] = (
    weights_channel125["core.b_in.weight"] + scale130 * connection128
)
save_edit("core.b_in.weight", candidate130["core.b_in.weight"])
print("Saved response scale:", scale130)
print("Edited tensors:", edited())

print("\n--- Sustained-effect notebook sections ---")
import re
notes130 = open("/work/notes.md").read()
sections130 = re.split(r"(?=^## Run |^Runs |^Run )", notes130, flags=re.M)
selected130 = []
for section130 in sections130:
    if re.search(r"distributed|six.lag|two.pole|long.horizon|cumulative|sustained|impulse|kernel", section130, re.I):
        selected130.append(section130[:850])
print("\n".join(selected130)[-5500:])

print("\n--- Retained causal objects ---")
count130 = 0
for name130, obj130 in list(globals().items()):
    if any(s in name130.lower() for s in ["kernel", "lag", "dml", "policy", "panel"]):
        if callable(obj130):
            print(name130, inspect.signature(obj130))
        elif isinstance(obj130, dict):
            print(name130, list(obj130.keys())[:8])
        elif hasattr(obj130, "shape"):
            print(name130, obj130.shape)
        else:
            continue
        count130 += 1
        if count130 >= 35:
            break
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 130\nSaved provisional core.b_in.weight response at scale 0.4. Complete training replay dNLL=-0.000229 and out RMSE=0.00561; scale 0.6 is unstable. This is not yet rollout validation or proof of sustained causal accuracy. Ten tensors now edited.\n")
```