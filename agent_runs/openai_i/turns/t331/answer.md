The full route fails even with no added drug response: its factual loss increase is 0.01055 nats. I’ll remove the routing changes and retest the earlier direct-token edit with the corrected empty-action gate. Its earlier failure may have involved spurious signals on drug-free bundles; lower loss now would support that explanation.

```python
with np.load("/work/token257.npz", allow_pickle=False) as file286:
    token286 = {k: file286[k].copy() for k in file286.files if k in NAMES}
assert "z_in.weight" in token286
direct_delta286 = token286["z_in.weight"] - neutral284["z_in.weight"]
print("Direct-token delta norm", float(np.linalg.norm(direct_delta286)),
      "active columns",
      np.flatnonzero(np.linalg.norm(direct_delta286, axis=0) > 1e-7).tolist())

direct286 = {k: v.copy() for k, v in neutral284.items()}
for name286, value286 in corrected284.items():
    if name286.startswith("z_post."):
        direct286[name286] = value286.copy()
direct286["z_in.weight"] = (
    neutral284["z_in.weight"] + direct_delta286
).astype(np.float32)
if "z_in.bias" in token286:
    direct286["z_in.bias"] = token286["z_in.bias"].copy()

assert not any(k.startswith("core.") for k in direct286)
install_checkpoint285(direct286)
np.savez("/work/corrected_direct286.npz", **direct286)
print("Installed corrected direct-token candidate, saved gain 0.3.")
print("Current edits", edited())

print("\nPRIVATE CHANNEL / ROUTING HELPERS")
import inspect
for helper286 in ["replay_action_token216", "replay_route228"]:
    func286 = globals().get(helper286)
    if callable(func286):
        try:
            print(helper286, inspect.getsource(func286)[:2400])
        except (OSError, TypeError):
            print(helper286, "source unavailable")

with open("/work/notes.md", "a") as log286:
    log286.write(
        "\n## Python 286 — reject full routing overhead; correct direct-token test\n"
        "- Forward 32 / fwd_33 zero-response full route dNLL +0.01055 on second cohort; rejected.\n"
        "- Installed corrected_direct286: token257 gain 0.3 plus corrected269 empty-action posterior gates.\n"
        "- All prelude attention and FFN weights restored to original.\n"
        "- Prior direct-token failure preceded discovery of the nonzero empty-action encoding.\n"
        "- Hypothesis: fixing that leakage may permit direct causal transport without routing overhead.\n"
        "- Candidate remains provisional; requires matched factual and dose validation.\n"
    )
```