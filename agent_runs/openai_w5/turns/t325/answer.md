The factual gates are in their linear range, and the logit tilts are monotone. Gate clipping therefore does not explain the asymmetry. I’ll try a reversible 3% softening of the value distribution, leaving dose transport unchanged. It is useful only if factual NLL improves without materially weakening the dose responses.

```python
from safetensors.torch import save_file

BASE282 = {name: torch.as_tensor(load(name)).clone() for name in edited()}
save_file({name: value.contiguous() for name, value in BASE282.items()},
          "/work/checkpoint_run282.safetensors")

temperature_names282 = [
    "attr_heads.value_bin.2.weight",
    "attr_heads.value_bin.2.bias",
] + [name for name in NAMES if name.startswith("embed_transition.value_bin")]
assert len(temperature_names282) == 3, temperature_names282
temperature_base282 = {
    name: torch.as_tensor(load(name)).clone() for name in temperature_names282
}
logit_scale282 = 0.97
for name, value in temperature_base282.items():
    save_edit(name, value * logit_scale282)

print("Installed reversible value-logit scale:", logit_scale282)
print("Scaled tensors:", temperature_names282)
print("Dose transport and first-layer sensor corrections unchanged.")
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 282: reversible value-distribution calibration\n"
            "- Saved all pre-calibration edits in checkpoint_run282.safetensors.\n"
            "- Scaled all three value-logit components by 0.97, including transition logits.\n"
            "- Drug transport, persistence, and first-layer centering unchanged.\n"
            "- Motivation: glucose-finger NLL worsens despite improved aggregate mean; "
            "distributional calibration is a hypothesis, not an established cause.\n"
            "- Requires paired GPU factual validation and subsequent finite rollouts; "
            "not an accepted candidate.\n")
```
