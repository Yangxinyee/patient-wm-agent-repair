The cached contrasts include paired latent samples, so we can constrain factual behavior over the same variation that exposed the previous failures. Source inspection failed, but the needed tensors are available. I’ll check the record-derived effect and persistence estimates next. Stable estimates would justify specific targets; uncertain persistence would limit what we can claim about full rollouts.

```python
print("=== RECORD-DERIVED RESPONSE ESTIMATES ===")
for filename in ["dynamic_iv.csv", "sensor_iv.csv", "kernel_targets.csv", "local_projections.csv"]:
    df = pd.read_csv("/work/" + filename)
    print("\n" + filename + " " + str(df.shape))
    print(df.to_string(index=False))

print("\n=== RELEVANT NOTEBOOK SECTIONS ===")
sections150 = notes_text.split("\n## ")
selected150 = []
for section in sections150:
    title = section.splitlines()[0]
    if any(s in title.lower() for s in ["instrument", "identif", "kernel", "sensor", "causal", "projection", "proxy"]):
        selected150.append(section)
print(("\n## ".join(selected150))[:4200])

print("\n=== CURRENT TRAIN FEATURES ===")
print(rebuilt_train145.head(3).to_string(index=False))
print("Event counts:", rebuilt_train145.event.value_counts().to_dict())
print("Effects:", {k: v for k, v in list(globals().items())
                   if "coef" in k.lower() and isinstance(v, dict)})
print("Factual MC NLL:", wm_train145["nll"].mean().item(),
      wm_tune145["nll"].mean().item())
print("Original final-layer shapes:", wm_weights145["attr_heads.value_bin.2.weight"].shape,
      wm_weights145["attr_heads.value_bin.2.bias"].shape)
print("Active edits:", edited())
```
