The records support next-hour effects near −1.95 for insulin and +3.1 for norepinephrine, with different decay patterns. I’ll reconstruct the value-head inputs and test dose factors from zero to three, so a repair need not depend only on doubling. Exact agreement with cached activations is the prerequisite.

```python
def head_inputs96(ctx, const, eps):
    prior = mlp63(ctx, "z_prior")
    z = prior[None, :, :32] + eps * prior[None, :, 32:].exp()
    view = ctx[None] + linear63(z, "z_head")
    q = norm63(view + w63("type_emb.weight")[1], "query_norm")
    return q + const[None]

input_fact96 = head_inputs96(ctx_fact80, const_train67, epsread73)
recovered_hidden96 = F.gelu(
    linear63(input_fact96, "attr_heads.value_bin.0")
)
input_error96 = float(
    ((recovered_hidden96 - hidden_fact80)**2).mean().sqrt()
)
print("Head activation reconstruction RMS:", input_error96)
assert input_error96 < 1e-5

decoder_backup96 = decode67
last_input96 = None

def decoder_capture96(ctx, const, trans, centres, eps,
                      return_features=False):
    global last_input96
    last_input96 = head_inputs96(ctx, const, eps)
    return decoder_backup96(ctx, const, trans, centres, eps,
                            return_features=return_features)

cf_inputs96 = {}
summary96 = {}
try:
    decode67 = decoder_capture96
    for drug in ("insulin", "norepinephrine"):
        for factor in (0.0, 0.5, 1.5, 2.0, 3.0):
            case = make_cf81(drug, factor)
            case["input"] = last_input96
            cf_inputs96[(drug, factor)] = case
            rows = case["rows"]
            relevant = np.array([
                str(event).startswith(
                    "glucose" if drug == "insulin" else "map"
                ) for event in row_events94[rows]
            ])
            delta_rate = case["delta_rate"][relevant]
            shift = (
                case["means"].mean(0) -
                means_fact80[:, rows].mean(0)
            ).numpy()[relevant]
            slope = float(np.dot(delta_rate, shift) /
                          max(1e-12, np.dot(delta_rate, delta_rate)))
            summary96[(drug, factor)] = slope
            print(drug, factor, "original next-reading slope:", slope,
                  "input shape:", tuple(last_input96.shape))
finally:
    decode67 = decoder_backup96

print("\n=== PREVIOUS LOGIT-TARGET FUNCTION ===")
show91(target_logits83, 2500)

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 96 / value-head input reconstruction\n"
            f"Hidden reconstruction RMS={input_error96:.6g}. "
            "Generated validated one-step head inputs for both drugs "
            "at factors 0,.5,1.5,2,3. Original dose slopes: "
            f"{summary96}. No tensors edited.\n")
```