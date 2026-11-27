The reconstruction bias is concentrated in readings with an earlier value. Also, the actual last-reading projection has eight inputs, so the architecture’s short description does not fully specify their layout. I’ll test plausible value/presence layouts without editing weights. A large drop toward the GPU sampling-noise floor would identify a useful correction; no improvement would rule out this hypothesis.

```python
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

torch.set_num_threads(8)
with torch.no_grad():
    old_numbers93 = head_recovery["numbers"].float()
    old_inputs93 = head_recovery["inputs"].float()
    numeric_W93 = wt("last_numeric.weight")
    numeric_b93 = wt("last_numeric.bias")
    fixed_inputs93 = old_inputs93 - F.linear(old_numbers93, numeric_W93, numeric_b93)
    has93 = torch.as_tensor(head_meta.has_last.to_numpy(), dtype=torch.float32)
    age93 = torch.as_tensor(np.log1p(head_meta.age_s.to_numpy())/10, dtype=torch.float32)
    value93 = torch.as_tensor(head_meta.zlast.to_numpy(), dtype=torch.float32)
    codes93 = torch.as_tensor(head_meta.code.to_numpy(), dtype=torch.long)
    bins93 = torch.as_tensor(head_meta.last_bin.to_numpy(), dtype=torch.long)
    transition93 = wt("embed_transition.value_bin.weight")[codes93*17+bins93]
    centres93 = torch.as_tensor(repair_C17, dtype=torch.float32)
    target93 = head_meta.original_mean.to_numpy()
    scales93 = head_meta.scale.to_numpy()
    target_bins93 = torch.as_tensor(repair_fbins, dtype=torch.long)
    layouts93 = [(2, (), "no_extra_presence")]
    for value_col93 in range(2, 8):
        for presence_col93 in range(2, 8):
            if presence_col93 != value_col93:
                layouts93.append((value_col93, (presence_col93,),
                                  f"value{value_col93}_presence{presence_col93}"))
    layouts93 += [(2, (3, 5), "value2_presence3_5"),
                  (2, (5, 7), "value2_presence5_7")]
    layout_results93 = []
    layout_features93 = {}
    for value_col93, presence_cols93, label93 in layouts93:
        numbers93 = torch.zeros_like(old_numbers93)
        numbers93[:, 0] = has93
        numbers93[:, 1] = age93
        numbers93[:, value_col93] = value93
        for presence_col93 in presence_cols93:
            numbers93[:, presence_col93] = has93
        inputs93 = fixed_inputs93 + F.linear(numbers93, numeric_W93, numeric_b93)
        hidden93 = F.gelu(F.linear(inputs93, wt("attr_heads.value_bin.0.weight"),
                                  wt("attr_heads.value_bin.0.bias")))
        logits93 = F.linear(hidden93, wt("attr_heads.value_bin.2.weight"),
                            wt("attr_heads.value_bin.2.bias")) + transition93
        probs93 = logits93.softmax(-1)
        means93 = (probs93*centres93).sum(-1).numpy()
        errors93 = (means93-target93)/scales93
        nll93 = -probs93[torch.arange(len(probs93)), target_bins93].clamp_min(1e-30).log()
        row93 = {"layout": label93, "scaled_rmse": np.sqrt(np.mean(errors93**2)),
                 "scaled_bias": errors93.mean(), "nll": float(nll93.mean())}
        for sensor93 in ("glucose_finger", "glucose_lab", "map_arterial", "map_cuff"):
            row93[sensor93+"_bias"] = errors93[head_meta.event.eq(sensor93).to_numpy()].mean()
        layout_results93.append(row93)
    layout_table93 = pd.DataFrame(layout_results93).sort_values("scaled_rmse")
    print(layout_table93.head(14).to_string(index=False))
    print("\nCURRENT LAYOUT:")
    print(layout_table93[layout_table93.layout.eq("value2_presence3")].to_string(index=False))

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 90–93: head equivalence audit\n"
            "- Stored rank-one coefficients and intercept are converted correctly; "
            "the earlier standardized-feature comparison was invalid.\n"
            "- Original reconstructed NLL is 1.71733 versus GPU 1.64546 on the same "
            "training readings, with sensor-dependent bias mostly after prior readings.\n"
            "- Testing the eight-input last_numeric layout before fitting further edits.\n")
print("ACTIVE EDITS:", edited())
```