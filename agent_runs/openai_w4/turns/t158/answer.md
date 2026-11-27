Scale 0.5 remained close to the original factual likelihood in training replay; scale 0.6 clearly deteriorated. I’ll test 0.5 on the GPU before attempting a larger redesign. Stronger insulin responses with factual loss below the limit would support this adjustment; poor likelihood or erratic responses would reject it.

```python
import pandas as pd

decoder141 = torch.as_tensor(chosen_channel116["decoder"], dtype=torch.float32)
direction141 = torch.as_tensor(directions118[0.01], dtype=torch.float32)
base_b141 = torch.as_tensor(load_original("core.b_in.weight"), dtype=torch.float32)
outer141 = torch.outer(direction141, decoder141)
current_b141 = torch.as_tensor(load("core.b_in.weight"), dtype=torch.float32)

print("Current connection versus documented scale 0.4:",
      float(torch.norm(current_b141 - (base_b141 + 0.4 * outer141))))
print("Training replay original NLL:", float(np.mean(nll_original129)))
for scale141 in [0.4, 0.5, 0.6]:
    print("scale", scale141, "replay dNLL:",
          float(np.mean(nll_scales129[scale141]) - np.mean(nll_original129)))

backup_b141 = current_b141.clone()
assert torch.allclose(current_b141, base_b141 + 0.4 * outer141,
                      atol=2e-6, rtol=2e-5), "Recover connection recipe before editing"
save_edit("core.b_in.weight", (base_b141 + 0.5 * outer141).numpy())

tuning_ids141 = pd.read_csv("/data/tuning.csv", usecols=["stay_id"])["stay_id"].drop_duplicates().tolist()
print("First 100 tuning stay IDs:", tuning_ids141[:100])
print("Edited tensors:", edited())

with open("/work/notes.md", "a") as f141:
    f141.write(
        "\n## Run 141\n"
        "Scale 0.5 training replay is stable enough to test: mean NLL "
        f"{float(np.mean(nll_scales129[0.5])):.7f}, original "
        f"{float(np.mean(nll_original129)):.7f}. Saved B_in response scale 0.5; "
        "scale 0.4 backed up in backup_b141. Awaiting independent GPU factual "
        "and rollout validation; no causal claim from replay alone.\n"
    )
```