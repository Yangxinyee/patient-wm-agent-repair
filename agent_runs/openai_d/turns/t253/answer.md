Removing the replaced neurons costs almost nothing, so that hypothesis was wrong: the added correction causes nearly all the factual loss. I’ll test subtracting its typical factual contribution within each reading type and previous-value bin. This preserves immediate dose sensitivity; lower loss on independent stays would support it, but a rollout check must show that it does not erase the treatment effect.

```python
r217_tname = "embed_transition.value_bin.weight"
r217_T0 = torch.as_tensor(load_original(r217_tname)).float()
r217_cache_rows = {}
for split, cache in [("train", c210_train), ("tune100", c210_tune)]:
    meta = cache["meta"]
    r217_cache_rows[split] = torch.tensor(
        [r201_codes[e] * 17 + int(b)
         for e, b in zip(meta["event"], meta["last_bin"])], dtype=torch.long)

def r217_eval(weights, cache, table_delta, split):
    logits = c211_logits(weights, cache) + table_delta[r217_cache_rows[split]][None]
    probs = torch.softmax(logits, dim=-1).mean(dim=0)
    loss = -torch.log(probs[torch.arange(len(cache["y"])), cache["y"]].clamp_min(1e-30))
    means = (probs * cache["centres"]).sum(dim=-1)
    return loss.numpy(), means.numpy()

r217_tables = {}
r217_screen = []
r217_train_rows = r217_cache_rows["train"].numpy()
for tag in ["insulin_only", "combined_strong"]:
    W = c206_candidates[tag]
    delta = (c211_logits(W, c210_train) - c210_train["L0"]).mean(dim=0).numpy()
    for shrink in [10., 50., 150.]:
        T = np.zeros(tuple(r217_T0.shape), dtype=np.float64)
        counts = np.bincount(r217_train_rows, minlength=len(T))
        np.add.at(T, r217_train_rows, delta)
        T /= (counts[:, None] + shrink)
        T = torch.tensor(-T, dtype=torch.float32)
        r217_tables[(tag, shrink)] = T
        for strength in [0.5, 1.0]:
            row = {"candidate": tag, "shrink": shrink, "centering": strength}
            for split, cache in [("train", c210_train), ("tune100", c210_tune)]:
                loss, mean = r217_eval(W, cache, T * strength, split)
                row[split] = float(np.mean(loss - cache["loss0"]))
            r217_screen.append(row)
print("=== CONDITIONAL FACTUAL CENTERING ===")
print(pd.DataFrame(r217_screen).to_string(index=False))
print("=== INITIAL CAUSAL/ROLLOUT NOTES ===")
print(Path("/work/notes.md").read_text()[:2800])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 217: neuron-removal hypothesis refuted; conditional centering screen\n"
            "Removal-only NLL costs are under 0.00008, so added causal correction dominates factual loss. "
            "Tested training-only conditional logit centering by reading code and previous bin; "
            "no weights installed. Rollout effects of any centering remain unverified.\n"
            + pd.DataFrame(r217_screen).to_string(index=False) + "\n")
```