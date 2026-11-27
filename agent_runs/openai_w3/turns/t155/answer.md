The slot-level check passed: factual dose scores are small in every slot, and a 0.1-strength injection barely changes the normalized prelude. I’ll install the full-strength candidate reversibly and check factual fit first. An NLL increase above 0.005 would reject that strength; passing would justify testing its rollouts.

```python
pilot141_names = ['core.prelude.0.ffn.2.weight', 'core.prelude.0.ffn.2.bias']
pilot141_gains = np.array([1.0, 1.0])
pilot141_dW = (directions136 * pilot141_gains[None, :]) @ B136.T
pilot141_db = -pilot141_dW @ mu136
for name141, delta141 in zip(pilot141_names, [pilot141_dW, pilot141_db]):
    base141 = load_original(name141)
    save_edit(name141, base141 + torch.as_tensor(delta141, dtype=base141.dtype))
    print(name141, "original norm", float(base141.norm()),
          "edit norm", float(torch.as_tensor(delta141).norm()))
sf141 = (XF134.astype(np.float64) - mu136) @ B136
print("Factual score mean", sf141.mean(0), "SD", sf141.std(0),
      "absolute percentiles", np.quantile(np.abs(sf141), [.5,.9,.99,1], axis=0))
print("Pilot linear immediate targets:\n", K136 @ directions136)
print("Current edits:", edited())
with open('/work/notes.md', 'a') as f:
    f.write("\nPython runs 137-141; forwards 7. Recovered architecture and identification evidence. "
            "Record-derived immediate effects ~-1.94 insulin/glucose and +3.2 NE/MAP; "
            "12-hour tuning reference shifts -12.96,+6.48,+4.28,-2.14. "
            "Candidate FFN hidden decoder has small factual residuals in each slot "
            "(insulin SD ~.06, NE ~.034); no cancellation across slots. "
            "Installed reversible rank-two correction to core.prelude.0.ffn.2 weight and bias: "
            "dW=directions136@B136.T, db=-dW@mu136, gains [1,1]. "
            "Directions are numerical finite-difference calibrations, no model gradients. "
            "Factual acceptance and actual rollout effects remain untested. "
            "NE halving/stopping decoder is known imperfect; this is a pathway pilot, not a final edit.\n")
```
