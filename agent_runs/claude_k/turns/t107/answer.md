My offline head reconstruction is approximate. Predicted means correlate 0.986–0.997 with the tool's (MAE 1.4–2.5 units), but per-reading NLL correlates only 0.84–0.93 and runs 0.01–0.05 nats high. The most likely cause is that the tool averages over 4 prior z draws while I used a single z. That is good enough to fit probes and to compare edits by difference, but I'll confirm factual NLL of any edit with the tool.

Before building anything, I want to know what the head's input already holds about the doses, and in what form. Specifically: is the dose information available as a linear signal, and does it extrapolate to doses outside the recorded range? That decides whether a linear correction can work at all. I'll check two things:
- that the cached arrays chain correctly (with_code → pre-activation → GELU → logits), so a weight edit can be costed exactly on them;
- what the Xi table holds, since I need per-reading dose-surprise features (recorded rate minus the rate expected from the last reading) at lags 1–2 as probe targets.

```python
W1 = load_original('attr_heads.value_bin.0.weight').astype(np.float64); b1 = load_original('attr_heads.value_bin.0.bias').astype(np.float64)
W2 = load_original('attr_heads.value_bin.2.weight').astype(np.float64); b2 = load_original('attr_heads.value_bin.2.bias').astype(np.float64)
pre_chk = WC99 @ W1.T + b1
print('PRE match', np.abs(pre_chk - PRE).max().round(5), '| Hact=gelu(PRE)', np.abs(gelu_np(PRE) - Hact99).max().round(5))
lg_chk = Hact99 @ W2.T + b2 + TR99
print('logit match', np.abs(lg_chk - lg99).max().round(5))
print('Xi cols', list(Xi.columns)); print(Xi.iloc[3].to_dict())
```