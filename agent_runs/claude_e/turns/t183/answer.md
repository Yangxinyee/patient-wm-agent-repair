With the correct labels the insulin gate is near-perfect on held-out rows. On rows after an insulin bundle the pair output equals the readout (mean error 0.005), on other glucose rows it is exactly 0, and only 4 of 3,222 rows are misclassified. So the gate is not the source of the factual shift on glucose rows without recent insulin.

The other candidate is the four hidden units I repurposed. Their original contributions to the value head were removed, and any part not offset by the transition-table edit changes predictions on every row, gated or not.

I'll test this offline on the held-out rows. I'll compute the full value head's predicted mean from WC with the original and with the edited weights, and look at the shift on glucose and MAP rows without the relevant drug in the previous bundle. A shift around 0.5 there would confirm the unit-removal explanation.

```python
b2 = load_original('attr_heads.value_bin.2.bias'); b2 = b2.numpy() if hasattr(b2,'numpy') else np.asarray(b2)
print([k for k in VOCAB.keys()][:20])
cent = VOCAB['value_bin_centres']
print(type(cent), (list(cent.keys())[:6] if isinstance(cent, dict) else np.shape(cent)))
```