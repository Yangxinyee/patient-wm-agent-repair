The projected costs show that centring matters a great deal.

- A lag-1 insulin effect of −1.98 per U/h on the raw dose would cost about 0.0072 nats per reading.
- Centred at the median dose of 1.5 U/h, which is how the z-scores are centred, the cost is about 0.0023. That fits within the 0.005 limit.
- So amplifying the z-score dose pathway is a viable route, provided it acts linearly.

The original model's lag-1 insulin x2 shift rises only 0.15 per U/h of recorded dose; the estimated true slope is 1.98. For norepinephrine, the odd negative responses come from stays at the 1.0 cap. At the cap the x1/2 response has the right sign but is too small (−0.7 vs −1.7). The x2 response is wrong (−0.5 to −1.0), probably because rates above 1.0 are never seen in training. My ×5 embedding edit flattened norepinephrine responses to about zero.

The ×5 edit raised the insulin response only 1.36× (x2) and 2.1× (x1/2). This points to saturation somewhere. It could be in the action encoder (whose final LayerNorm can compress large inputs) or further downstream.

To find out where, I will run a states pass with both models on the same 100 tuning stays. I will compare how strongly the z posterior and the state change across insulin bundles depend on dose.

- If the state's dose sensitivity grew about 5× while the glucose response grew 1.4×, the bottleneck is downstream (heads or state-to-prediction).
- If it grew only about 1.4×, the bottleneck is the encoder, and I must amplify after it.

First I need the exact list of the 100 stay ids used before.

```python
ids100 = [int(x) for x in pd.unique(S3['stay_id'])]
print(len(ids100), ids100[:3], ids100[-3:])
print(json.dumps(ids100))
```