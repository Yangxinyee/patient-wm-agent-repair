The vocab explains how the model sees numbers:
- Each value is a robust z-score, (x − median)/IQR clipped to ±10, plus one of 16 fine bins.
- The value head is shared across codes: a 17-way softmax over per-code bins. The predicted mean is Σ p(b) × centre(b). Glucose bins are 10–17 mg/dL wide in the treated range.

This has consequences for the edits:
- Norepinephrine ×2 can push rates above 1.0. Any rate above 1.0 falls in rate bin 16, and any dose above 4.8 in dose bin 16. Training rates never exceed 1.0, so those embeddings were never trained.
- Insulin ×2 pushes rates of 3 or more into bin 16, which training saw only about 1,600 times.

So I need to see how the dose embeddings, the action encoder and the value head are built before choosing an edit. I will list every tensor with its shape, grouped by module, and print the architecture parts on heads and the core.

```python
T = json.load(open('/model/tensors.json'))
print(type(T), (list(T.keys())[:5] if isinstance(T, dict) else T[:3]))
for n in NAMES:
    print(n, SHAPES[n])
```