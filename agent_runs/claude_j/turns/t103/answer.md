The value head is a two-layer MLP: 192 inputs to 192 hidden units to 17 classes. The hidden layer makes a code-specific correction possible. One option is to repurpose a hidden unit that is nearly inactive and gate it through `embed.code[code]`, so that it fires only for glucose codes, or only for MAP codes. It would then add a dose-dependent tilt to the logits. Also, `last_numeric` takes 8 inputs, not the 4 the document lists.

To do this I need a replica of the heads only, not of the full step. The inputs are the tool's states, the time flow, the prior over z, the view and the value head, and I would check it against the per-reading predicted means from fwd_2. Next I will list the remaining head-related tensors (gap, z_prior, flow) and the columns of the fwd_2 CSV, printing only names and shapes.

```python
for n in NAMES:
    if any(s in n for s in ['gap', 'z_prior', 'flow', 'embed.code', 'embed.bins', 'embed.numeric', 'categorical']):
        print(n, SHAPES[n])
fa_ = pd.read_csv('/work/fwd_2/factual_original.csv')
print(fa_.shape, list(fa_.columns))
print(fa_.head(3).to_string()[:600])
print('ACT', ACT, 'an', an)
```