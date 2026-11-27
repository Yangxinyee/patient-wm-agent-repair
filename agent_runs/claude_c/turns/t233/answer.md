Pass 26 puts the combo at NLL 1.63969, which is +0.0051 over the original. That breaks the 0.005 limit, so this combo cannot be submitted as is. The architecture confirms the dose only enters through u_k → prelude attention → prelude_norm. Its effect on E is diluted by |h| (about 842), which explains why input-side gains saturate.

My metric estimate says numeric x5 alone barely beats the original on average. Its NE-half readout gets worse, and NE x2 is distorted by out-of-range rates at the cap (bin 16). Next I will print the numeric feature list and the shapes of the embedding, encode_action and z_post tensors. Then I can check two things: whether the out-of-range bins can be repaired by mixing rows of the bin tables, and whether z_post is linear.

```python
print(VOCAB['numeric'])
print(str(VOCAB['numeric_note'])[:400])
for n in NAMES:
    if any(k in n for k in ['embed.','encode_action','z_post','z_in']): print(n, SHAPES[n])
```