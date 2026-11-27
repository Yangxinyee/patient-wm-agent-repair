wc_vec builds the head input from five pieces. q = LNn('query_norm', ctx + z_head(z_prior mean) + type_emb[1]), with ctx = out + out_norm(mean of decayed h) + gap(log dt / 10). Then the code embedding, the last-numeric MLP and the bin embeddings are added.

So z draws change only q, and with_code(z) = with_code(z̄) + [q(z) − q(z̄)]. That lets me add z-draw noise to existing rows if I know each row's previous state and dt. I'll look at gen_rows_all's row fields to find the bundle index. Then I'll check that my q_of(prev, dt, eps=0) reproduces wc_vec's q part: the difference wc − q_of should be the same constant for two different states with the same code and last-reading pattern.

```python
for r in ra2[:4]: print(r[:8])
print([round(b_['hour'],3) if isinstance(b_,dict) and 'hour' in b_ else type(b_).__name__ for b_ in B0_[:6]])
print(St0_.shape, St0_.dtype)
```