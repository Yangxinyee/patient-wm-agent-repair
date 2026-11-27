Run 68 printed ARCHITECTURE.md again, but the list of kernel variables was too long, so the output was cut before the input history appeared. The variable list does confirm that my partial rebuild of the forward pass is still in memory: functions such as `lin`, `mlp`, `lnp`, `gapf`, `enc`, `encA` and `emb_visit2`, and the parsed bundles in `bundles_B`.

The architecture also confirms why the original model ignores doses. Insulin and norepinephrine reach the state only through u_k, which feeds the z posterior and one prelude token. At prediction time z is drawn from the prior, and the value head's "last value" register holds only earlier readings of the same code. So any dose effect has to be written into h, kept across steps, and read out by the value head.

To keep fixing the rebuild instead of starting over, I'll first record the current status in the notebook. Then I'll print the code of my last three runs (65–67), which tested the rebuild against the real model states. I expect to see which pieces I already checked and where the 8.7% mismatch could come from: normalisation type, attention details, or the coda.

```python
with open('/work/notes.md','a') as fh:
    fh.write("\n## Runs 43-67 (recalled after restart): building a local torch reimplementation of the forward pass "
             "(lin, mlp, lnp, gapf, enc, encA, emb_visit2 ...) checked against fwd_2 states. Best variant so far: "
             "gelu_tanh + 'fuse', rel err 0.0865 -> not exact yet.\n## Run 68: re-read ARCHITECTURE.md.\n")
n_in = len(In)
print('len(In)', n_in)
for i in range(n_in-5, n_in-2):
    print('=====', i)
    print(In[i][:3000])
```