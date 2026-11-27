The u-encoder replica is available (`ins_atom`, `ne_atom`, `U_of`). I'll check that `U_of` reproduces the u_k token of every drug bundle from its recorded insulin and NE rates. Then I can build exact prelude inputs with insulin rates halved or doubled, to make the expected-rate readout insensitive to the current dose and to check the whole injection under the evaluation's edits.

```python
def u_for(ri, rn):
    at = []
    if ri > 0: at.append(ins_atom(ri))
    if rn > 0: at.append(ne_atom(rn))
    return U_of(at)
t0 = time.time()
Urec = torch.stack([torch.as_tensor(u_for(a, b)).reshape(-1).float() for a, b in zip(rI, rN)])
err_u = (Urec - TOK[:,5,:]).abs().max(1).values
print('U_of shape', tuple(Urec.shape), 'max err %.2e  frac>1e-3 %.4f  time %.1fs' % (float(err_u.max()), float((err_u>1e-3).float().mean()), time.time()-t0))
```