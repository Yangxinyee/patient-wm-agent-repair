Head 2 of the prelude is a near-perfect carrier. Its attention from every slot to the u token averages 0.996 (sd 0.05, 10th percentile 1.000). Anything written into head 2's value for u therefore reaches all four slots with gain ≈1, regardless of state. Heads 0 and 3 are fairly steady (0.92); head 1 is erratic.

That makes a linear dose channel buildable along this path:
- **Encoder:** a repurposed hidden unit in `encode_action.mlp1` that reads the numeric z_rate input in its linear regime, gated on the drug code.
- **Encoder output:** a repurposed unit in `mlp2` that carries it into u, through the LayerNorm.
- **Prelude:** head 2's value and output projections carry it into the slots.
- **Core to state:** `b_in`, `c_out` and the coda carry it into `out`.
- **Value head:** gated units turn it into a glucose or MAP shift.

My earlier self seems to have built such a channel (`make_core_chan` produced the edited weight dict `W2`, with `corechan.npz`), but I don't know how far it got. Before rebuilding, I'll check which tensors W2 changed and by how much, and what scale it used. If the edits sit in encode_action and the prelude attention, I can reuse that construction rather than rebuild it.

```python
W0_=getW() if 'getW' in globals() and callable(getW) else None
diffs=[]
for n_ in NAMES:
    a0=load_original(n_); a2=W2[n_] if n_ in W2 else None
    if a2 is None: continue
    a2=a2.numpy() if torch.is_tensor(a2) else np.asarray(a2)
    a0=a0.numpy() if torch.is_tensor(a0) else np.asarray(a0)
    if a0.shape!=a2.shape: diffs.append((n_,'shape',a0.shape,a2.shape)); continue
    d_=np.abs(a2-a0)
    if d_.max()>1e-7:
        nz=np.where(d_.reshape(d_.shape[0],-1).max(1)>1e-7)[0] if d_.ndim>1 else np.where(d_>1e-7)[0]
        diffs.append((n_, round(float(d_.max()),4), 'rows', nz[:6].tolist(), len(nz)))
print('n diff', len(diffs)); 
for x_ in diffs: print(x_)
print('nsc', float(nsc), '|vI|', round(float(np.linalg.norm(vI)),3), '|wI48|', round(float(np.linalg.norm(wI48)),3))
```