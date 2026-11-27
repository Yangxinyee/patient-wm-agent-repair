Run 255 failed on an unneeded call (`getW` needs an argument), so nothing ran. The question is the same as before: does the earlier channel build `W2` already contain encoder and prelude edits I can reuse? Rerun without that line.

```python
diffs=[]
for n_ in NAMES:
    if n_ not in W2: continue
    a0=load_original(n_); a2=W2[n_]
    a2=a2.numpy() if torch.is_tensor(a2) else np.asarray(a2)
    a0=a0.numpy() if torch.is_tensor(a0) else np.asarray(a0)
    if a0.shape!=a2.shape: diffs.append((n_,'shape',a0.shape,a2.shape)); continue
    d_=np.abs(a2-a0)
    if d_.max()>1e-7:
        nz=np.where(d_.reshape(d_.shape[0],-1).max(1)>1e-7)[0] if d_.ndim>1 else np.where(d_>1e-7)[0]
        diffs.append((n_, round(float(d_.max()),4), 'rows', nz[:6].tolist(), len(nz)))
print('n diff', len(diffs))
for x_ in diffs: print(x_)
print('nsc', float(nsc), '|vI|', round(float(np.linalg.norm(vI)),3), '|wI48|', round(float(np.linalg.norm(wI48)),3))
```