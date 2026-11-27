The leftover error is not a constant offset (mean cosine 0.19, RMS 0.22), so it depends on the input. Before chasing encoder details I'll check the saved state against the documented relation `out = out_norm(mean over slots of h)` on many rows, and print the dtype and size of the saved values. An exact match would confirm the layout and the LayerNorm settings; a mismatch would show the documented state relations are incomplete.

```python
print(Z['x'].dtype, 'z rms', round(float(X0[:,960:992].pow(2).mean().sqrt()),3))
H_=X0[:,192:960].reshape(-1,4,192)
for eps_ in [1e-5,1e-6]:
    o_=Fn.layer_norm(H_.mean(1),(192,),Wt['core.out_norm.weight'],Wt['core.out_norm.bias'],eps_)
    print(eps_, 'max|out - LN(mean h)|', round(float((o_-X0[:,:192]).abs().max()),5), 'rms out', round(float(X0[:,:192].pow(2).mean().sqrt()),3))
```