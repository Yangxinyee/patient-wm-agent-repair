The run failed on a shape detail. For a single-target fit, sklearn returned the NE decoder's coefficients as a flat vector rather than a column, so indexing it as a column broke. The decoders and tilts were already computed, so I only reshape the NE decoder and rerun the build and the offline evaluation.

The evaluation reports the four-draw NLL change by code and the spread of the injected mean shifts. If total ΔNLL comes out near my rough +0.01 estimate, the full NE scaling has to be cut.

```python
Wd_ne = np.asarray(Wd_ne).reshape(192,-1); w0_ne = np.ravel(np.asarray(w0_ne))
print('dec shapes', Wd_ins.shape, Wd_ne.shape, 'tilt norms', round(float(np.linalg.norm(t_G)),3), round(float(np.linalg.norm(t_M)),3))
E1 = build()
ne1, dm1 = eval_edit(E1)
print('dNLL all', round(float(ne1.mean()-base4),5), {c: round(float((ne1-nb)[evc==c].mean()),4) for c in ['glucose_finger','glucose_lab','map_arterial','map_cuff']})
print('dmean sd gluc/map', round(float(dm1[isg].std()),2), round(float(dm1[ism].std()),2))
```