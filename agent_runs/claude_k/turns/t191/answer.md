The per-row loop also fails. The single call that worked used `np.array([3600.0])`, while the loop passes a numpy float64 from `dts99`. A scalar-vs-array difference must be breaking the gap-feature construction inside `view_from`.

I'll try the first reading with different dt forms (Python float, numpy scalar, 1-element array, 2-D array) and see which work and whether they match `Q99[0]`. Then I'll use the form that works.

```python
kp=kidx[0]; d0=dts99[0]; print('d0',d0,type(d0), 'hr dtype',hr.dtype, 'kidx dtype',kidx.dtype)
for nm_,dd_ in [('pyfloat',float(d0)),('npscalar',np.float64(d0)),('arr1',np.array([float(d0)])),('arr2d',np.array([[float(d0)]]))]:
    for pn_,pv_ in [('2d',X8o[kp][None,:]),('1d',X8o[kp])]:
        try:
            qv=np.asarray(view_from(pv_, dd_)).ravel(); print(nm_,pn_,'ok err',np.round(np.abs(qv-Q99[0]).max(),6), qv.shape)
        except Exception as ex: print(nm_,pn_,'fail',repr(ex)[:60])
```