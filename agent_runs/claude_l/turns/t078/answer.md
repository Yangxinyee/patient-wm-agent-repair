`z` still shows a maximum error of 2.14. That maximum may come from one or two particular bundles, such as the first steps after admission, while the rest match. The old run-66 check also showed exact matches for most bundles.

Next I print the per-bundle errors for `z`, `out` and `h` for one variant, with each bundle's time and contents. If only a few early bundles fail on `z`, the encoders are fine and the remaining `out`/`h` error is a core-structure problem to localise. If `z` fails everywhere, something in ctx differs from the old function.

```python
er=onestep_err2(2020000000,dict(age_unit='s',age_mode='key'),False,False,nmax=30)
bl=BUN[2020000000]
for k in range(1,30):
    t,atoms=bl[k]
    print(k, t, [(a['code'],a['kind']) for a in atoms], 'z %.2e out %.3f h %.2f'%(er[k-1,2],er[k-1,0],er[k-1,1]))
```