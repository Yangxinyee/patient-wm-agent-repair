Run 218 failed on a shape mismatch: each code has 16 value centres for logit bins 1–16, and bin 0 is a special slot. Next I will fix the mean to use bins 1–16 and print the bin-0 probability mass to check it is negligible. Then I will rerun the same lam 1/2/4 scan with the same criteria as before.

```python
code=code.replace("float(pf[i]@c),float(pc[i]@c)","float(pf[i][1:]@c),float(pc[i][1:]@c)")
open('/work/amp.py','w').write(code); exec(code)
for lam in [1.0,2.0,4.0]:
    r=summ(lam,stays100); print(lam,'nll %.4f glu %.4f map %.4f | tool %.4f | GFshift %.2f n%d'%(r[0],r[1],r[2],r[5],r[3],r[4]))
set_amp(1.0)
```