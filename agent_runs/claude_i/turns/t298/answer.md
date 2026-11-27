Result:
- The view-side numbers are encouraging. The head reacts strongly to its input view: about 13 mg/dL of glucose_finger, 11 of glucose_lab and 4.4 mmHg of MAP per unit of view norm. The glucose and MAP directions are nearly orthogonal (cosine 0.13).
- z_prior has 29 dead hidden units, and z_head maps 32 prior dimensions into the view. So a head-side amplifier is available.
- The token-linearity test is contradictory. Along the fitted glucose direction the shift was essentially zero at every norm, although the per-axis sensitivities were about 0.27 per unit.

Either the projected direction is degenerate (glucose and MAP token sensitivities nearly parallel), or the bias-patching trick misbehaves for some inputs. I'll check this on one bundle before building on it:
- the cosine between the glucose and MAP token sensitivities;
- the predicted change along the direction compared with the measured change;
- the out-vector change for a single-axis perturbation compared with the fitted direction.

```python
k = kI[0]; j = int(np.argmax(np.abs(Jtok[0,:,0])))
print('cos token Jg,Jmp %.3f  |v| %.4f  Jtok0.vh %.4f' % (Jg@Jmp/np.linalg.norm(Jg)/np.linalg.norm(Jmp), np.linalg.norm(v), Jtok[0,:,0]@N(vh)))
_,o0x = one_step(k)
for name, d in (('axis j', torch.zeros(192,dtype=dtw).index_fill_(0, torch.tensor([j]), 0.25)), ('2vh', 2*vh), ('40vh', 40*vh)):
    h1,o1 = one_step(k, d); P = preds_next(torch.stack([h1, one_step(k)[0]]), torch.stack([o1, o0x]), k)
    print(name, 'dout %.4f' % float((o1-o0x).norm()), 'dpred', (P[0]-P[1]).round(3))
print('vh dtype', vh.dtype, 'norm', float(vh.norm()), 'nan?', bool(torch.isnan(vh).any()))
```