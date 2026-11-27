The new dims are cheap: the full write-side build PA costs −0.0019 on sel2 and −0.0004 on dsE, so moving to dims 79/99/50 fixed the factual-cost problem. The decoder looks wrong, though. At insulin bundles the residual sd is 0.90 unit/h, far worse than the 0.07 I found earlier from the same encoder output. The decoder is what writes the dose into memory, so this error would go straight into it. Either my target is misaligned with the bundle rows, or a linear readout fails where insulin and norepinephrine share a bundle. To tell which, I'll score the old decoder (`dvec`/`d0` on all dims except 4, 2, 67) on the same bundles and split the residuals by whether norepinephrine is in the same bundle. If the old decoder is also poor here, the target or alignment is wrong; if it does well, I'll reuse its approach.

```python
keepO=np.array([k for k in range(192) if k not in [4,2,67]])
prO=u[:,keepO]@np.asarray(dvec)+d0
m_i=brI>0; both=m_i&(brN>0); solo=m_i&(brN==0)
for lab_,pp_ in [('old',prO),('new',pr)]:
    print(lab_, 'ins-solo n=%d sd %.3f | ins+ne n=%d sd %.3f | ne-only sd %.3f | nodrug mean %.3f'%(solo.sum(), np.std((y-pp_)[solo]), both.sum(), np.std((y-pp_)[both]), np.std((y-pp_)[(brN>0)&~m_i]), np.mean((y-pp_)[~isd])))
print('rate range ins', np.percentile(brI[m_i],[1,25,50,75,99]).round(2), 'dsE u-row alignment check:', len(u), len(brI), len(bd))
```