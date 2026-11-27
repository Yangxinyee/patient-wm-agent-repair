Putting a common vector into the prelude's slot outputs barely moves the next glucose prediction. The gradient is about 0.02 mg/dL per unit, consistent in direction (cosine 0.93), and linear but tiny: −0.66 mg/dL at norm 30. MAP is essentially untouched.

So the core state has very little grip on the next reading. The prediction is likely driven mostly by the head's last-value register, the previous reading, and the memory of readings. Pushing a stronger signal through head 3 will not fix that by itself.

To tell whether the bottleneck is the core update or the head, I'll measure the head's own leverage. I'll take the gradient of the next glucose prediction with respect to the view vector at the same 20 insulin bundles, then the response along the best direction at norms 1 to 30, compared with the view's natural spread along it.
- Strong view leverage means the core attenuates the signal, and an injection that bypasses the LayerNorm stack is needed.
- Weak view leverage means the head is register-dominated, and the dose effect must be added inside the value head itself.

```python
def pred_from_view(k, cv):
    return np.array([float(value_mean(cv.double().reshape(1,192), [cd], np.array([LV[cd][k+1]]), np.array([(INFO[k+1]['t']-LT[cd][k+1])*60.0]))[0][0]) for cd in ('glucose_finger','map_arterial')])
def view_of(k):
    c, z, h, o = step_local(k, ztok_add=zero); dt1 = dts[k+1]; return ctx_from(o, h*torch.exp(-dt1*torch.exp(fa)), dt1)
JV = np.zeros((len(selX),192,2)); V0 = []
for a_,k in enumerate(selX):
    cv = view_of(k); V0.append(cv); b0 = pred_from_view(k, cv)
    for i in range(192): JV[a_,i] = pred_from_view(k, cv + I192[i]) - b0
gV = JV[:,:,0].mean(0); cV = (JV[:,:,0]@gV)/(np.linalg.norm(JV[:,:,0],axis=1)*np.linalg.norm(gV))
print('view sens gf: |g| med %.3f mean-dir %.3f cos med %.2f; MAP |g| med %.3f'%(np.median(np.linalg.norm(JV[:,:,0],axis=1)), np.linalg.norm(gV), np.median(cV), np.median(np.linalg.norm(JV[:,:,1],axis=1))))
dV = torch.tensor(-gV/np.linalg.norm(gV), dtype=torch.float32); Vst = torch.stack(V0)
print('natural sd of view along dir: %.2f ; view norm med %.1f'%(float((Vst@dV).std()), float(Vst.norm(dim=1).median())))
for s_ in [1,3,10,30]:
    d = np.array([pred_from_view(k, V0[a_] + s_*dV) - pred_from_view(k, V0[a_]) for a_,k in enumerate(selX[:10])]); print(s_, d.mean(0).round(2), d[:,0].std().round(2))
```