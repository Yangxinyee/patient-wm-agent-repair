The ungated read is too noisy. At action bundles the held-out residual sd is 0.44–0.62 for insulin and 0.17–0.30 for norepinephrine, which would put about ±1.5 mg/dL of noise on glucose predictions. Head 2 represents the u token (actions) and the e token (observations) in overlapping directions, so no linear read can separate them.

A cleaner route: edit the action encoder so that u carries the rate along a direction that e, slot and z tokens hardly use after norm1. Head 2 would then pass the rate cleanly at action bundles and carry almost nothing at no-action bundles, since it attends u at 0.99 at action bundles. Whether this works depends on whether such quiet directions exist. I will compute the smallest-variance directions of the norm1 vectors of e, slot and z tokens, and the u tokens' spread along them. Success means several directions with sd far below about 0.05, in which u can carry a rate signal.

```python
NEG=np.concatenate([n1[:,0:4].reshape(-1,192), n1[:,4], n1[:,6]],0)
U_act=n1[act,5]; U0=n1[~act,5]
mu=NEG.mean(0); C=np.cov((NEG-mu).T); ev,evec=np.linalg.eigh(C)
print('NEG n',len(NEG),'smallest sd',np.round(np.sqrt(np.maximum(ev[:12],0)),5))
print('largest sd',np.round(np.sqrt(ev[-5:]),3))
for j in range(8):
    v=evec[:,j]; print(j,'neg mean %.4f sd %.5f | u_act mean %.3f sd %.3f | u0 %.4f'%(NEG@v.mean() if False else (NEG@v).mean(), (NEG@v).std(), (U_act@v).mean(), (U_act@v).std(), (U0@v).mean()))
print('u0 const?', np.abs(U0-U0[0]).max(), '| norm1 of u0', np.round(U0[0][:5],4))
import subprocess
print([l for l in open('/model/ARCHITECTURE.md').read().split('\n') if 'encode_action' in l or 'action' in l.lower()][:12])
```