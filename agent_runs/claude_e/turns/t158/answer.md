The helpers are clear now. gated3 builds four repurposed hidden units (`units`) in the value head. ROi5 and ROn4 are the readout weights, Gi3 and Gn2 are the gates, tg and tm are the logit directions for glucose and MAP.

My working assumption is that each drug uses a pair of units, GELU(readout + gate) - GELU(gate), so the output equals the readout when the gate is on and 0 when it is off. Before writing a new NE pair from a rate-level readout, I'll check this against gated3(-1.9, 3).

For the NE pair I'll compare the row difference (u3 - u4) and the common part with ROn4 and Gn2, and look at the biases and W2 columns. If the difference matches the readout and the common part matches the gate, I can rebuild only the NE pair and keep the insulin pair.

```python
E = gated3(-1.9, 3)
W0e = npv(E['W0']) if not isinstance(E['W0'], np.ndarray) else E['W0']
b0e = npv(E['b0']) if not isinstance(E['b0'], np.ndarray) else E['b0']
W2e = npv(E['W2']) if not isinstance(E['W2'], np.ndarray) else E['W2']
print('units', units, 'W0', W0e.shape, 'W2', W2e.shape)
def cos(a,b): a=np.ravel(a); b=np.ravel(b); return float(a@b/np.linalg.norm(a)/np.linalg.norm(b))
u1,u2,u3,u4 = units
for nm,(ua,ub),ro,gt in [('ins',(u1,u2),ROi5,Gi3),('ne',(u3,u4),ROn4,Gn2)]:
    dif = W0e[ua]-W0e[ub]; com = W0e[ub]
    print(nm, 'b0', round(float(b0e[ua]),3), round(float(b0e[ub]),3),
          '| cos(dif,ro.w) %.3f |dif|/|ro.w| %.3f'%(cos(dif,ro['w']), np.linalg.norm(dif)/np.linalg.norm(ro['w'])),
          '| cos(com,gate.w) %.3f'%cos(com, gt['w']), '| ro w0 %.3f gain %.3f r2 %.3f'%(ro['w0'], ro.get('gain',np.nan), ro.get('r2',np.nan)))
    print('   W2 cols cos tg %.3f tm %.3f, W2a+W2b norm %.2e'%(cos(W2e[:,ua],tg), cos(W2e[:,ua],tm), np.linalg.norm(W2e[:,ua]+W2e[:,ub])))
print('Gi3', {k:(np.shape(v) if np.ndim(v) else v) for k,v in Gi3.items()}, 'Gn2', {k:(np.shape(v) if np.ndim(v) else v) for k,v in Gn2.items()})
print('globals', {k: (globals()[k] if np.ndim(globals()[k])==0 else np.shape(globals()[k])) for k in ['K_','A_g','M_i','B_n','M_n']}, 'gvn', np.shape(gvn))
```