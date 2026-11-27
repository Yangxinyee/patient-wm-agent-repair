Restricted to glucose readings right after an insulin record (n = 4,583), the lag-1 surprise does decode well from the value-head input (R² 0.655, against 0.70 from `out` and 0.71 from the full state). So little is lost before the head. The low pooled R² comes from mixing in readings with no preceding insulin, where the expected dose is a step function of glucose that a linear readout cannot cancel.

The clean structure is therefore a gated correction: gain × (insulin was given) × (decoded rate − decoded expected rate). That gating needs a pair of GELU hidden units per drug. The value head has no spare units, so I would have to repurpose units whose outputs are almost linear combinations of the others, folding their contribution into the rest of W2. Next I'll measure how redundant each of the 192 hidden units is across the 19,451 readings. If several units have R² above 0.999 on the others, repurposing four of them costs almost nothing in factual NLL.

```python
A0 = WC0@W1.T + b1; Hh0 = gelu_(A0)            # hidden activations (mean z)
Hc = Hh0 - Hh0.mean(0)
G_ = Hc.T@Hc
r2u = np.zeros(192)
for j in range(192):
    o = np.r_[0:j, j+1:192]
    beta_ = np.linalg.solve(G_[np.ix_(o,o)] + 1e-6*np.eye(191), G_[o,j])
    r2u[j] = 1 - (G_[j,j] - G_[o,j]@beta_)/G_[j,j]
imp = np.linalg.norm(W2,axis=0)*Hh0.std(0)
order = np.argsort(-r2u)
print('top redundant units', [(int(j), round(r2u[j],5), round(imp[j],3)) for j in order[:10]])
print('R2 quantiles', np.quantile(r2u,[0,.1,.5,.9,1]).round(4), 'dead units (std<1e-3):', int((Hh0.std(0)<1e-3).sum()))
print('frac pre-act >3 per unit: glucose/map, top units:', [(int(j), round((A0[glm_all:=RD2.event.str.startswith('glucose').values][:,j]>3).mean(),2)) for j in order[:5]])
```