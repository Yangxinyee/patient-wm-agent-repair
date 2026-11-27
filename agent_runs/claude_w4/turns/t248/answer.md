Adding the flowed-slot term changes little. The view-based sensitivities, per unit of injection after prelude attention, are:

| Readout | per unit of injection | injection norm per 1 unit | consistency across bundles (cos) |
|---|---|---|---|
| finger glucose | 0.037 mg/dL | ≈27 | 0.999 |
| arterial MAP | 0.058 mmHg | ≈17 | 1.000 |
| lab glucose | 0.010 mg/dL | ≈95 | 0.994 |
| cuff MAP | 0.028 mmHg | ≈36 | 0.999 |

Finger and arterial gradients are nearly orthogonal (cos 0.03), so the two drugs can be injected independently.

So a first-step kick of 1.5–2 mg/dL per unit/h of insulin needs an injection of norm about 40–55 per unit/h. That is 5–7% of the ~800 slot norm per unit/h, roughly linear for typical rates of 1–3 units/h and increasingly nonlinear above about 4. NE is easier: about 50–90 per mcg/kg/min over a 0–1 range.

This is feasible, and a forward dose_edit will settle the real gain, since rollout persistence and register feedback can amplify a first-step kick. I'll build v4 as a head-1 carrier with three properties:
- **Leak-free readouts.** Each readout is fitted to recover the rate from the normalised action token and constrained to be about 0 on slot, event, z and no-action tokens.
- **Remapped output.** Head-1 output is remapped along unused value directions, so the injection can point exactly along the glucose or MAP gradient.
- **No side effects.** The remap must leave every original token's contribution unchanged.

Before fitting, I need two checks:
1. Do the v3 value directions (aI3, aN3, aP3) have ~0 original value on the slot, event and z tokens too, not just the action tokens? This decides whether the remap is safe.
2. What rate labels exist for the cached action-token sets (UG/grid_pairs, UR/pairs_real), so the readouts can be trained across edited doses?

```python
V1d=V1.astype(np.float64)
def vproj(X):  # X raw tokens (n,192) -> head-1 value projections on aI3,aN3,aP3
    return (LN1(np.asarray(X,dtype=np.float64))@V1d.T)@np.stack([aI3,aN3,aP3]).T
for nm,X in [('slot tokens (drug bundles)',TOK[:,:4].reshape(-1,192)),('e tokens drug',TOK[:,4]),('z tokens drug',TOK[:,6]),('slot tokens non-drug',TOKn[:,:4].reshape(-1,192)),('e tokens non-drug',TOKn[:,4]),('u tokens non-drug',TOKn[:,5]),('core.slots',sl)]:
    P=vproj(X); print(f'{nm:28s} n {len(P):5d}  proj mean {np.round(P.mean(0),4)}  sd {np.round(P.std(0),4)}  max|.| {np.round(np.abs(P).max(0),3)}')
print('TOKn u tokens distinct:', len(np.unique(np.round(np.asarray(TOKn[:,5]),4),axis=0)), ' -> no-action token?')
print('grid_pairs sample', grid_pairs[:4], len(grid_pairs), ' UG', UG.shape, ' pairs_real sample', pairs_real[:4], len(pairs_real), ' UR', UR.shape)
print('UC', UC.shape, 'Ucomb keys sample', list(Ucomb.keys())[:3])
```