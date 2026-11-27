The diagnostic splits the result:

| Unit pair | Glucose rows, insulin before | Glucose rows, no insulin | Arterial (7) | Cuff (5) |
|---|---|---|---|---|
| Insulin pair | equals the readout (error ≤ 0.14) | 0 (a few outliers) | median 91, not 0 | median 49, not 0 |
| Norepinephrine pair | — | finger 0; lab small leakage (to −1.2) | equals the readout exactly | 0 |

So the norepinephrine pair is correct, apart from the lab leakage. The insulin pair turns fully on for the MAP codes, which explains the huge MAP shifts.

By my arithmetic the glucose-versus-MAP code gate alone should give −40 for MAP rows. The insulin-gate term must therefore be far larger on MAP rows than I measured on averaged inputs.

Next I will decompose the insulin pair's pre-activation on MAP rows into its terms: code-gate term, insulin-gate term and bias. I will check that the hidden output matches GELU of my computed pre-activation. I will also repeat the projection with the averaged inputs, to look for a row-alignment or input mismatch.

If one term explains the ~+130, the fix is a gate fitted against all codes on per-draw inputs. If my computed pre-activation does not match the head's hidden output, `head_run` builds the input differently from what I assume.

```python
W0e=npv(Eg['W0']).astype(np.float64); b0e=npv(Eg['b0']).astype(np.float64)
pre_u1=WCd0@W0e[u1]+b0e[u1]
gel=lambda x: 0.5*x*(1+erf(x/np.sqrt(2)))
print('hidden u1 vs GELU(pre) maxabs %.3e'%np.abs(H_[:,:,u1]-gel(pre_u1)).max())
wg,cgv,sdg=gi_
term_gate=80*(WCd0@wg-cgv)/sdg; term_code=40*(WCd0@gvn-1.76)/2.6
for c in [6,10,7,5]:
    m_=cd0==c
    print('code',c,'ins-gate term median %.1f [%.1f,%.1f] | code term median %.1f | pre median %.1f'%(np.median(term_gate[:,m_]),term_gate[:,m_].min(),term_gate[:,m_].max(),np.median(term_code[:,m_]),np.median(pre_u1[:,m_])))
pa_=(ALLW[0]@wg-cgv)/sdg
print('avg-input ins-gate proj by code (dataset0):',{c:round(float(np.median(pa_[cd0==c])),2) for c in [6,10,7,5]})
print('per-draw ins-gate proj by code (dataset0):',{c:round(float(np.median(((WCd0@wg-cgv)/sdg)[:,cd0==c])),2) for c in [6,10,7,5]})
```