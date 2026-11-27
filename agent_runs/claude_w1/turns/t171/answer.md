The per-query least-squares fit does find consistent directions (insulin R² 0.98, NE R² 0.90), but only with enormous weights. The insulin direction needs |α| ≈ 1,000 per 1 mg/dL per u/h, while the realistic budget is about 70 per 3 mg/dL. Consistency inside the prelude costs about 30 times more than the best average direction. The cause is the LayerNorm after the prelude together with the large norm of the state h: an exact-dose injection there turns into a weak, context-dependent effect. So the action-token injection on its own is not viable.

This also exposes a conceptual trap for NE. The counterfactual is a sustained ×2 level, not a one-off innovation. So the correction must respond to the dose level, and the factual compensation must depend on readings only, not on past doses. Otherwise a sustained doubling cancels after the first hour.

The most direct place to control a predicted mean is the value head. There the correction is c × decoded dose − comp(last reading). The open question is whether head-level decoding noise fits the factual budget. I can estimate that offline on setA with exact tool residuals, using expected NLL change ≈ (E[δ²] − 2E[rδ]) / (2σ²) per code. First I need to see what my stored decoder dictionary holds.

```python
print(type(DEC).__name__, list(DEC.keys())[:20] if isinstance(DEC, dict) else '')
for k in (list(DEC.keys())[:4] if isinstance(DEC, dict) else []):
    v = DEC[k]; print(k, type(v).__name__, [ (type(x).__name__, np.shape(x)) for x in (v if isinstance(v,(tuple,list)) else [v])][:4])
print(F2df.groupby('event').agg(n=('nll','size'), nll=('nll','mean'), rsd=('observed', lambda s: 0)).to_string())
F2df['res'] = F2df.observed - F2df.predicted_mean
print(F2df.groupby('event').res.agg(['mean','std']).round(2).to_string())
```