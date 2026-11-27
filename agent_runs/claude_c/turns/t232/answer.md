Pass 26 succeeded, but the printout cut off before the edited model's NLL. The architecture grep shows the prelude attends over 7 tokens per bundle: 4 h-slot tokens, the event token e_k, the action token u_k and a z token. Next I will print the overall and per-event NLL for both models, plus the core computation section of the architecture. This tells me whether the combo stays within the 0.005 budget and exactly where u_k enters.

```python
S=json.load(open('/work/fwd_26/summary.json'))['results']
for m in ['original','edited']:
    r=S[m]; print(m, r['value_nll_per_value'], {k:round(v['nll'],4) for k,v in r['by_event'].items()})
i=A.find('**Prelude**'); j=A.find('**The world')
print(A[max(0,i-900):j][:2600])
```