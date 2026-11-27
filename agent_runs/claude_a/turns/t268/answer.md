The truth kernels are long. Insulin lowers glucose over about 9 hours, with a total of −11.2 mg/dL per unit/h of sustained infusion. Norepinephrine raises MAP over about 5 hours, with a total of +9.2 per unit. The real surprise is that the reserve-only scaffold on dims [4, 2, 67] already costs +0.0093, essentially all of P1's cost. So the write pathway is not the problem. The scaffold does more than the plain dim ablation, which earlier cost only about 0.002 for 4 dims. It also zeroes these dims in the prelude norm1 parameters, the query_norm bias, and the value-MLP input columns. Any of those can remove a constant, code-dependent offset rather than replace it by its mean, and that would explain the broad damage (25% of readings worse by more than 0.05).

I'll measure the cost of each part separately on sel2 by restoring them one at a time. If the extra zeroings (especially the value-MLP columns or the query_norm bias) carry the cost, I'll replace them with mean-compensated versions. If plain ablation of these dims is already expensive on sel2, I need different dims.

```python
J=[4,2,67]; pre='core.prelude.0'
def restore(Pd, parts):
    Q={k:v.clone() for k,v in Pd.items()}
    if 'n1' in parts:
        for t_ in ['.norm1.weight','.norm1.bias']: Q[pre+t_][J]=PO[pre+t_][J]
    if 'qb' in parts: Q['query_norm.bias'][J]=PO['query_norm.bias'][J]
    if 'vw' in parts: Q['attr_heads.value_bin.0.weight'][:,J]=PO['attr_heads.value_bin.0.weight'][:,J]
    return Q
for parts in [('n1','qb','vw'),('qb','vw'),('n1','vw'),('n1','qb'),('vw',),('qb',),('n1',)]:
    print('restored',parts,'d=%+.5f'%(nll_of(restore(Psc,parts))-n0))
```