The offline factual evaluator works again after the fix. `tf_nll(PO)` gives 1.40449 per reading on the 5,802 sel2 readings, against the tool's 1.40394; the gap is z-sampling noise, since I use fixed prior draws.

It returns per-reading NLL, probabilities, predicted means, and the h and out states, in about 0.5 s. With fixed draws, edited-minus-original differences are paired, so small costs can be measured precisely.

Next I'll measure the cost of the kill scaffold `PS`, and of `P_` (head-side kills only). If the 8 role dims cost about 0.001–0.002, there is room for the memory wiring within the 0.005 budget. If they cost more, I'll need to rechoose dims.

```python
def nllm(P): 
    r=tf_nll(P, ret=True); return float(r[0].mean()), r
n0,R0=nllm(PO); nS,RS=nllm(PS); n_,R_=nllm(P_)
print('PO %.5f | PS %.5f (d %.5f) | P_ %.5f (d %.5f)'%(n0,nS,nS-n0,n_,n_-n0))
d=(RS[0]-R0[0]).numpy(); print('PS per-reading diff: mean %.5f sd %.4f se %.5f'%(d.mean(), d.std(), d.std()/np.sqrt(len(d))))
cc=codes_.numpy() if torch.is_tensor(codes_) else codes_
for c in np.unique(cc): print('code',int(c),'n',int((cc==c).sum()),'dNLL %.5f'%d[cc==c].mean())
```