The 300 fresh train stays give a baseline of 1.630 per reading (14,296 readings). That is higher than sel2's 1.404, so either sel2 is easier or this set differs in some way; I'll keep that in mind.

My explicit kill function shows the dim choice matters a lot:

| dims killed | factual cost (nats/reading) |
|---|---|
| old 8 role dims | +0.075 (se 0.003) |
| 4 old role dims | +0.037 |
| 4 smallest-rms dims (1, 17, 4, 132) | +0.0020 (se 0.0008) |
| 7 smallest-rms dims | +0.0072 (se 0.0013) |

So rms is a much better guide than the old noisy single-dim costs. Still, 7 dims (insulin plus NE) would break the budget on their own.

Next I'll measure single-dim kill costs on this larger set for the 60 smallest-rms dims, to find the truly cheapest ones. If several dims cost under about 0.0003 each, a 4-dim insulin scaffold for roughly 0.001 is realistic, and perhaps 7 dims for about 0.003.

```python
o=np.argsort(sc)[:60]
res=[]
t0=time.time()
for j in o:
    d=(cost(kill(PO,[int(j)]))-base).numpy(); res.append((int(j), d.mean(), d.std()/np.sqrt(len(d))))
print('%.0fs'%(time.time()-t0))
res.sort(key=lambda x:x[1])
print('cheapest:', [(j, round(m,5)) for j,m,s in res[:20]])
print('typical se %.5f'%np.median([s for _,_,s in res]))
print('sum of 4 cheapest %.5f, 7 cheapest %.5f'%(sum(m for _,m,_ in res[:4]), sum(m for _,m,_ in res[:7])))
single_cost={j:m for j,m,_ in res}
```