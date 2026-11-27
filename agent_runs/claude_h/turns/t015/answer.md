The norepinephrine policy is smooth rather than a step scale. The probability of a dose falls from 99% at MAP ≤ 40 to 0.2% at MAP > 90, crossing about 50% near MAP 67–70, and the mean rate rises as MAP falls. The policy works similarly on cuff MAP when there is no arterial line.

Norepinephrine is also given in hours with no MAP reading (mean rate 0.13), so it likely depends on the previous rate: a titrated continuous infusion. And 94% of hours after a norepinephrine hour have some MAP reading.

Next I will model the norepinephrine rate as a function of the previous rate and the current MAP. I will look at Δrate against MAP and the previous rate, and at the residual's autocorrelation and its relation to the MAP trend. That tells me whether there is an iid random component to exploit, as with insulin. With titration, the residual noise in each hour's change (if iid) is the instrument.

```python
d = P.copy()
d['map_any'] = d.ma.combine_first(d.mc)
d['src'] = np.where(d.ma.notna(),'art', np.where(d.mc.notna(),'cuff','none'))
d['dne'] = d.ne - d.ne_prev
sub = d[(d.ne_prev>0)]
sub['mb'] = pd.cut(sub.map_any, [0,50,55,60,62,64,65,66,68,70,75,80,200])
print(sub.groupby(['mb']).agg(dne=('dne','mean'), sd=('dne','std'), stop=('ne', lambda x:(x==0).mean()), n=('ne','size')).round(3).to_string())
print(sub.groupby('src').agg(dne=('dne','mean'), stop=('ne', lambda x:(x==0).mean()), n=('ne','size')).round(3))
# start rule when ne_prev==0
s0 = d[(d.ne_prev==0)]
s0['mb'] = pd.cut(s0.map_any, [0,50,55,60,62,64,65,66,68,70,75,80,200])
print(s0.groupby('mb').agg(start=('ne', lambda x:(x>0).mean()), rate_if=('ne', lambda x: x[x>0].mean()), n=('ne','size')).round(3).to_string())
# examples
print(d.loc[1010000002][['ma','mc','ne','dne']].head(26).T.round(2).to_string())
```