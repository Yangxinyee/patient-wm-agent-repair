Norepinephrine is titrated: r_t ≈ 0.25 + 0.85·r_{t−1} − 0.003·MAP + …, and the residual after conditioning on MAP has lag-1 autocorrelation 0.83. The identifying variation is therefore the titration innovation, i.e. the change in rate beyond what MAP and the previous rate predict, not the rate level.

Next I will characterise the titration rule. I will look at Δr = r_t − r_{t−1} against the current MAP among running infusions, and check how start/stop and the cap at 1.0 behave, to get the right model for E[r_t | MAP_t, r_{t−1}]. A clean rule plus an i.i.d. innovation would give a valid instrument, as for insulin.

```python
on = art[(art.rprev>0)].copy(); on['dr']=on.r-on.rprev
print(on.groupby('m5').agg(n=('dr','size'),mean_dr=('dr','mean'),sd_dr=('dr','std'),p_stop=('r',lambda x:(x==0).mean()),
      p_cap=('r',lambda x:(x>=0.999).mean())).round(3).to_string())
# start behaviour: rprev==0
st = art[(art.rprev==0)]
print(st.groupby('m5').agg(n=('r','size'),p_start=('r',lambda x:(x>0).mean()),mean_r_if_start=('r',lambda x:x[x>0].mean())).round(3).to_string())
```