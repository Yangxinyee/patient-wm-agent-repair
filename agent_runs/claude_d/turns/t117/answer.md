The head is close but not exact. My predicted means correlate 0.88–0.97 with the tool's, the prior spread of z is moderate (sd about 0.6), and z_head contributes little. The gap embedding has the largest norm of any view term (16), so its input scaling is the prime suspect: the text writes it as log dt/10 for the context and log dt for the view. I'll grid the view's gap scaling, including seconds versus hours, jointly with the type index. A combination with MAE far below 9 would settle both.

```python
res=[]
for nm,gs,unit in [('log_s/10',10.0,1.0),('log_s',1.0,1.0),('log_h/10',10.0,3600.0),('log_h',1.0,3600.0)]:
    df2=gfm.copy(); 
    for t in range(15):
        i=df2.prev.values; dt=df2.dt_s.values
        # monkeypatch gap input via gs and unit: view uses log(dt/unit)/gs
        m_,_=head(df2.assign(dt_s=df2.dt_s), t, gs=gs) if unit==1.0 else (None,None)
        if m_ is None: continue
        res.append((nm,t,round(np.mean(np.abs(m_-pm)),2)))
res=sorted(res,key=lambda r:r[2]); print(res[:6])
```