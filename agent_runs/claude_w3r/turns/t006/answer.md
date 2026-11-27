MAP arterial is recorded in only 48% of stays, but nearly every hour in those stays. Norepinephrine is given in 44% of stays and insulin in 61%. Insulin use rises over the day. Glucose finger is measured in about 55% of stay-hours and glucose lab in 20%.

Next I will look at the insulin dosing policy. I will compare the insulin rate at hour t with the last known glucose_finger (and glucose_lab) up to hour t and with the previous hour's rate. The aim is to see how deterministic the policy is: how much of the rate variance the observed history explains, and what the residual looks like.

```python
def ffill_last(X):
    # last observed value up to and including each hour, and its age in hours
    n,T = X.shape; last = np.full((n,T),np.nan); age = np.full((n,T),np.nan)
    cur = np.full(n,np.nan); cage = np.full(n,np.nan)
    for t in range(T):
        obs = ~np.isnan(X[:,t]); cur = np.where(obs, X[:,t], cur); cage = np.where(obs, 0, cage+1)
        last[:,t]=cur; age[:,t]=cage
    return last, age
gf_last, gf_age = ffill_last(A['glucose_finger']); gl_last, gl_age = ffill_last(A['glucose_lab'])
ins_r = np.nan_to_num(A['insulin'], nan=0.0); ne_r = np.nan_to_num(A['norepinephrine'], nan=0.0)
# insulin hours 0..23 (cols 1..24) vs last glucose finger at that hour
rows=[]
for t in range(1,25):
    rows.append(pd.DataFrame({'t':t, 'g':gf_last[:,t], 'gage':gf_age[:,t], 'gobs':~np.isnan(A['glucose_finger'][:,t]),
                              'r':ins_r[:,t], 'rprev':ins_r[:,t-1], 'gl':gl_last[:,t]}))
D = pd.concat(rows); D['gbin']=pd.cut(D.g,[0,100,120,140,160,180,200,220,250,300,400])
print(D.groupby('gbin').agg(n=('r','size'), p_on=('r',lambda x:(x>0).mean()), mean_r=('r','mean')).round(3).to_string())
print(pd.crosstab(D.rprev.clip(0,5), D.r.clip(0,5), normalize='index').round(2).to_string())
```