Glucose_lab gives the same impulse response free of selection: -2.08, -2.45, -2.24, -1.52, -1.44, -1.02, -0.65, -0.43, then about 0, with a cumulative -12.3. This matches the kernel K(k) = -2·k·0.6^(k-1), whose total is -12.5. Same-hour glucose_finger and glucose_lab readings agree on average (difference -0.02, SD 9). They measure the same latent glucose with independent noise, and the drug acts on that latent glucose.

So for insulin I now believe:
- One unit/h for one infusion-hour lowers glucose at k hours later by about 2·k·0.6^(k-1) mg/dL.
- The effect is linear in dose and roughly homogeneous, and it is the same on glucose_finger and glucose_lab.

Next is norepinephrine. Its rate looked like titration: on 39% of the time at MAP 70-75 and high rates even at high MAP. I need its policy before I can find exogenous variation. I will tabulate the norepinephrine rate against the previous hour's rate and the current map_arterial and map_cuff readings: start and stop rules, step sizes, and which MAP drives it. If the rate is a deterministic titration of the previous rate plus a function of MAP with i.i.d. noise, I can reuse the surprise approach. If it mixes in something unobserved, I would need another source of variation.

```python
P['N_prev'] = g['N'].shift(1); P['N0']=P.N.fillna(0.0); P['N0_prev']=g['N0'].shift(1)
P['MAP_now'] = P.MA.fillna(P.MC)
z = P[(P.h>=0)]
print("NE on given prev on/off and MA available:")
z2 = z.copy(); z2['prevOn']=(z2.N0_prev>0).astype(int)
print(z2.groupby(['prevOn', pd.cut(z2.MA,[0,55,60,65,70,75,80,90,200])]).agg(n=('anyN','size'), pN=('anyN','mean'), meanN=('N','mean')).to_string())
print("when MA missing, by MC:")
z3 = z2[z2.MA.isna()]
print(z3.groupby(['prevOn', pd.cut(z3.MC,[0,55,60,65,70,75,80,90,200])]).agg(n=('anyN','size'), pN=('anyN','mean'), meanN=('N','mean')).to_string())
print("when both MAP missing:", z2[z2.MA.isna()&z2.MC.isna()].groupby('prevOn').anyN.agg(['size','mean']))
```