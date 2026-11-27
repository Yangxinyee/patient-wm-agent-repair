The b_in route works in the linear analysis.
- **Signal:** the best combination of the 48 candidate directions gives a whitened SNR of 162 per unit rate with only 0.88 RMS disturbance. At the scale actually needed (SNR about 10) that is roughly 0.05 RMS, which is negligible.
- **Persistence:** the response is larger at readings long after the last drug bundle. The channel seems to integrate past doses rather than forget them quickly.

That matters for the design. The true effect kernel decays over roughly 5–10 hours, so I need the channel's impulse response before I can fit the head readout.

Using the stored responses I'll compute the Fisher-decoded channel value at each reading and regress it on the drug-I rates summed in lag bins (0–1 h, 1–2 h, …, over 24 h). The lag-bin coefficients give the channel's impulse response.
- **Usable:** a decay over a few hours.
- **Needs a decay mechanism:** flat coefficients out to 24 h, meaning a pure integrator.

```python
g2=S2i@J2@t_; d2=(DF2@t_)@g2
LB=[0,1.01,2.01,3.01,4.01,6.01,9.01,12.01,24.01,1e9]; Fl=[]
for s in ST:
    B=BUN[s]; mins=np.array([b[0] for b in B],float); sg=info[s]['sig']; rm=RTs[s]['minute'].values; pr=info[s]['pr']
    for i in range(len(rm)):
        js=np.where((sg[:pr[i]+1]>0))[0]; lag=(rm[i]-mins[js])/60.; f=np.zeros(len(LB)-1)
        for a in range(len(LB)-1): f[a]=sg[js][(lag>LB[a])&(lag<=LB[a+1])].sum() if len(js) else 0
        Fl.append(f)
Fl=np.array(Fl); coef,res,rk,sv=np.linalg.lstsq(Fl,d2,rcond=None); pred=Fl@coef
print('R2',round(1-((d2-pred)**2).sum()/((d2-d2.mean())**2).sum(),3),'coef/first',np.round(coef/coef[0],2),'first',round(float(coef[0]),2))
```