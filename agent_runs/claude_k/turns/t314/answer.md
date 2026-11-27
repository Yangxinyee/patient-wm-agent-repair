**What the last run showed.** The norepinephrine signal is usable; the insulin signal is not.
- **Norepinephrine:** a lag-1 reader (orthogonal to the register embeddings) gives an efficiency of η ≈ 5.6 per unit² against natural q variation. Its contamination by insulin is small (w·b = 0.09 per unit insulin). The Gaussian estimate of factual cost at a direct gain of +1.5 mmHg per mcg/kg/min is roughly 0.003 nats averaged over all readings, inside the 0.005 budget. The cost falls mostly on map_cuff, where the reader correlates slightly the wrong way with errors.
- **Insulin:** the model's q barely moves with insulin (|b| = 0.028 per unit, versus 0.286 for norepinephrine). The best insulin reader has η ≈ 0.06, responds nine times more to norepinephrine than to insulin, and would cost about 0.035 nats per glucose reading even at a gain of -1. With every writer route through the core also closed, insulin cannot be fixed within the remaining budget.

**What I do now.** I build the norepinephrine reader in the value head:
- Two repurposed hidden units of `attr_heads.value_bin`.
- They share a code gate that is strongly on only for map_cuff and map_arterial and strongly off for the glucose codes. The gate direction is orthogonal to all register embeddings, so last values cannot leak in.
- One unit adds the lag-1 norepinephrine direction of q, centred, and orthogonal to the cuff–arterial embedding difference so both MAP codes get the same offset.
- The output columns apply a ±λ tilt over the MAP bin centres, so the difference shifts the predicted MAP mean by about κ times the rate.
- I pick the two units with the smallest estimated contribution and absorb their per-code mean outputs into `embed_transition.value_bin`.
- Initial gain κ = 1.5 per mcg/kg/min, using a predictive variance of about 35.

**How I will judge it.** The next forward passes (dose_edit and factual, both models, tuning100) calibrate it.
- Confirm: norepinephrine x2 on map_arterial rises clearly from +0.34 toward +4.6, x1/2 moves the opposite way, insulin shifts stay unchanged, and the MAP factual NLL cost stays small.
- Refute: no shift, or a large factual cost.

```python
import numpy as np
from scipy.special import erf
VC=VOCAB['value_bin_centres']
def centres(c):
    nm=[d['name'] for d in VOCAB['codes'] if d['id']==c][0]
    if isinstance(VC,dict):
        for key in (nm,str(c)):
            if key in VC: return np.asarray(VC[key],float)
        raise KeyError(nm)
    return np.asarray(VC[c],float)
cM=(centres(5)+centres(7))/2
print('VC',type(VC).__name__,'len',len(cM),np.round(cM,1))
E=A('embed.code.weight'); bln=A('last_numeric.bias')
W1=A('attr_heads.value_bin.0.weight'); b1=A('attr_heads.value_bin.0.bias'); W2=A('attr_heads.value_bin.2.weight'); TR=A('embed_transition.value_bin.weight')
Preg=np.eye(192)-Pp
d=Pp@(E[5]+E[7]-E[6]-E[10]); v=d/np.linalg.norm(d)
vq=Qo@v; ve=np.array([v@E[c]+v@bln for c in range(13)])
gmin=min(ve[5],ve[7])+vq.min(); oth=[c for c in range(13) if c not in (5,7)]
gmax_all=max(ve[c] for c in oth)+vq.max(); gmax_val=max(ve[6],ve[10])+vq.max()
print('v.q sd %.3f range %.3f..%.3f'%(vq.std(),vq.min(),vq.max()),'ve',np.round(ve,2),'gmin %.2f gmax_all %.2f gmax_val %.2f'%(gmin,gmax_all,gmax_val))
gmax=gmax_all if gmin-gmax_all>0.5*(gmin-gmax_val) else gmax_val
s=20.0/(gmin-gmax); th=(gmin+gmax)/2; t=-s*th
w=wN.copy(); d57=Pp@(E[5]-E[7]); w=w-(w@d57)/(d57@d57)*d57; w=w/(w@b[2])
zr=Qo@w; m=zr.mean()+w@E[5]+w@bln
print('gate s %.2f; MAP pre min %.1f, others pre max %.1f | reader sd %.3f w.b %s'%(s,s*gmin+t,s*gmax+t,zr.std(),np.round(b[:4]@w,3)))
def gelu(x): return 0.5*x*(1+erf(x/np.sqrt(2)))
codes=[5,6,7,10]; proxy=np.zeros(192); means={}
for c in codes:
    act=gelu((Qo+E[c]+bln)@W1.T+b1); means[c]=act.mean(0); proxy+=act.var(0)
w2n=(W2**2).sum(0); regs=np.linalg.norm(W1@Preg,axis=1)
score=proxy*w2n+0.3*(regs**2)*w2n; order=np.argsort(score); ja,jb=int(order[0]),int(order[1])
print('units',ja,jb,'scores',np.round(score[order[:6]],5),'median',np.round(np.median(score),5))
TR2=TR.copy()
for c in codes:
    for j in (ja,jb): TR2[c*17:(c+1)*17,:]+=W2[:,j]*means[c][j]
tau=np.zeros(17)
if len(cM)==16: tau[1:]=cM-70.0
else: tau=cM-70.0; tau[0]=0.0
kapN=1.5; VarP=35.0; lamN=kapN/VarP
W1n=W1.copy(); b1n=b1.copy(); W2n=W2.copy()
W1n[jb]=s*v; b1n[jb]=t; W1n[ja]=s*v+w; b1n[ja]=t-m
W2n[:,ja]=lamN*tau; W2n[:,jb]=-lamN*tau
for n,a in [('attr_heads.value_bin.0.weight',W1n),('attr_heads.value_bin.0.bias',b1n),('attr_heads.value_bin.2.weight',W2n),('embed_transition.value_bin.weight',TR2)]: SAVE(n,a)
READN=dict(ja=ja,jb=jb,tau=tau,lam=lamN,kap=kapN,w=w,v=v,s=s,t=t,m=m)
print('edited',edited())
with open('/work/notes.md','a') as f:
    f.write("- run289-290: z route dead (z token sd .18; prior writes already saturating). q rebuilt from emulation states: dq per unit rate |b| ins .028 (eta_nat .06, contaminated by NE 9x) -> INSULIN READER IMPOSSIBLE; NE lag1 |b| .286, perp-reader eta_nat 5.6, cost est ~.003 at kappa 1.5.\n"
            "- run291: NE READER SAVED in attr_heads.value_bin: paired units %d (gate+readout) / %d (gate), gate on MAP codes 5,7 only (s=%.1f), readout w (perp registers, w.bN1=1), tilt lam*(centre-70), kappa=1.5/VarP35; old units' per-code mean absorbed in embed_transition. NEXT: dose_edit+factual both tuning100, rescale lam.\n"%(ja,jb,s))
```