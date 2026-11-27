The decoder fitted on the head input fails, so the plain head-only edit is not viable as built.

- **Doubling-type arms:** the decoded change follows almost none of the true change (slope 0.09 for x2, 0.01 for x3, 0.34 for x1.5).
- **Dose cuts:** it tracks the change only partly (0.87 for x½, 0.59 for x0).
- **Factual rows:** R² is negative, with error sd 1.4 mg/dL.

The predictive sd is fairly uniform (median 12.3, 5–95% range 9.6–19.8), so the variance dependence of a logit tilt would be a minor issue.

There are two possible causes, and they lead to different fixes:
- **Lost dose information:** the state at the head barely carries the dose. That points to a core-level channel.
- **z noise or normalisation:** the head input adds a random prior-z term and a layer normalisation that may bury a signal that is present before them.

I'll separate the two by refitting the same decoder on three inputs: the pre-normalisation head base, the head input with z at its prior mean, and the head input with a random z. If the base or prior-mean versions show counterfactual slopes near 1, the information is there and the edit needs to read it more cleverly. If all three are poor, the dose has to be carried by a core channel.

```python
code_gen2 = code_gen.replace("out={k:[] for k in ['wc',","out={k:[] for k in ['wcm','base',").replace(
"out['wc'].append(wc.numpy().astype(np.float32));","out['wc'].append(wc.numpy().astype(np.float32)); out['base'].append(base.numpy().astype(np.float32)); out['wcm'].append(head_value_logits(base,codes,f8,lb_,torch.zeros(len(hj),dtype=torch.long),W,zmu,1)[1].numpy().astype(np.float32));")
code_gen2 = code_gen2.replace("'lb','P','pm0','val']","'lb','P','pm0','val','wc']")
exec(code_gen2)
Dw,res_w=gen_wc(RD,armsW,W,a_i,c_i,pf_np,seed=0)
for nm in ['base','wcm','wc']:
    for al in [0.01,0.1]:
        w,b0=ridge_fit(Dw[nm][glu&~te],r_all[glu&~te],al); rh=Dw[nm]@w+b0; m0=glu&te&(Dw['arm']==0)
        print(nm,al,'R2',round(R2(r_all[glu&te],rh[glu&te]),3),'factErr',round(float(np.std(rh[m0]-r_all[m0])),2),'cf',pair_eval(rh,glu&te))
```