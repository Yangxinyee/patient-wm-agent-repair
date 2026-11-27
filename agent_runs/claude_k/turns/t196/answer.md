A linear function of ctx predicts the dose fairly well at drug bundles:
- insulin rate R² 0.70;
- norepinephrine rate R² 0.58;
- previous norepinephrine rate R² 0.60.

So a ctx-based baseline is usable, especially for insulin. There's a catch: because ctx also remembers past doses, the baseline would partly absorb an edit's effect. Whatever attenuation that causes can be calibrated with dose_edit.

The more important unknown is whether the new channel works at all. Is there a z direction the model uses for "this glucose reading came in lower/higher than predicted"? If so, would a shift along it at a drug bundle move the next glucose prediction strongly?

I'll estimate those directions from the 99 stays with tool predictions. At each reading bundle, I'll regress the posterior-minus-prior z shift on the surprise of each reading type (reading minus predicted mean). If R² is high and the glucose direction is distinct from the MAP direction, they become the injection directions.

```python
print('pmall vs tool? corr with vals', np.round(np.corrcoef(pmall,vals99)[0,1],3), 'mean abs diff', np.round(np.abs(pmall-vals99).mean(),2))
ZP_prior=np.asarray(view_from(X8o[kidx], dts99)[2])[:,:32]
kb=kidx+1
ok_b=(kb<len(sid8))&(sid8[np.minimum(kb,len(sid8)-1)]==sid8[kidx])&(np.abs(hr8[np.minimum(kb,len(sid8)-1)]-hr)<1e-6)
print('readings whose next bundle is the reading bundle',ok_b.mean())
bund=np.unique(kb[ok_b])
cmap_={6:0,10:1,7:2,5:3}
Sur=np.zeros((len(bund),4)); Pres=np.zeros((len(bund),4)); bpos={b:i for i,b in enumerate(bund)}; prior_b=np.zeros((len(bund),32))
for r__ in np.nonzero(ok_b)[0]:
    i__=bpos[kb[r__]]; c__=int(c99[r__])
    if c__ in cmap_: Sur[i__,cmap_[c__]]=vals99[r__]-pmall[r__]; Pres[i__,cmap_[c__]]=1
    prior_b[i__]=ZP_prior[r__]
dZ=X8o[bund,960:992]-prior_b
Xr_=np.c_[np.ones(len(bund)),Pres,Sur]
Bz,_,_,_=np.linalg.lstsq(Xr_,dZ,rcond=None); res_=dZ-Xr_@Bz
print('R2 of z innovation on surprises (pooled)',np.round(1-(res_**2).sum()/((dZ-dZ.mean(0))**2).sum(),3))
vG=Bz[5]; vGl=Bz[6]; vMa=Bz[7]; vMc=Bz[8]
for nm_,v_ in [('gf',vG),('gl',vGl),('ma',vMa),('mc',vMc)]: print(nm_,'|v| per unit',np.round(np.linalg.norm(v_),4),'top dims',np.argsort(-np.abs(v_))[:5])
print('cos gf-gl',np.round(vG@vGl/np.linalg.norm(vG)/np.linalg.norm(vGl),3),'cos gf-ma',np.round(vG@vMa/np.linalg.norm(vG)/np.linalg.norm(vMa),3),'cos ma-mc',np.round(vMa@vMc/np.linalg.norm(vMa)/np.linalg.norm(vMc),3))
```