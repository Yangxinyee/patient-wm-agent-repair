Run 233 shows the rate read-outs are adequate on realistic rates, including every scaled-dose multiplier I need.
- **Insulin:** rms error about 0.17–0.19 with negligible bias at ×0 to ×2. Only ×3 degrades (0.5), because very high rates saturate in the encoder.
- **Norepinephrine:** rms about 0.03–0.04 at every multiplier.

I'll use the moderate-ridge version (λ = 1e-4) to keep the weight norms reasonable.

**Next: build the writer and save it.** I start from the original z_post weights, which discards the old unit-55/109 edit. Units 12, 88 and 105 become:
- **baseline (12):** a gate unit;
- **insulin (88):** gate + (decoded rate − ctx-predicted rate);
- **norepinephrine (105):** the same for its rate.

Their output difference adds λ·innovation along the glucose-surprise or MAP-surprise z directions. It is exactly zero on non-drug bundles. The starting gains are λ_A = −0.18 and λ_B = 0.6, guessed from the natural surprise relation; a factual comparison will calibrate them.

Before saving, I check offline that the z shift equals λ·innovation on drug bundles and is about zero on reading bundles.

```python
pA=PD[('Ar',1e-4)]; pB=PD[('Br',1e-4)]; pg=PD[('g',1e-4)]
wA,bA_=dec['ins']; wB,bB_=dec['ne']
dA=vG/np.linalg.norm(vG); dB=vM/np.linalg.norm(vM)
def build_writer(lamA,lamB,j0=12,jA=88,jB=105,B0=20.0,G=40.0,sA=1.0,sB=10.0):
    W0n=load_original('z_post.0.weight').numpy().astype(np.float64).copy(); b0n=load_original('z_post.0.bias').numpy().astype(np.float64).copy()
    W2n=load_original('z_post.2.weight').numpy().astype(np.float64).copy(); b2n=load_original('z_post.2.bias').numpy().astype(np.float64).copy()
    for j in (j0,jA,jB): W0n[j]=0.0
    W0n[j0,384:]=G*pg; b0n[j0]=B0-G
    W0n[jA,:192]=-sA*wA; W0n[jA,384:]=G*pg+sA*pA; b0n[jA]=B0-G-sA*bA_
    W0n[jB,:192]=-sB*wB; W0n[jB,384:]=G*pg+sB*pB; b0n[jB]=B0-G-sB*bB_
    W2n[:,[j0,jA,jB]]=0.0
    W2n[:32,jA]=lamA/sA*dA; W2n[:32,jB]=lamB/sB*dB; W2n[:32,j0]=-(lamA/sA)*dA-(lamB/sB)*dB
    return W0n,b0n,W2n,b2n
def zmean(inp,W0n,b0n,W2n,b2n): return gelu_np(inp@W0n.T+b0n)@W2n[:32].T+b2n[:32]
lamA0,lamB0=-0.18,0.6
Wn=build_writer(lamA0,lamB0)
zo_D=zmean(inpD,Wz0,bz0,Wz2,bz2); ze_D=zmean(inpD,*Wn); dzD=ze_D-zo_D
innA_s=riA[sub_]-(CTA[sub_]@wA+bA_); innB_s=rnA[sub_]-(CTA[sub_]@wB+bB_)
predz=lamA0*innA_s[:,None]*dA+lamB0*innB_s[:,None]*dB
print('drug bundles: max|dz-pred| %.4f, sd(dz.dA) %.3f, corr with innovA %.3f'%(np.abs(dzD-predz).max(),(dzD@dA).std(),np.corrcoef(dzD@dA,innA_s)[0,1]))
zo_R=zmean(inpR,Wz0,bz0,Wz2,bz2); ze_R=zmean(inpR,*Wn); print('reading bundles: max|dz| %.2e'%np.abs(ze_R-zo_R).max())
preE=inpD@Wn[0].T+Wn[1]; print('drug-bundle pre ranges: j0 [%.1f,%.1f] jA [%.1f,%.1f] jB [%.1f,%.1f]'%(preE[:,12].min(),preE[:,12].max(),preE[:,88].min(),preE[:,88].max(),preE[:,105].min(),preE[:,105].max()))
for n_,arr_ in zip(['z_post.0.weight','z_post.0.bias','z_post.2.weight','z_post.2.bias'],Wn): save_edit(n_,torch.tensor(arr_,dtype=torch.float32))
print('edited',edited())
open('/work/notes.md','a').write("\n- runs 205-233: c_out not identity, abar~.35: no linear SSM channel. Drug bundles at h+0.083 (action-only); END atoms are encoded in e (attrs none). Readings never in drug bundles.\n- vG/vM = z-surprise directions (corr .76/.58 with glucose/MAP surprise); next-pred gain .54/.66.\n- ctx decoders of rate: ins cvR2 .70 (innov sd .66), ne .58 (sd .23) saved /work/decoders.npz.\n- model factual err ~ ctx-innovation lags: gluc_finger -2.04,-1.43,~0; map_art +4.05,+.76.\n- dead z_post units 12,88,105 (max pre <-4.4). WRITER v2: j0=12 gate, jA=88 ins innov, jB=105 ne innov (u-decoders pg,pA,pB fitted on realistic+edited rates, lam1e-4), dz=lam*innov*dir. saved lamA=-.18 lamB=.6.\n")
```