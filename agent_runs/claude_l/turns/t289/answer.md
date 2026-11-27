Run 265 showed the injection tool (`bstep_z`, where z token = zfn(ctx, e, u, info)) and many fitted objects in the kernel: direct-lag coefficient tuples (`Cdir_*`), decoders (`DECS`), finite-difference directions (`DIRS`) and Jacobian-derived vectors (`dstar`, `wdec`, `Vg`). The small string variables (`com`, `tag`, `src_`) may hold notes from the undocumented runs. Next I will print those strings and the key coefficient arrays, in compact form. If `Cdir_*_12` holds the 12-lag residual coefficients and `dstar`/`wdec` are an injection direction and decoder, Stage 1 was largely built and I can calibrate it. If not, I rebuild it from `tf_train.npz`.

```python
for k in ['com','tag','scol','src_','AE','ae']:
    v=globals()[k]; print(k,'=',repr(v)[:300])
for k in ['Cdir_ins_12','Cdir_ne_12','Cdir_ins_1','Cdir_ne_1']:
    a,b=globals()[k]; print(k,np.round(np.ravel(a)[:14],2),'|',np.round(np.ravel(b)[:14],2))
k0=list(DIRS)[0]; print('DIRS',k0,desc(DIRS[k0])); k1=list(DECS)[0]; print('DECS',k1,desc(DECS[k1]))
print('dstar',np.linalg.norm(dstar).round(3),'wdec',np.linalg.norm(wdec).round(3),'Vg colnorm',np.linalg.norm(Vg,axis=0).round(3))
print('g_k',np.round(np.percentile(g_k,[5,25,50,75,95]),3),'rh',np.round(np.percentile(rh,[5,50,95]),3),'dx',np.linalg.norm(dx).round(3))
```