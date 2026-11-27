`ACTMAP` already marks the gap MLP and both encoders as GELU, while everything else defaults to SiLU. That is the likely culprit: `z_post`, and probably every MLP in the model, uses GELU. My probe ran in the near-linear regime (bias 30), so it could not reveal this.

I'll test `z_post` with exact GELU and with tanh-GELU, keeping ctx as before. If z then matches to about 1e-5, ctx is confirmed. I'll then make GELU the default and verify the full step.

```python
for act_name in ['gelu','gelu_tanh','silu']:
    ACTMAP['z_post']=act_name
    er=np.concatenate([zcheck2(s,gaps['ls/10'],1.0,True) for s in stays[:3]])
    print('z_post act',act_name,'median %.2e max %.2e'%(np.median(er),er.max()))
ACTMAP['z_post']='gelu'
```