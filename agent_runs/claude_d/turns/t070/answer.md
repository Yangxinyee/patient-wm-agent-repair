The tensor list gives a natural edit site. `z_post.0` reads the concatenation [ctx (192), e_k (192), u_k (192)], so the posterior latent sees both the pre-bundle state and the action. It is the one place where a dose relative to what the history implies (the dose surprise) can be represented. `z_in` (192×32) maps z into the prelude token and is used only in the state update. The heads use z drawn from the prior, through `z_head`.

My hypothesis is that an edit of `z_in` along a z-direction that encodes the dose surprise can strengthen the state's response to doses without biasing factual predictions. To test this I need states on many insulin and norepinephrine bundles. Before spending a forward pass I'll check the kernel's training panels (EPS, BASE, INS, NOR) and the event layout of one stay, so I can choose informative stays.

```python
for nm_ in ['EPS','BASE','INS','NOR','GIVEN','GF','GL','MA','Gdec']:
    v_=globals()[nm_]; print(nm_, getattr(v_,'shape',type(v_)), np.nanmean(v_).round(3) if hasattr(v_,'shape') else '')
print(type(stays), len(stays), list(stays[:3]))
print(df.shape, df.columns.tolist())
s0=df.stay_id.iloc[0]; print(df[df.stay_id==s0].head(12).to_string()[:1500])
```