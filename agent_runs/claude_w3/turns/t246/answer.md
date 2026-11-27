The tensor list gives every reader of the state: the prelude, block and coda norms; `out_norm`, which feeds `z_prior`/`z_post` through ctx; and `query_norm`, which feeds `value_bin.0`. Next I'll score each dimension's downstream influence. The score is its spread across bundles times the norm gamma and the next layer's column norm, summed over readers, plus the linear reads through `c_out` and `b_in`. If a few dimensions score far below the median, they are candidates to free up. If all scores are similar, I'll need another route.

```python
def W(n): return load(n).numpy()
def cn(M,j): return np.linalg.norm(M[:,j])
P='core.'; S=np.zeros(192); parts={}
rd=[('prelude.0.norm1',['prelude.0.attn.q','prelude.0.attn.kv']),('prelude.0.norm3',['prelude.0.ffn.0'])]
for b in ['blocks.0.0','blocks.0.1']:
    rd+= [(b+'.norm1',[b+'.attn.q',b+'.attn.kv']),(b+'.norm2',[b+'.cross.q']),(b+'.norm3',[b+'.ffn.0'])]
rd+=[('coda.0.norm1',['coda.0.attn.q','coda.0.attn.kv']),('coda.0.norm3',['coda.0.ffn.0'])]
ln=np.zeros(192)
for g,ws in rd:
    gam=W(P+g+'.weight')
    M=np.concatenate([W(P+w+'.weight') for w in ws],0)
    ln+=np.abs(gam)*np.linalg.norm(M,axis=0)
co=np.linalg.norm(W(P+'c_out.weight'),axis=0); bi=np.linalg.norm(W(P+'b_in.weight'),axis=0)
hd=np.abs(W(P+'out_norm.weight'))*np.abs(W('query_norm.weight'))*np.linalg.norm(W('attr_heads.value_bin.0.weight'),axis=0)
score=sdm*(ln/65+co)
o=np.argsort(score)
print('score q',np.round(np.quantile(score,[0,.05,.5]),3))
for j in o[:8]: print(j,'sd %.2f ln %.2f co %.2f bi %.2f hd %.2f sc %.3f'%(sdm[j],ln[j],co[j],bi[j],hd[j],score[j]))
```