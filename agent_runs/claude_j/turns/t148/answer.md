`encode` calls `ln(mlp(s_, p+'.mlp2'), …)`, but my printout stopped at the inner `mlp` call. I need the rest of the arguments to recreate `ln`. Most likely it's `ln(x, p+'.norm')`, the same as `lnp(x, name)`, which uses `W[name+'.weight']` / `W[name+'.bias']`. I'll print the instructions up to the second call and check which weight dict `ln` should read (`W` or `Wt`).

```python
for i_,x_ in enumerate(ins_l):
    if x_.opname=='LOAD_GLOBAL' and x_.argval=='ln':
        seq=[];nc=0
        for y_ in ins_l[i_+1:i_+20]:
            seq.append(f'{y_.opname[:9]}:{str(y_.argval)[:18]}')
            if y_.opname.startswith('CALL'):
                nc+=1
                if nc==2: break
        print(' | '.join(seq)[:420])
print('Wt keys sample', list(Wt.keys())[:2], type(list(Wt.values())[0]).__name__, list(Wt.values())[0].dtype)
```