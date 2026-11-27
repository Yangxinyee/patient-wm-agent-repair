Found it: my bytecode-printing loop used `ln` as its loop variable, which overwrote the replica's LayerNorm helper `ln` with an integer. To restore it, I need its original calling convention. I'll print how `encode` calls `ln` (the arguments loaded before each call) and the signature of the similar helper `lnp`. Then I'll redefine `ln` to match.

```python
print('lnp sig', inspect.signature(lnp), 'names', lnp.__code__.co_names, 'consts', [str(c)[:20] for c in lnp.__code__.co_consts])
ins_l=list(dis.get_instructions(encode))
for i_,x_ in enumerate(ins_l):
    if x_.opname=='LOAD_GLOBAL' and x_.argval=='ln':
        seq=[]
        for y_ in ins_l[i_+1:i_+12]:
            if y_.opname.startswith('CALL'): seq.append(f'CALL{y_.arg}'); break
            seq.append(f'{y_.opname[:10]}:{str(y_.argval)[:22]}')
        print('ln call:', ' | '.join(seq)[:300])
```