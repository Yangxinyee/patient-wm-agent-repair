Run 236 shows that `step` does build a z token through `z_in`, but changing z had exactly no effect. Either the z token is dropped somewhere in the replica's prelude, or my perturbation was handled wrongly. Next I will try a large change to z (+10 in every dimension) and print `step`'s local variable names to see how the tokens are assembled. If even a large change has no effect, the replica drops the z token. In that case z's real role must be checked against the tool before relying on it.

```python
r3=step(o_,h_,dts,e_,u_,z0+10.0,Mem,ages); print('big z diff',[float((a-b).abs().max()) for a,b in zip(r1,r3)])
print(step.__code__.co_varnames)
```