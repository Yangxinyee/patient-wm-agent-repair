The interfaces are clear. bundles_of_stay(df, RU0) builds bundles from a stay's records, run_stay(B) runs the exact core replica, and q_of(prev_state, dt) gives the head query. Next I will inspect the outputs for one train stay so I can align readings with the preceding state. Then I can compare recorded versus insulin-x2 records directly.

```python
s0=selT[0]; df0=G3[s0]; B0x=bundles_of_stay(df0,RU0); St0x=run_stay(B0x)
print(type(B0x).__name__,len(B0x), type(B0x[0]).__name__, (list(B0x[0].keys()) if isinstance(B0x[0],dict) else len(B0x[0])))
print(type(St0x).__name__, getattr(St0x,'shape',None) or [getattr(a,'shape',None) for a in St0x][:4])
```