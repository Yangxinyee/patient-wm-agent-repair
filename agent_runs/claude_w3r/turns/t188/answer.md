Run 175 shows the local core-step functions read weights from the dict `Wt`, and the head functions read from `W`. So I can simulate a weight edit locally by temporarily changing `Wt` or `W` and re-running steps. The planned channel route (u token → prelude → E → b_in → h → out → glucose head) depends on exact details: which tensor is the recurrent state, the coda, the time decay, and how the head reads h. My notes conflict on whether the recurrent h is post-coda. Before designing gains I'll re-read the architecture spec. If it says the state is post-coda y, an injected h component passes through c_out and the coda every step, which changes how I control its persistence.

```python
print(open('/model/ARCHITECTURE.md').read())
```