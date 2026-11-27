The kernel kept its state: I have the train grid `gg`, the structural impulse estimates (`Kins`, `Knor`), the states from fwd_3, and many helper functions. There is no input history, so I cannot read back my earlier code.

Before spending more runs on an exact re-implementation of the value head, I want to re-read the architecture with the edit design in mind. The design question is this. The true insulin effect builds up and wears off over about 10 hours, with a lag-1 effect of −2.0 per U/h and a cumulative −13.7. So the model needs a memory of the dose history that its value head can read. The original model's shift instead grows roughly linearly.

The parts of the architecture that decide this are:
- exactly what the "last_numeric" and "last rate bin" head inputs refer to. Is it the last event of the predicted code, or the last event of any code (for example the insulin infusion just before a glucose reading)?
- how h is carried and decayed between bundles (a_bar, c_out, coda).

If the head reads the most recent drug event directly, a lag-1 correction is easy, but the longer lags must still come from state memory. If not, all dose information must come from h.

I also print my unsaved note fragment `notes_add` and try to recover the source of my last head functions.

```python
import inspect
txt = open('/model/ARCHITECTURE.md').read()
print(len(txt))
print(txt[:14000])
print('=== notes_add ===')
print(notes_add[:2300])
for fnm in ['ev2', 'run13']:
    try:
        print('=== src', fnm, '==='); print(inspect.getsource(globals()[fnm])[:2500])
    except Exception as ex:
        print('no source', fnm, ex)
```