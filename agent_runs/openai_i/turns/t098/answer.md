The lag estimates suggest a much larger cumulative effect than the original rollouts predict: roughly −11 glucose units for doubled insulin and +3.8 MAP units for doubled norepinephrine on the previously examined subset. These are record-based estimates, not known counterfactual truth. I’ll inspect the exact encoder and posterior code next; reproducing exposed states would justify using those features to compute a targeted correction.

```python
import inspect
import dis
from pathlib import Path

def show90(name, maxlines=65):
    fn = globals().get(name)
    if not callable(fn):
        return
    print("\n===", name, "===")
    try:
        print(inspect.getsource(fn)[:3600])
    except (OSError, TypeError):
        print("arguments:", fn.__code__.co_varnames[:fn.__code__.co_argcount])
        print("constants:", fn.__code__.co_consts)
        lines = dis.Bytecode(fn).dis().splitlines()
        print("\n".join(lines[:maxlines]))
        if len(lines) > maxlines:
            print("[remaining disassembly omitted:", len(lines) - maxlines, "lines]")

show90("complete_atom57", 55)
show90("repair_atom_base", 75)
show90("posterior84", 45)
show90("pooled84", 45)

print("\n=== LATENT / RECURRENT TENSORS ===")
for n90 in NAMES:
    if any(s90 in n90.lower() for s90 in ("post", "prior", "latent", "z_in")):
        print(n90, SHAPES[n90])

print("\n=== EARLIER EXACT RECONSTRUCTION CONVENTIONS ===")
txt90 = Path("/work/notes.md").read_text()
for label90 in ("Python 57", "Python 58", "Python 59", "Python 60", "Python 30"):
    at90 = txt90.find("## " + label90)
    if at90 >= 0:
        end90 = txt90.find("\n## ", at90 + 4)
        print(txt90[at90:end90 if end90 >= 0 else at90 + 800][:850])

with open("/work/notes.md", "a") as f90:
    f90.write(
        "\n## Python 90; forward 6 — distinguish proximal and cumulative targets\n"
        "- Controlled-lag convolution previously suggested insulin x2 about -11 and NE x2 about +3.8 on a selected subset.\n"
        "- These are observationally identified estimates, not supplied intervention truth.\n"
        "- Inspecting atom/posterior implementations and earlier exact conventions; no weights changed.\n"
    )
```