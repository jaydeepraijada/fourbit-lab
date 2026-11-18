"""Modal chores through one entry point. Set MODAL_PROFILE yourself if you use more than one Modal profile.

    python3 mo.py modal <any modal args>          e.g. modal deploy real.py
    python3 mo.py spawn <function> k=v k=v ...    start a deployed function and return at once
    python3 mo.py call <function> k=v ...         run a deployed function and print what it returns

The app is fourbit-tinystories by default (train.py); set FOURBIT_APP=fourbit-real for the functions in real.py.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.environ.get("FOURBIT_APP", "fourbit-tinystories")


def conv(v):
    try:
        return json.loads(v)
    except Exception:
        return v


if __name__ == "__main__":
    what, rest = sys.argv[1], sys.argv[2:]
    if what == "modal":
        sys.exit(subprocess.run(["modal"] + rest, cwd=HERE).returncode)
    import modal
    fn = modal.Function.from_name(APP, rest[0])
    kw = {a.split("=", 1)[0]: conv(a.split("=", 1)[1]) for a in rest[1:]}
    if what == "spawn":
        print(fn.spawn(**kw).object_id)
    else:
        r = fn.remote(**kw)
        print(r if isinstance(r, str) else json.dumps(r, indent=1))
