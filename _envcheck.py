# -*- coding: utf-8 -*-
import sys, importlib
print("exe   ", sys.executable)
print("python", sys.version.split()[0])
for m in ["numpy", "pandas", "sklearn", "scipy", "matplotlib", "seaborn", "pybaselines", "joblib"]:
    try:
        mod = importlib.import_module(m)
        print(f"OK   {m:14s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"MISS {m:14s} {type(e).__name__}")
