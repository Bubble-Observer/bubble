# Tests

The default pytest collection is the only active suite — there is no hidden
legacy test lane. A test's retention depends on the live contract it protects,
not on its age: migration, replay, idempotency, conflict and historical-defect
counterexamples remain active regression tests. Tests are removed only in the
same change that explicitly retires both the implementation and its contract.

```powershell
$env:PYTHONPATH=(Resolve-Path .\src).Path
python -m pytest -q --import-mode=importlib --no-cov
```

The suite covers the world-agent composition root, the world store, Graph Shell,
the local Console, channels, acquisition security, ASR subprocess lifecycle,
the offline demo, and evaluation scenarios. Tests do not call the network or a
paid model.
