"""Substitut minimal de Streamlit pour exécuter les pages en test (enregistre les appels)."""
from __future__ import annotations

import types
from contextlib import contextmanager


class _Recorder:
    def __init__(self, log, buttons):
        self._log = log
        self._buttons = buttons

    def _rec(self, name, *a, **k):
        self._log.append((name, a, k))

    # affichage
    def __getattr__(self, name):
        if name in {"title", "caption", "warning", "info", "error", "success", "write", "markdown", "subheader",
                    "json", "dataframe", "metric", "exception", "set_page_config", "download_button"}:
            return lambda *a, **k: self._rec(name, *a, **k)
        raise AttributeError(name)

    def columns(self, n):
        n = n if isinstance(n, int) else len(n)
        return [self for _ in range(n)]

    def tabs(self, labels):
        return [self._ctx() for _ in labels]

    @contextmanager
    def _cm(self):
        yield self

    def _ctx(self):
        rec = self

        class C:
            def __enter__(self_inner):
                return rec

            def __exit__(self_inner, *exc):
                return False
        return C()

    def expander(self, *a, **k):
        return self._ctx()

    def spinner(self, *a, **k):
        return self._ctx()

    # widgets
    def date_input(self, label, value=None, **k):
        return value

    def checkbox(self, label, value=False, **k):
        return value

    def button(self, label, **k):
        self._rec("button", label)
        return any(b in label for b in self._buttons)

    def selectbox(self, label, options, format_func=None, **k):
        opts = list(options)
        if format_func:
            [format_func(o) for o in opts]
        return opts[0]

    def progress(self, *a, **k):
        return types.SimpleNamespace(progress=lambda *a, **k: None)


def make_fake(buttons=("Analyser les courses",)):
    log = []
    rec = _Recorder(log, buttons)
    mod = types.ModuleType("streamlit")
    for name in ["title", "caption", "warning", "info", "error", "success", "write", "markdown", "subheader",
                 "json", "dataframe", "metric", "exception", "set_page_config", "download_button"]:
        setattr(mod, name, getattr(rec, name))
    for name in ["columns", "tabs", "expander", "spinner", "date_input", "checkbox", "button", "selectbox", "progress"]:
        setattr(mod, name, getattr(rec, name))

    def cache(*dargs, **dk):
        def deco(f):
            f.clear = lambda: None
            return f
        if dargs and callable(dargs[0]):
            return deco(dargs[0])
        return deco

    class _Stop(Exception):
        pass

    def stop():
        raise _Stop()

    mod.stop = stop
    mod.StopException = _Stop
    mod.cache_data = cache
    mod.cache_resource = cache
    mod.session_state = {}
    mod.secrets = {}
    mod._log = log
    return mod
