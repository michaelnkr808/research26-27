"""Attach the check 2 step logger to transformers.Trainer (PLAN.md, Amendment 2026-11-20 (b)).

Python imports this file at start-up because PYTHONPATH points at this folder. It reaches
every Python process of the run (the torch.distributed.run parent, the training process,
data loader workers, torch.compile workers), so it is lazy and guarded:

* it imports neither torch nor transformers here;
* it does nothing unless LOCAL_RANK is set (torch.distributed.run sets it in its worker,
  the training process) and CHECK2_HOOK_DISABLE is not set;
* otherwise it only puts an import finder at the front of sys.meta_path. The first time
  transformers.trainer is imported, the finder lets the normal machinery load it and
  then wraps Trainer.__init__ so that, after the original __init__ returns, one
  Check2StepLogger is added with add_callback. Processes that never import
  transformers.trainer (the parent, compile workers) are not touched.

If anything here fails it prints a line starting with "[check2]" to stderr and leaves
the import alone; RUN.md stops the trial if check2_steps.jsonl does not appear.
"""

import os
import sys

_TARGET = "transformers.trainer"
_MARK = "_check2_patched"


def _log(msg):
    try:
        sys.stderr.write("[check2] " + msg + "\n")
        sys.stderr.flush()
    except Exception:
        pass


def _patch_trainer(module):
    trainer_cls = getattr(module, "Trainer", None)
    if trainer_cls is None or getattr(trainer_cls.__init__, _MARK, False):
        return False
    orig_init = trainer_cls.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        try:
            from check2_steplog import Check2StepLogger

            handler = getattr(self, "callback_handler", None)
            existing = getattr(handler, "callbacks", []) if handler is not None else []
            if not any(isinstance(cb, Check2StepLogger) for cb in existing):
                self.add_callback(Check2StepLogger())
                _log("step logger attached to %s (pid %d)" % (type(self).__name__, os.getpid()))
        except Exception as exc:  # never break training; the missing file stops the trial
            _log("could not attach the step logger: %r" % (exc,))

    setattr(__init__, _MARK, True)
    __init__.__wrapped__ = orig_init
    __init__.__name__ = orig_init.__name__
    __init__.__doc__ = orig_init.__doc__
    trainer_cls.__init__ = __init__
    return True


class _Loader:
    """Delegates to the real loader, then patches the module after it has executed."""

    def __init__(self, inner):
        self._inner = inner

    def create_module(self, spec):
        create = getattr(self._inner, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module):
        self._inner.exec_module(module)
        try:
            _patch_trainer(module)
        except Exception as exc:
            _log("patch failed: %r" % (exc,))

    def __getattr__(self, name):  # get_source, is_package, ... go to the real loader
        return getattr(self._inner, name)


class _Finder:
    def __init__(self):
        self.done = False

    def find_spec(self, fullname, path=None, target=None):
        if self.done or fullname != _TARGET:
            return None
        self.done = True
        try:
            for finder in sys.meta_path:
                if finder is self or not hasattr(finder, "find_spec"):
                    continue
                spec = finder.find_spec(fullname, path, target)
                if spec is not None:
                    if spec.loader is not None and hasattr(spec.loader, "exec_module"):
                        spec.loader = _Loader(spec.loader)
                    else:
                        _log("unexpected loader for %s: %r" % (fullname, spec.loader))
                    return spec
        except Exception as exc:
            _log("finder failed: %r" % (exc,))
        return None


def _install():
    if os.environ.get("CHECK2_HOOK_DISABLE"):
        return None
    if os.environ.get("LOCAL_RANK") is None:
        return None
    if _TARGET in sys.modules:  # imported before us; patch directly
        _patch_trainer(sys.modules[_TARGET])
        return None
    finder = _Finder()
    sys.meta_path.insert(0, finder)
    return finder


_CHECK2_FINDER = _install()
