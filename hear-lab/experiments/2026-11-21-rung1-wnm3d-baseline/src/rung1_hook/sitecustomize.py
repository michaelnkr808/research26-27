"""rung 1 episode filter, loaded only by the GN0 remote eval client that run_eval.sh starts.

run_eval.sh puts this folder on PYTHONPATH for the one eval_remote.sh call and sets RUNG1_EPISODE_IDS.
Python imports sitecustomize at start-up, before gn0.evaluation.remote runs, so the patch in
rung1_filter.install() is in place when remote.py does "from gn0.evaluation.common import ...".
No GN0 or GN-Bench file is changed.

If anything here fails, the process exits at once (code 97) instead of evaluating unfiltered episodes.
Without RUNG1_EPISODE_IDS this file does nothing.
"""
import os

if os.environ.get("RUNG1_EPISODE_IDS"):
    try:
        import rung1_filter

        rung1_filter.install()
    except BaseException:  # noqa: BLE001 - any failure must stop the client
        import sys
        import traceback

        traceback.print_exc()
        sys.stderr.write("[rung1] episode filter failed to attach; refusing to run unfiltered\n")
        sys.stderr.flush()
        os._exit(97)
