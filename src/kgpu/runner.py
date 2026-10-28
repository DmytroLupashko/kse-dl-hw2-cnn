"""Kernel side of job: stdlib only (runs before installs, Python 3.11), source pasted into notebook cell by `kgpu.job`.

`run(spec)`: `pythonpath` folders on `sys.path` and `PYTHONPATH`; `setup.mkdir` folders made, `setup.pip` installed,
`setup.pip_missing` installed only when module absent (image builds kept); `env` set, value `{find: <file>}` = folder of
first match under kernel inputs, variable already set kept; `tasks.items` queued, one `tasks.command` process per GPU at
a time, output into `tasks.log` and kernel log with `[<item>]` prefix; kernel fails if any task failed.
Placeholders in `command`, `log`, env values: `{item}`, `{gpu}`, `{workers}` (CPUs per GPU), `{workdir}`, `{items}`, `vars`.
"""

import importlib.util
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

INPUTS = "/kaggle/input"
POLL_SECONDS = 10


def argv(command, **values):
    """`command` formatted with `values`, split like shell; leading `python` -> this interpreter."""
    args = shlex.split(command.format(**values))

    return [sys.executable, *args[1:]] if args[0] == "python" else args


def gpus():
    """Visible CUDA GPUs by `nvidia-smi -L`; 0 without driver."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True
        ).stdout
    except FileNotFoundError:
        return 0

    return sum(line.startswith("GPU ") for line in out.splitlines())


def resolve(value, **values):
    """Env value: `{find: <file>}` -> folder of first match under `KGPU_INPUTS` (default Kaggle inputs), else formatted."""
    if isinstance(value, dict):
        return str(
            next(
                Path(os.environ.get("KGPU_INPUTS", INPUTS)).rglob(value["find"])
            ).parent
        )

    return str(value).format(**values)


def tee(process, item, log):
    with open(log, "w") as f:
        for line in process.stdout:
            f.write(line)
            print(f"[{item}] {line}", end="", flush=True)


def run(spec):
    # imports, folders, packages, env shared by tasks
    paths = [os.path.abspath(p) for p in spec["pythonpath"]]
    sys.path[:0] = paths
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [*paths, os.environ.get("PYTHONPATH", "")]
    )
    setup, tasks = spec["setup"], spec["tasks"]

    for folder in setup["mkdir"]:
        Path(folder).mkdir(parents=True, exist_ok=True)

    missing = [m for m in setup["pip_missing"] if importlib.util.find_spec(m) is None]

    if setup["pip"] or missing:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", *setup["pip"], *missing],
            check=True,
        )

    n_gpus = gpus()

    if not n_gpus and spec["device"] == "t4x2":
        sys.exit("no GPU: accelerator not granted, check weekly quota")

    slots = max(n_gpus, 1)
    values = {
        **spec["vars"],
        "workdir": os.getcwd(),
        "items": tasks["items"],
        "workers": max(1, (os.cpu_count() or 2) // slots),
    }
    env = {
        **os.environ,
        **{
            k: resolve(v, **values)
            for k, v in spec["env"].items()
            if k not in os.environ
        },
    }
    queue, running, failed = (
        list(tasks["items"]),
        {},
        [],
    )  # running: slot -> (item, process, tee thread, log)

    while queue or running:
        # start next task on each free slot
        for slot in range(slots):
            if slot in running or not queue:
                continue

            item = queue.pop(0)
            task = {**values, "item": item, "gpu": slot}
            task_env = {
                **env,
                **{k: resolve(v, **task) for k, v in tasks["env"].items()},
            }

            if n_gpus:
                task_env["CUDA_VISIBLE_DEVICES"] = str(slot)

            log = Path(tasks["log"].format(**task))
            log.parent.mkdir(parents=True, exist_ok=True)
            process = subprocess.Popen(
                argv(tasks["command"], **task),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=task_env,
                text=True,
            )
            thread = threading.Thread(target=tee, args=(process, item, log))
            thread.start()
            running[slot] = item, process, thread, log
            print(f"[{item}] started on slot {slot}", flush=True)

        # wait, then free slots of finished tasks
        time.sleep(POLL_SECONDS)

        for slot, (item, process, thread, log) in list(running.items()):
            if process.poll() is None:
                continue

            thread.join()  # rest of output written
            del running[slot]

            if process.returncode:
                failed.append(item)

            print(f"[{item}] exit {process.returncode}, log {log}", flush=True)

    if failed:
        sys.exit(f"failed tasks: {failed}")
