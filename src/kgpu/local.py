"""`local` device: notebook code cells run as IPython script in background, output into log file."""

import os
import signal
import subprocess
import sys

import nbformat


def start(notebook, folder, workdir, log):
    """Detached run of `notebook` code cells with `sys.executable`, cwd `workdir`; returns pid. Exit code written into
    `folder/exitcode` on finish."""
    folder, workdir, log = (
        folder.resolve(),
        workdir.resolve(),
        log.resolve(),
    )  # run cwd is `workdir`
    script = folder / "run.ipy"
    script.write_text(
        "\n\n".join(
            c.source
            for c in nbformat.read(notebook, as_version=4).cells
            if c.cell_type == "code"
        )
    )
    (folder / "exitcode").unlink(missing_ok=True)
    command = f'"{sys.executable}" -m IPython --no-banner "{script}" > "{log}" 2>&1; echo $? > "{folder / "exitcode"}"'
    env = {**os.environ, "KGPU_WORKDIR": str(workdir), "PYTHONUNBUFFERED": "1"}

    return subprocess.Popen(
        ["sh", "-c", command], cwd=workdir, env=env, start_new_session=True
    ).pid


def status(folder, pid):
    """COMPLETE or ERROR by exit code, RUNNING while process alive, ERROR when gone without exit code."""
    exitcode = folder / "exitcode"

    if exitcode.exists():
        return "COMPLETE" if exitcode.read_text().strip() == "0" else "ERROR"

    try:
        os.kill(pid, 0)
        return "RUNNING"
    except ProcessLookupError:
        return "ERROR"


def stop(pid):
    """Whole process group: shell, IPython, its children."""
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
