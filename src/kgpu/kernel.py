"""Kernel: one notebook pushed under one name; folder `<home>/<name>/` holds built notebook, metadata, run history,
outputs."""

import base64
import hashlib
import io
import json
import re
import tarfile
import time
from datetime import datetime
from pathlib import Path

import nbformat

from . import local
from .kaggle import ACTIVE, DEVICES, kaggle, run_status, username

# first cell of pushed notebook; stdlib only, Python 3.11
CELL = """\
# kgpu: working dir, code bundle, deadline
import base64, io, os, sys, tarfile, time
os.chdir(os.environ.get("KGPU_WORKDIR", "/kaggle/working"))
"""
UNPACK = """\
tarfile.open(fileobj=io.BytesIO(base64.b64decode("{bundle}"))).extractall(".", filter="data")
sys.path.insert(0, os.getcwd())
"""
DEADLINE = 'os.environ["KGPU_DEADLINE"] = str(time.time() + {budget})  # training stops and saves before it\n'
PUSH_LAG = 120  # seconds after push while Kaggle status may be of previous version
SOURCES = {
    "competition": "competition_sources",
    "dataset": "dataset_sources",
    "kernel": "kernel_sources",
}


def bundle(code):
    """base64 tar.gz of `code` folders and files, relative paths kept as given; same bytes for same files."""
    buf = io.BytesIO()

    with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=9) as tar:
        for item in map(Path, code):
            files = [item] if item.is_file() else item.rglob("*")

            for path in sorted(
                p for p in files if p.is_file() and "__pycache__" not in p.parts
            ):
                info = tar.gettarinfo(path, arcname=str(path))
                info.mtime = info.uid = info.gid = 0
                info.uname = info.gname = ""

                with path.open("rb") as f:
                    tar.addfile(info, f)

    data = buf.getvalue()
    data = data[:4] + b"\0\0\0\0" + data[8:]  # gzip header mtime

    return base64.b64encode(data).decode()


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:50]


def now():
    return datetime.now().isoformat(timespec="seconds")


class Kernel:
    def __init__(self, name, home):
        self.name = slug(name)
        self.dir = Path(home) / self.name
        self.output = self.dir / "output"
        self.notebook = self.dir / f"{self.name}.ipynb"
        self.history = self.dir / "runs.jsonl"

    @classmethod
    def all(cls, home):
        return [
            cls(p.parent.name, home) for p in sorted(Path(home).glob("*/runs.jsonl"))
        ]

    @property
    def id(self):
        return f"{username()}/{self.name}"

    # run history: one JSON line per push

    def runs(self):
        if not self.history.exists():
            return []

        return [json.loads(line) for line in self.history.read_text().splitlines()]

    def save_runs(self, runs):
        self.history.write_text("".join(json.dumps(r) + "\n" for r in runs))

    def last(self):
        """Latest run, status refreshed while active; `finished` set when first seen finished."""
        runs = self.runs()

        if not runs:
            raise SystemExit(f"no runs of {self.name} in {self.dir.parent}")

        run = runs[-1]

        if run["status"] in ACTIVE:
            if run["device"] == "local":
                status = local.status(self.dir, run["pid"])
            else:
                status = run_status(self.id) or "DELETED"

                if status not in ACTIVE and self.since_push(run) < PUSH_LAG:
                    status = run["status"]  # Kaggle may still report previous version

            if status != run["status"]:
                run["status"] = status

                if status not in ACTIVE:
                    run["finished"] = now()

                self.save_runs(runs)

        return run

    # push

    def build(self, source, code, data, device, buffer):
        """Kernel folder: notebook of kgpu cell and `source` cell, `kernel-metadata.json`. Returns run record."""
        accelerator, limit = DEVICES[device]
        self.dir.mkdir(parents=True, exist_ok=True)

        # first cell: working dir, code, deadline
        first = CELL
        packed = code and bundle(code)

        if packed:
            first += UNPACK.format(bundle=packed)

        budget = limit and limit - buffer

        if budget:
            first += DEADLINE.format(budget=budget)

        nb = nbformat.v4.new_notebook(
            cells=[nbformat.v4.new_code_cell(first), nbformat.v4.new_code_cell(source)],
            metadata={
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                }
            },
        )
        nbformat.write(nb, self.notebook)

        # kernel settings: private, internet, accelerator, data sources
        metadata = {
            "id": self.id,
            "title": self.name,  # Kaggle derives slug from title
            "code_file": self.notebook.name,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": "true",
            "enable_gpu": str(device == "t4x2").lower(),
            "enable_tpu": str(device == "tpu").lower(),
            "enable_internet": "true",
            **{key: [] for key in SOURCES.values()},
        }

        if accelerator:
            metadata["machine_shape"] = accelerator

        for item in data:
            kind, _, ref = item.partition(":")

            if kind not in SOURCES or not ref:
                raise SystemExit(
                    f"data source {item!r}: expected {'|'.join(SOURCES)}:<ref>"
                )

            metadata[SOURCES[kind]].append(ref)

        (self.dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))

        code_hash = packed and hashlib.sha256(packed.encode()).hexdigest()[:12]

        return {
            "pushed": now(),
            "device": device,
            "budget": budget,
            "code": [str(c) for c in code],
            "code_hash": code_hash,
            "data": list(data),
            "source": source,
            "status": "QUEUED",
            "finished": None,
        }

    def push(self, source, code=(), data=(), device="t4x2", buffer=1800, dry_run=False):
        """Notebook running `source` code. New Kaggle version, run starts at once; `local`: background run. `dry_run`: folder only."""
        if self.runs() and self.last()["status"] in ACTIVE:
            raise SystemExit(
                f"{self.name} still {self.last()['status']}: delete it or push under other name"
            )

        run = self.build(source, code, data, device, buffer)

        if dry_run:
            print(f"built {self.dir}, not pushed")
            return

        # start run
        if device == "local":
            self.output.mkdir(exist_ok=True)
            run["pid"] = local.start(
                self.notebook, self.dir, self.output, self.log_file()
            )
            run["status"] = "RUNNING"
        else:
            args = ["--accelerator", DEVICES[device][0]] if DEVICES[device][0] else []
            print(kaggle("kernels", "push", "-p", str(self.dir), *args))

        self.save_runs([*self.runs(), run])
        print(f"{self.name}: pushed to {device}")

    # run state

    def log_file(self):
        return self.output / f"{self.name}.log"

    def log(self):
        """Log text of latest run so far; Kaggle stream endpoint can fail (500) while session starts: empty then."""
        if self.last()["device"] == "local":
            return self.log_file().read_text() if self.log_file().exists() else ""

        return kaggle("kernels", "logs", self.id, check=False) or ""

    def deadline_left(self, run):
        """Seconds to deadline, counted from push: run start is later, so real budget is larger."""
        if not run["budget"]:
            return None

        return run["budget"] - self.since_push(run)

    @staticmethod
    def since_push(run):
        return time.time() - datetime.fromisoformat(run["pushed"]).timestamp()

    def pull(self):
        """Outputs of finished Kaggle run into `output/`. Refused while active: output would be of previous run."""
        run = self.last()

        if run["status"] in ACTIVE:
            raise SystemExit(f"{self.name} {run['status']}: pull after it finishes")

        if run["device"] == "local":
            print(f"local run: outputs already in {self.output}")
            return

        kaggle("kernels", "output", self.id, "-p", str(self.output), "-o", "-q")
        print(f"{self.name}: outputs in {self.output}")

    def delete(self):
        """Kaggle kernel deleted, running or not; local run killed. Folder and history kept."""
        runs = self.runs()
        run = self.last()

        if run["device"] == "local":
            local.stop(run["pid"])
        else:
            kaggle("kernels", "delete", self.id, "-y")

        if run["status"] in ACTIVE:
            runs[-1] = {**run, "status": "DELETED", "finished": now()}
            self.save_runs(runs)

        print(f"{self.name}: deleted")
