"""`kgpu:progress {json}` lines of run log: task, step, total, optional metrics and elapsed seconds."""

import json

MARKER = "kgpu:progress "
DECODER = json.JSONDecoder()


def parse(log):
    """Last progress per task, in order of first appearance. Kaggle notebook logs wrap output in JSON: quotes escaped."""
    tasks = {}
    start = log.find(MARKER)

    while start != -1:
        rest = log[start + len(MARKER) : start + 4096]

        for candidate in (rest, rest.replace('\\"', '"')):
            try:
                line = DECODER.raw_decode(candidate)[0]  # stops at end of object
                break
            except json.JSONDecodeError:
                line = None

        if isinstance(line, dict) and "task" in line:
            tasks[line["task"]] = line

        start = log.find(MARKER, start + 1)

    return list(tasks.values())


def eta(line):
    """Seconds left from elapsed and fraction done; None without elapsed or before first step."""
    step, total, elapsed = line.get("step", 0), line.get("total"), line.get("elapsed")

    if not (step and total and elapsed):
        return None

    return elapsed * (total - step) / step


def duration(seconds):
    if seconds is None:
        return "-"

    hours, rest = divmod(int(seconds), 3600)

    return f"{hours}h{rest // 60:02d}m"


def render(lines, width=30):
    """One bar per task: step/total, ETA, metrics."""
    rows = []

    for line in lines:
        step, total = line.get("step", 0), line.get("total") or 0
        done = int(width * step / total) if total else 0
        metrics = " ".join(
            f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in line.get("metrics", {}).items()
        )
        rows.append(
            f"{line['task']:<28} [{'#' * done}{'.' * (width - done)}] {step}/{total} eta {duration(eta(line))} {metrics}"
        )

    return "\n".join(rows) or "no progress lines yet"
