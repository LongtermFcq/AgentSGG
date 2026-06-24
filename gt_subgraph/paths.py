"""Shared paths for gt_subgraph scripts.

Layout:
  gt_subgraph/
  ├── outputs/gt/     <- run.py deliverables (full dataset)
  └── demo/
      ├── overlays/   <- smoke_test.py
      ├── stats/      <- stats.py
      └── viz/        <- visualize.py

Change ROOT here when relocating the 3RScan dataset.
"""
import os

GT_SUBGRAPH_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(GT_SUBGRAPH_DIR)

# 3RScan dataset root
ROOT = os.path.join(PROJECT_ROOT, "3RScan")

# Full-run GT JSON (run.py)
GT_OUT = os.path.join(GT_SUBGRAPH_DIR, "outputs", "gt")

# Demo / sample-only artifacts
DEMO_DIR = os.path.join(GT_SUBGRAPH_DIR, "demo")
DEMO_OVERLAYS = os.path.join(DEMO_DIR, "overlays")
DEMO_STATS = os.path.join(DEMO_DIR, "stats")
DEMO_VIZ = os.path.join(DEMO_DIR, "viz")

DEFAULT_SCANS = [
    "7272e16c-a01b-20f6-8961-a0927b4a7629",
    "7272e161-a01b-20f6-8b5a-0b97efeb6545",
    "f62fd5fd-9a3f-2f44-883a-1e5cf819608e",
]


def gt_json_path(scan_id):
    """Path to compact GT JSON for a scan (keyed by first 8 chars of scan_id)."""
    return os.path.join(GT_OUT, f"gt_{scan_id[:8]}.json")


def ensure_gt_out():
    os.makedirs(GT_OUT, exist_ok=True)


def ensure_demo_dirs():
    os.makedirs(DEMO_OVERLAYS, exist_ok=True)
    os.makedirs(DEMO_STATS, exist_ok=True)
    os.makedirs(DEMO_VIZ, exist_ok=True)
