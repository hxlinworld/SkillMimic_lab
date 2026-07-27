"""Configure per-process writable state for Isaac Sim Kit."""

from __future__ import annotations

import os
import sys
import tempfile


def configure_kit_runtime(*, disable_ngx: bool = False) -> str:
    """Give each Kit worker its own node-local portable directory."""

    runtime_root = os.environ.get("SKILLMIMIC_KIT_RUNTIME_ROOT")
    if runtime_root is None:
        runtime_root = os.path.join(tempfile.gettempdir(), f"skillmimic-kit-{os.getpid()}")

    local_rank = os.environ.get("LOCAL_RANK", "0")
    worker_root = os.path.abspath(os.path.join(runtime_root, f"rank-{local_rank}"))
    os.makedirs(worker_root, exist_ok=True)

    # SimulationApp reads Kit's unprocessed arguments directly from sys.argv.
    # A unique portable root prevents workers and cluster nodes from sharing
    # writable driver/shader caches from the standalone Isaac Sim archive.
    sys.argv.extend(("--portable-root", worker_root))
    if disable_ngx:
        # Physics-only headless runs do not use DLSS. Avoid initializing NGX,
        # which is part of the Vulkan startup path on older Kit releases.
        sys.argv.append("--/ngx/enabled=false")
    # Sim 5.1's MJCF importer emits one harmless warning per drive axis while
    # it converts the legacy hinge stack. Isaac Lab reapplies those gains via
    # the articulation tensor API, so keep errors but suppress that flood.
    sys.argv.append("--/log/channels/omni.physx.plugin=error")

    kit_log_dir = os.environ.get("SKILLMIMIC_KIT_LOG_DIR")
    if kit_log_dir is not None:
        kit_log_dir = os.path.abspath(kit_log_dir)
        os.makedirs(kit_log_dir, exist_ok=True)
        sys.argv.append(f"--/log/file={os.path.join(kit_log_dir, f'rank-{local_rank}.log')}")

    print(
        f"[SkillMimic Lab] Kit runtime: rank={local_rank} portable_root={worker_root} "
        f"ngx_enabled={not disable_ngx}",
        flush=True,
    )
    return worker_root
