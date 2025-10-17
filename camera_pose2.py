"""Legacy entry point maintained for backwards compatibility.

This script simply proxies to :func:`camera_pose.main`.  Historically it was
used for experimental development during the summer school project.  The new
implementation keeps the file so that existing documentation continues to work
while benefiting from the modernised code base.
"""

from __future__ import annotations

from camera_pose import main


if __name__ == "__main__":  # pragma: no cover - convenience wrapper
    raise SystemExit(main())
