"""Polar occupancy grid for short-term spatial memory.

Why: per-frame depth tells the agent what's in view *right now*. Turn the
camera away from an obstacle and that information is gone. The grid stores
depth observations in a robot-centric polar frame and updates them via
ego-motion (orientation delta + linear speed) so "I just saw a wall on my
left" survives a turn.

Frame convention:
- Robot-centered, robot-aligned (heading = 0° in grid frame = robot's forward)
- Bearing positive = left of forward, negative = right (matches the agent's
  angular convention)
- Cells are indexed [angular_bin, distance_bin]; values are occupancy in [0, 1]

Usage:
    grid = PolarOccupancyGrid()
    grid.update_from_depth(depth_reading, hfov_deg, vfov_deg, mount_height_m)
    grid.advect(orientation_delta_deg, linear_speed_mps, dt_seconds)
    summary = grid.summary_lines()                # list of bin descriptions
    blocked = grid.is_blocked_at(0.0, max_distance=1.5)  # forward, 1.5m
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

try:
    from config import (
        logger,
        GRID_ANGULAR_BINS,
        GRID_DISTANCE_EDGES,
        GRID_DECAY,
        GRID_OCCUPIED_THRESHOLD,
        SAFE_FORWARD_CLEARANCE_M,
    )
except ImportError:
    from .config import (
        logger,
        GRID_ANGULAR_BINS,
        GRID_DISTANCE_EDGES,
        GRID_DECAY,
        GRID_OCCUPIED_THRESHOLD,
        SAFE_FORWARD_CLEARANCE_M,
    )


def _parse_distance_edges(spec: str) -> List[float]:
    edges = [float(s.strip()) for s in spec.split(",") if s.strip()]
    if len(edges) < 2:
        raise ValueError(f"GRID_DISTANCE_EDGES must have >=2 values, got: {spec!r}")
    if any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError(f"GRID_DISTANCE_EDGES must be strictly increasing: {edges}")
    return edges


# ==============================================================================

class PolarOccupancyGrid:
    """Robot-centric polar occupancy grid with ego-motion integration."""

    def __init__(
        self,
        n_angular: int = GRID_ANGULAR_BINS,
        distance_edges: Optional[List[float]] = None,
        decay: float = GRID_DECAY,
        occupied_threshold: float = GRID_OCCUPIED_THRESHOLD,
    ):
        self.n_angular = n_angular
        self.distance_edges = (distance_edges
                               if distance_edges is not None
                               else _parse_distance_edges(GRID_DISTANCE_EDGES))
        self.n_distance = len(self.distance_edges) - 1
        self.decay = float(decay)
        self.occupied_threshold = float(occupied_threshold)

        self.cells = np.zeros((self.n_angular, self.n_distance), dtype=np.float32)
        self.last_update_ts: float = 0.0

    # ------------------------------------------------------------------
    # Indexing helpers
    # ------------------------------------------------------------------
    def _angular_bin(self, bearing_deg: float) -> int:
        """Bearing in degrees -> angular bin index. Bin 0 spans [-half, +half]."""
        bin_width = 360.0 / self.n_angular
        # Map to [0, 360)
        a = (bearing_deg + 360.0) % 360.0
        # Center bin 0 on 0° (forward) — shift by half a bin
        a_shifted = (a + bin_width / 2.0) % 360.0
        return int(a_shifted // bin_width)

    def _bin_center_deg(self, bin_idx: int) -> float:
        """Returns the bearing (signed, -180..180) at the center of a bin."""
        bin_width = 360.0 / self.n_angular
        center = bin_idx * bin_width
        # Convert 0..360 -> -180..180 for human-friendly output
        if center > 180.0:
            center -= 360.0
        return center

    def _distance_bin(self, distance_m: float) -> int:
        """Distance in meters -> distance bin index. Returns -1 if out of range."""
        for i in range(self.n_distance):
            if self.distance_edges[i] <= distance_m < self.distance_edges[i + 1]:
                return i
        return -1

    # ------------------------------------------------------------------
    # Update from depth
    # ------------------------------------------------------------------
    def update_from_depth(
        self,
        depth_map: Optional[np.ndarray],
        is_metric: bool,
        hfov_deg: float,
        vfov_deg: float,
        mount_height_m: float,
        camera_pitch_deg: float = 0.0,
        forward_only: bool = True,
    ) -> bool:
        """Project a depth map into the grid by sampling depth pixels.

        Returns True if the update was applied. Skips silently when the depth
        map is missing or non-metric (we can't project relative depth into
        meters without external scale).
        """
        if depth_map is None or not is_metric:
            return False

        h, w = depth_map.shape
        # Sample on a coarse stride for speed — every 4 pixels is ~16x speedup
        stride_y = max(1, h // 32)
        stride_x = max(1, w // 32)

        # Per-cell evidence (max depth-confidence) accumulator
        new_cells = np.zeros_like(self.cells)
        cell_count = np.zeros_like(self.cells)

        for y in range(0, h, stride_y):
            # Skip rows that are clearly sky (top 25%)
            if y < int(0.25 * h):
                continue
            for x in range(0, w, stride_x):
                d = float(depth_map[y, x])
                if d <= 0.05 or d > self.distance_edges[-1]:
                    continue
                # bearing: x position -> degrees off forward axis
                # x=0 (left edge) => +hfov/2 ; x=w-1 (right edge) => -hfov/2
                centered = (w / 2.0 - x) / (w / 2.0)
                bearing = centered * (hfov_deg / 2.0)

                if forward_only and abs(bearing) > 90.0:
                    continue

                a_bin = self._angular_bin(bearing)
                d_bin = self._distance_bin(d)
                if d_bin < 0:
                    continue
                new_cells[a_bin, d_bin] += 1.0
                cell_count[a_bin, d_bin] += 1.0

        if cell_count.sum() == 0:
            return False

        # Convert hit counts to occupancy values 0..1
        # Saturating function: occupancy = 1 - exp(-count / scale)
        scale = 6.0
        new_occ = 1.0 - np.exp(-new_cells / scale)

        # MAX-aggregate with existing decayed cells (no overriding old info
        # with weak new info, but boost when we re-observe).
        self.cells = np.maximum(self.cells, new_occ)
        return True

    # ------------------------------------------------------------------
    # Ego-motion advection
    # ------------------------------------------------------------------
    def advect(
        self,
        orientation_delta_deg: float,
        linear_speed_mps: float,
        dt: float,
    ):
        """Update grid by ego-motion. Apply per-tick decay, then rotate (by
        orientation delta) and shift forward (by linear distance)."""
        # 1. Decay
        self.cells *= self.decay

        # 2. Rotate: shift angular bins. Positive orientation_delta = robot
        # turned left (toward higher bearing); the world appears to rotate
        # right (lower bearing), so cells should shift to LOWER angular bins.
        if abs(orientation_delta_deg) > 1e-3:
            bin_width = 360.0 / self.n_angular
            shift = int(round(-orientation_delta_deg / bin_width))
            if shift != 0:
                self.cells = np.roll(self.cells, shift=shift, axis=0)

        # 3. Translate: move forward by speed * dt. Cells in front of the
        # robot get pulled to closer distance bins. Cells behind get pushed
        # farther. Approximation: only handle the forward cone (|bearing| < 90°).
        distance_m = linear_speed_mps * dt
        if abs(distance_m) > 0.02:  # ~2 cm threshold
            self._translate_forward(distance_m)

    def _translate_forward(self, distance_m: float):
        """Shift cells along the bin's radial axis by `distance_m` meters."""
        new = np.zeros_like(self.cells)
        # For each angular bin, project radial cells onto new distance bins
        for a in range(self.n_angular):
            bearing_center = self._bin_center_deg(a)
            cos_b = math.cos(math.radians(bearing_center))
            # In the forward cone, moving forward by distance_m brings cells
            # closer by distance_m * cos(bearing). For rear bins (cos < 0),
            # cells move farther.
            shift_m = distance_m * cos_b
            for d in range(self.n_distance):
                old_lo = self.distance_edges[d]
                old_hi = self.distance_edges[d + 1]
                new_lo = old_lo - shift_m
                new_hi = old_hi - shift_m
                # Find which new bins this overlaps
                for nd in range(self.n_distance):
                    n_lo = self.distance_edges[nd]
                    n_hi = self.distance_edges[nd + 1]
                    overlap = max(0.0, min(new_hi, n_hi) - max(new_lo, n_lo))
                    span = old_hi - old_lo
                    if overlap > 0 and span > 0:
                        new[a, nd] = max(new[a, nd], self.cells[a, d] * (overlap / span))
        self.cells = new

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def nearest_blocked_distance(self, bearing_deg: float) -> Optional[float]:
        """Closest occupied distance in the angular bin containing `bearing_deg`.
        Returns None if nothing is blocked in that direction within the grid."""
        a = self._angular_bin(bearing_deg)
        for d in range(self.n_distance):
            if self.cells[a, d] >= self.occupied_threshold:
                # Return the FAR edge of the blocked bin = a "blocked at this distance" marker
                return self.distance_edges[d + 1]
        return None

    def is_blocked_at(self, bearing_deg: float, max_distance_m: float) -> bool:
        """True if any cell within max_distance_m at the given bearing is occupied."""
        a = self._angular_bin(bearing_deg)
        for d in range(self.n_distance):
            if self.distance_edges[d + 1] > max_distance_m:
                break
            if self.cells[a, d] >= self.occupied_threshold:
                return True
        return False

    def best_free_bearing(self, max_bearing_abs_deg: float = 90.0) -> Optional[float]:
        """Among bins within ±max_bearing_abs_deg of forward, return the bearing
        of the bin with the LARGEST clear distance (deepest free space)."""
        best_bearing: Optional[float] = None
        best_clear_dist = -1.0
        for a in range(self.n_angular):
            bearing = self._bin_center_deg(a)
            if abs(bearing) > max_bearing_abs_deg:
                continue
            # Find first occupied distance bin in this direction; "clear" up to that distance
            clear_dist = self.distance_edges[-1]
            for d in range(self.n_distance):
                if self.cells[a, d] >= self.occupied_threshold:
                    clear_dist = self.distance_edges[d]
                    break
            if clear_dist > best_clear_dist:
                best_clear_dist = clear_dist
                best_bearing = bearing
        return best_bearing

    def summary_lines(self, max_bearing_abs_deg: float = 180.0) -> List[str]:
        """Compact human-readable summary, one line per angular bin."""
        out = []
        labels = self._direction_labels()
        for a in range(self.n_angular):
            bearing = self._bin_center_deg(a)
            if abs(bearing) > max_bearing_abs_deg:
                continue
            blocked_dist = None
            for d in range(self.n_distance):
                if self.cells[a, d] >= self.occupied_threshold:
                    blocked_dist = self.distance_edges[d + 1]
                    break
            tag = labels[a]
            if blocked_dist is None:
                out.append(f"  {tag} ({bearing:+.0f}°): free for >{self.distance_edges[-1]:.0f}m")
            else:
                out.append(f"  {tag} ({bearing:+.0f}°): blocked at <{blocked_dist:.1f}m")
        return out

    def to_compact_summary(self) -> str:
        """Single-line forward-cone summary used in the brain prompt."""
        # Bins within ±90° of forward
        parts = []
        for a in range(self.n_angular):
            bearing = self._bin_center_deg(a)
            if abs(bearing) > 90.0:
                continue
            blocked_dist = None
            for d in range(self.n_distance):
                if self.cells[a, d] >= self.occupied_threshold:
                    blocked_dist = self.distance_edges[d + 1]
                    break
            if blocked_dist is None:
                parts.append(f"{bearing:+.0f}°:free")
            else:
                parts.append(f"{bearing:+.0f}°:<{blocked_dist:.1f}m")
        return " | ".join(parts) if parts else "(empty)"

    def _direction_labels(self) -> List[str]:
        """Cardinal-style labels per angular bin (Front / Front-Right / ...)."""
        labels = []
        for a in range(self.n_angular):
            b = self._bin_center_deg(a)
            if abs(b) < 22.5:
                labels.append("Front")
            elif b >= 22.5 and b < 67.5:
                labels.append("Front-Left")
            elif b >= 67.5 and b < 112.5:
                labels.append("Left")
            elif b >= 112.5 and b < 157.5:
                labels.append("Back-Left")
            elif b >= 157.5 or b < -157.5:
                labels.append("Back")
            elif b <= -22.5 and b > -67.5:
                labels.append("Front-Right")
            elif b <= -67.5 and b > -112.5:
                labels.append("Right")
            else:
                labels.append("Back-Right")
        return labels
