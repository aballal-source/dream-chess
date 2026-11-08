"""A simulated top camera that ray-traces pieces on a board, with realistic flaws.

It stands in for the RealSense so the whole vision path can be exercised without
hardware. Passing here shows the geometry and logic work, not real-camera accuracy.
"""
from __future__ import annotations

import chess
import numpy as np

from .core import Occupancy
from .vision import Frame, Intrinsics

# (height, radius) in meters, approximating a tournament set.
PIECE_SIZES = {
    chess.PAWN: (0.048, 0.014),
    chess.KNIGHT: (0.065, 0.017),
    chess.BISHOP: (0.075, 0.016),
    chess.ROOK: (0.055, 0.017),
    chess.QUEEN: (0.088, 0.018),
    chess.KING: (0.097, 0.019),
}
BGR = {"w": np.array([205, 220, 228.0]), "b": np.array([35, 42, 50.0])}
LIGHT_SQUARE, DARK_SQUARE = np.array([150, 195, 225.0]), np.array([55, 95, 135.0])
TABLE = np.array([120, 120, 115.0])

Piece = tuple[str, float, float]  # color code, height, radius


def pieces_from_board(board: chess.Board) -> dict[str, Piece]:
    return {
        chess.square_name(sq): ("w" if p.color else "b", *PIECE_SIZES[p.piece_type])
        for sq, p in board.piece_map().items()
    }


def pieces_from_occupancy(occ: Occupancy) -> dict[str, Piece]:
    """The occupancy has no piece types, so sizes vary by square to keep a realistic mix."""
    sizes = list(PIECE_SIZES.values())
    return {sq: (code, *sizes[chess.parse_square(sq) % len(sizes)]) for sq, code in occ.items()}


class SynthCamera:
    def __init__(
        self,
        tilt_deg: float = 25.0,
        distance: float = 0.85,
        width: int = 640,
        height: int = 360,
        square: float = 0.057,
        depth_noise: float = 0.003,
        dropout: float = 0.02,
        color_noise: float = 4.0,
        jitter: float = 0.005,
        with_depth: bool = True,
        seed: int = 0,
        side: str = "white",
    ):
        self.square = square
        self.depth_noise, self.dropout, self.color_noise = depth_noise, dropout, color_noise
        self.jitter, self.with_depth = jitter, with_depth
        self.rng = np.random.default_rng(seed)
        self.pieces: dict[str, Piece] = {}
        self._offsets: dict[str, np.ndarray] = {}

        focal = width * 0.727  # about a 69 degree field of view
        self.intrinsics = Intrinsics(focal, focal, width / 2, height / 2)
        center = np.array([4 * square, 4 * square, 0.0])
        tilt = np.radians(tilt_deg)
        # "white" looks from behind rank 1; "black" from behind rank 8, where the robot's own camera sits.
        facing = 1.0 if side == "white" else -1.0
        self.position = center + distance * np.array([0, -facing * np.sin(tilt), np.cos(tilt)])
        forward = (center - self.position) / distance
        right = np.cross(forward, [0, facing, 0])
        right /= np.linalg.norm(right)
        self.axes = np.column_stack((right, np.cross(forward, right), forward))  # camera to world

        v, u = np.indices((height, width), dtype=np.float64)
        k = self.intrinsics
        self.rays = np.dstack(((u - k.cx) / k.fx, (v - k.cy) / k.fy, np.ones_like(u))) @ self.axes.T

    def project(self, x: float, y: float, z: float = 0.0) -> tuple[float, float]:
        p = (np.array([x, y, z]) - self.position) @ self.axes
        k = self.intrinsics
        return k.fx * p[0] / p[2] + k.cx, k.fy * p[1] / p[2] + k.cy

    def corners(self, click_error: float = 0.0, seed: int = 0) -> np.ndarray:
        """The image position of a1, h1, h8, a8 at the board's outer corners."""
        s = self.square * 8
        points = np.array([self.project(0, 0), self.project(s, 0), self.project(s, s), self.project(0, s)])
        return points + np.random.default_rng(seed).uniform(-click_error, click_error, points.shape)

    def _center(self, name: str) -> np.ndarray:
        """Pieces sit slightly off-center, and stay put between frames like real ones."""
        if name not in self._offsets:
            self._offsets[name] = self.rng.uniform(-self.jitter, self.jitter, 2)
        sq = chess.parse_square(name)
        base = (np.array([chess.square_file(sq), chess.square_rank(sq)]) + 0.5) * self.square
        return base + self._offsets[name]

    def read(self) -> Frame:
        pos, rays = self.position, self.rays
        t = -pos[2] / rays[..., 2]  # hits on the board plane
        hit = pos + rays * t[..., None]

        s = self.square
        on_board = (hit[..., 0] >= 0) & (hit[..., 0] < 8 * s) & (hit[..., 1] >= 0) & (hit[..., 1] < 8 * s)
        dark = (np.floor(hit[..., 0] / s) + np.floor(hit[..., 1] / s)) % 2 == 0
        image = np.where(dark[..., None], DARK_SQUARE, LIGHT_SQUARE)
        image[~on_board] = TABLE

        for name, (code, height, radius) in self.pieces.items():
            cx, cy = self._center(name)
            ox, oy = pos[0] - cx, pos[1] - cy
            dx, dy = rays[..., 0], rays[..., 1]

            a, b, c = dx * dx + dy * dy, 2 * (ox * dx + oy * dy), ox * ox + oy * oy - radius * radius
            disc = b * b - 4 * a * c
            side = np.full(t.shape, np.inf)
            ok = disc >= 0
            with np.errstate(divide="ignore", invalid="ignore"):  # a ray straight down has a == 0
                side[ok] = (-b[ok] - np.sqrt(disc[ok])) / (2 * a[ok])
            side[np.isnan(side)] = np.inf
            z = pos[2] + side * rays[..., 2]
            side[~ok | (z < 0) | (z > height)] = np.inf

            cap = (height - pos[2]) / rays[..., 2]
            cap_x, cap_y = pos[0] + cap * dx - cx, pos[1] + cap * dy - cy
            cap[cap_x * cap_x + cap_y * cap_y > radius * radius] = np.inf

            closer_side = side < t
            t = np.where(closer_side, side, t)
            image[closer_side] = BGR[code] * 0.8  # shaded
            closer_cap = cap < t
            t = np.where(closer_cap, cap, t)
            image[closer_cap] = BGR[code]

        noisy = image + self.rng.normal(0, self.color_noise, image.shape)
        color = np.clip(noisy, 0, 255).astype(np.uint8)
        if not self.with_depth:
            return Frame(color)
        depth = (t + self.rng.normal(0, self.depth_noise, t.shape)).astype(np.float32)
        depth[self.rng.random(t.shape) < self.dropout] = 0
        return Frame(color, depth, self.intrinsics)
