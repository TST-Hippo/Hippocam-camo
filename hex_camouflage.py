#!/usr/bin/env python3
"""Generate a hexagonal digital-camouflage SVG from a correlated random field.

The camouflage is built by:
  1. drawing white (Gaussian) noise on an (nx, ny) hexagonal grid (one
     independent field per palette colour),
  2. convolving each field with a 2-D Gaussian kernel whose std dev equals
     the requested correlation length (measured in hexagon units),
  3. assigning each cell the colour whose (offset-shifted) field value is
     largest, where per-colour offsets are calibrated so that realised
     colour frequencies match the requested proportions,
  4. cleaning up isolated single-hex blobs, then emitting one ``<polygon>``
     per hex cell into an SVG file.

An optional text overlay can be drawn on top of the camouflage in two
ways: as glyphs burned into the hex grid (``--text-mode hex``, the
default) or as a real SVG ``<text>`` element with rotation, opacity,
arbitrary font, and optional repeated tiling along the text's own axes
(``--text-mode text``).

Required libraries:
  - NumPy (always),
  - Pillow (only when an overlay is requested via ``--text``; used both
    for rasterising glyphs in ``hex`` mode and as a font-resolution
    fallback in ``text`` mode).

Example
-------
    python hex_camouflage.py                        # use built-in defaults
    python hex_camouflage.py --width 60 --height 45 \
                             --correlation 4 --seed 7 --output woodland.svg
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Iterable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Random field
# ---------------------------------------------------------------------------
def correlated_gaussian_field(
    nx: int,
    ny: int,
    correlation_length: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return an (ny, nx) smooth Gaussian random field.

    The field is produced by spectral Gaussian smoothing of white noise.
    ``correlation_length`` is the std. dev. of the smoothing kernel,
    expressed in hexagon-grid units.
    """
    if rng is None:
        rng = np.random.default_rng()

    noise = rng.standard_normal((ny, nx))

    # Frequency grid (units of cycles per hex).
    ky = np.fft.fftfreq(ny)[:, None]
    kx = np.fft.fftfreq(nx)[None, :]

    # Fourier transform of a Gaussian of std sigma (in hex units):
    #     exp(-2 * pi^2 * sigma^2 * (kx^2 + ky^2))
    sigma = max(float(correlation_length), 1e-9)
    filt = np.exp(-2.0 * (math.pi * sigma) ** 2 * (kx * kx + ky * ky))

    spectrum = np.fft.fft2(noise) * filt
    field = np.real(np.fft.ifft2(spectrum))

    # Standardise so quantile thresholds are well-behaved.
    field -= field.mean()
    std = field.std()
    if std > 0:
        field /= std
    return field


def bin_field_by_proportion(
    field: np.ndarray, proportions: Sequence[float]
) -> np.ndarray:
    """Turn a scalar field into integer colour indices whose population
    frequencies match ``proportions`` (order preserved).

    Kept for reference/backwards-compatibility.  This produces a strict
    ordering of colours (colour ``i`` can only ever touch colours ``i-1``
    and ``i+1``) and is NOT used by :func:`assign_colors_by_argmax`.
    """
    p = np.asarray(proportions, dtype=float)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("proportions must be a non-empty 1-D sequence")
    if np.any(p < 0):
        raise ValueError("proportions must be non-negative")
    total = p.sum()
    if total <= 0:
        raise ValueError("proportions must sum to a positive value")
    p = p / total

    cumulative = np.cumsum(p)[:-1]         # upper thresholds (exclude 1.0)
    thresholds = np.quantile(field, cumulative)

    idx = np.zeros(field.shape, dtype=int)
    for t in thresholds:
        idx += (field > t).astype(int)
    return idx


def assign_colors_by_argmax(
    nx: int,
    ny: int,
    proportions: Sequence[float],
    correlation_length: float,
    rng: np.random.Generator | None = None,
    max_iterations: int = 200,
    tolerance: float = 0.002,
) -> np.ndarray:
    """Assign a colour index to each cell so that blobs of every colour can
    be adjacent to blobs of every other colour.

    An independent correlated Gaussian field is drawn per colour.  At each
    cell, the colour whose (offset-shifted) field value is largest wins.
    Because the fields are independent, there is no induced ordering of
    colours, so any two colours may share a boundary.

    Per-colour offsets are iteratively calibrated so that realised colour
    proportions match ``proportions``.

    Parameters
    ----------
    nx, ny : int
        Grid size in hexagons.
    proportions : sequence of floats
        Target relative frequency of each colour.
    correlation_length : float
        Spatial correlation length of each per-colour field (hex units).
    rng : numpy Generator, optional
    max_iterations : int
        Maximum offset-calibration steps.
    tolerance : float
        Stop when the largest per-colour proportion error falls below this.
    """
    if rng is None:
        rng = np.random.default_rng()

    targets = np.asarray(proportions, dtype=float)
    if targets.ndim != 1 or targets.size == 0:
        raise ValueError("proportions must be a non-empty 1-D sequence")
    if np.any(targets < 0):
        raise ValueError("proportions must be non-negative")
    total = targets.sum()
    if total <= 0:
        raise ValueError("proportions must sum to a positive value")
    targets = targets / total

    n = targets.size
    fields = np.stack(
        [
            correlated_gaussian_field(nx, ny, correlation_length, rng)
            for _ in range(n)
        ]
    )  # shape (n, ny, nx)

    offsets = np.zeros(n)
    assignments = np.argmax(fields, axis=0)
    step = 2.0
    prev_err = np.inf
    for _ in range(max_iterations):
        shifted = fields + offsets[:, None, None]
        assignments = np.argmax(shifted, axis=0)
        actual = np.bincount(assignments.ravel(), minlength=n) / assignments.size
        err = targets - actual
        max_err = float(np.max(np.abs(err)))
        if max_err < tolerance:
            break
        # Shrink step size if progress stalls, to avoid oscillation.
        if max_err >= prev_err:
            step *= 0.7
        prev_err = max_err
        offsets += err * step

    return assignments


# ---------------------------------------------------------------------------
# Hex grid topology helpers (odd-r offset coordinates)
# ---------------------------------------------------------------------------
# Six neighbour offsets, depending on whether the row index is even or odd.
_HEX_NEIGHBOURS_EVEN_ROW: tuple[tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0),
)
_HEX_NEIGHBOURS_ODD_ROW: tuple[tuple[int, int], ...] = (
    (-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0), (1, 1),
)


def _neighbour_colors(idx: np.ndarray) -> np.ndarray:
    """Return a ``(6, ny, nx)`` array with each cell's six hex-neighbour
    colours.  Cells whose neighbour falls outside the grid are filled
    with ``-1``.

    For each direction ``k`` and cell ``(r, c)``, ``out[k, r, c]`` is the
    colour at ``idx[r + dr, c + dc]``.
    """
    ny, nx = idx.shape
    out = np.full((6, ny, nx), -1, dtype=idx.dtype)
    for r in range(ny):
        deltas = _HEX_NEIGHBOURS_ODD_ROW if r % 2 == 1 else _HEX_NEIGHBOURS_EVEN_ROW
        for k, (dr, dc) in enumerate(deltas):
            rr = r + dr
            if not (0 <= rr < ny):
                continue
            # We want out[k, r, c] = idx[rr, c + dc].
            #   For dc >= 0, valid c is [0, nx - dc); src col is c + dc in [dc, nx).
            #   For dc <  0, valid c is [-dc, nx);   src col is c + dc in [0, nx + dc).
            if dc >= 0:
                dst_c0, dst_c1 = 0, nx - dc
                src_c0, src_c1 = dc, nx
            else:
                dst_c0, dst_c1 = -dc, nx
                src_c0, src_c1 = 0, nx + dc
            if src_c1 > src_c0:
                out[k, r, dst_c0:dst_c1] = idx[rr, src_c0:src_c1]
    return out


def remove_singleton_blobs(
    color_idx: np.ndarray, max_passes: int = 20
) -> np.ndarray:
    """Recolour every isolated single-hex blob.

    A hex cell is a "singleton" when none of its six hex neighbours share
    its colour.  Each such cell is recoloured to the colour that occurs
    most often among its neighbours (ties broken by lowest colour index).

    Recolouring one cell can in principle leave another cell newly
    singleton, so the routine iterates up to ``max_passes`` times until no
    singletons remain.  The input array is left unchanged.
    """
    out = color_idx.copy()
    if out.size == 0:
        return out

    for _ in range(max_passes):
        nbrs = _neighbour_colors(out)
        is_singleton = ~((nbrs == out[None, :, :]).any(axis=0))
        if not is_singleton.any():
            break
        rows, cols = np.where(is_singleton)
        for r, c in zip(rows.tolist(), cols.tolist()):
            valid = nbrs[:, r, c]
            valid = valid[valid >= 0]
            if valid.size == 0:
                continue
            out[r, c] = int(np.bincount(valid).argmax())
    return out


# ---------------------------------------------------------------------------
# Hexagon geometry (pointy-top, "odd-r" offset)
# ---------------------------------------------------------------------------
def _hex_vertices(cx: float, cy: float, r: float) -> list[tuple[float, float]]:
    """Vertices of a pointy-top hexagon of circumradius ``r`` centred at (cx, cy)."""
    verts = []
    for i in range(6):
        theta = math.radians(60.0 * i - 30.0)
        verts.append((cx + r * math.cos(theta), cy + r * math.sin(theta)))
    return verts


# ---------------------------------------------------------------------------
# Text overlay helpers (used by both --text-mode hex and --text-mode text)
# ---------------------------------------------------------------------------
# A few common sans-serif paths; the first one that exists is used as the
# default when the caller does not pass an explicit font.  Used by
# ``render_text_mask`` (hex mode) to rasterise glyphs.
_DEFAULT_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",          # macOS
    "C:/Windows/Fonts/arialbd.ttf",                 # Windows
)


def _find_default_font() -> str:
    for p in _DEFAULT_FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "No default TrueType font found; pass an explicit font path."
    )


def render_text_mask(
    text: str,
    height: int,
    font_path: str | None = None,
) -> np.ndarray:
    """Render ``text`` to a boolean mask of shape ``(height, width)``.

    Each ``True`` pixel corresponds to a single hex cell that the text
    covers, so ``height`` is the text's height measured in hexagons.  The
    width is determined by the aspect ratio of the rendered glyphs.

    The text is drawn at a comfortably high internal resolution, cropped
    to its ink bounds, then downsampled to exactly ``height`` rows.
    """
    if not text:
        return np.zeros((max(int(height), 1), 0), dtype=bool)
    if height <= 0:
        raise ValueError("text height must be positive")

    from PIL import Image, ImageDraw, ImageFont  # lazy import

    if font_path is None:
        font_path = _find_default_font()

    # Render at ~4x oversample for smooth downsampling.
    render_h = max(int(height) * 4, 48)
    font = ImageFont.truetype(font_path, size=render_h)
    bbox = font.getbbox(text)
    pad = max(2, render_h // 16)
    canvas_w = (bbox[2] - bbox[0]) + 2 * pad
    canvas_h = (bbox[3] - bbox[1]) + 2 * pad
    img = Image.new("L", (canvas_w, canvas_h), 0)
    ImageDraw.Draw(img).text((pad - bbox[0], pad - bbox[1]), text,
                              fill=255, font=font)
    arr = np.array(img)

    # Crop to the ink's actual bounding box.
    rows = np.where(arr.any(axis=1))[0]
    cols = np.where(arr.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return np.zeros((int(height), 0), dtype=bool)
    arr = arr[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

    # Downsample to exactly `height` rows; width scales to preserve aspect.
    ih, iw = arr.shape
    target_w = max(1, int(round(iw * height / ih)))
    small = Image.fromarray(arr).resize((target_w, int(height)), Image.LANCZOS)
    return np.array(small) > 127


def burn_text_into_grid(
    color_idx: np.ndarray,
    mask: np.ndarray,
    color_index: int,
    anchor: tuple[int, int],
) -> np.ndarray:
    """Return a copy of ``color_idx`` where every ``True`` cell of ``mask``,
    placed with its top-left at ``anchor = (row, col)``, has been
    overwritten with ``color_index``.  Pixels that fall outside the grid
    are silently clipped."""
    out = color_idx.copy()
    if mask.size == 0:
        return out
    ny, nx = out.shape
    th, tw = mask.shape
    r0, c0 = anchor

    r_start = max(0, r0)
    c_start = max(0, c0)
    r_end = min(ny, r0 + th)
    c_end = min(nx, c0 + tw)
    if r_end <= r_start or c_end <= c_start:
        return out

    sub_mask = mask[r_start - r0:r_end - r0, c_start - c0:c_end - c0]
    out_block = out[r_start:r_end, c_start:c_end]
    out_block[sub_mask] = color_index
    out[r_start:r_end, c_start:c_end] = out_block
    return out


def _resolve_text_anchor(
    text_position: tuple[int, int] | str,
    grid_shape: tuple[int, int],
    mask_shape: tuple[int, int],
) -> tuple[int, int]:
    """Convert a named position (or an explicit (row, col)) into the
    top-left hex coordinate where a text mask of shape ``mask_shape``
    should be placed within a grid of shape ``grid_shape``."""
    ny, nx = grid_shape
    th, tw = mask_shape
    margin = max(1, ny // 20)
    if isinstance(text_position, tuple):
        return (int(text_position[0]), int(text_position[1]))
    name = text_position.lower()
    c0 = (nx - tw) // 2
    if name in ("top", "top-center", "top_center"):
        return (margin, c0)
    if name in ("center", "centre", "middle"):
        return ((ny - th) // 2, c0)
    if name in ("bottom", "bottom-center", "bottom_center"):
        return (ny - th - margin, c0)
    raise ValueError(f"Unknown text_position: {text_position!r}")


def _resolve_text_pixel_position(
    text_position: tuple[int, int] | str,
    canvas_w: float,
    canvas_h: float,
    hex_radius: float,
    hex_w: float,
    row_spacing: float,
) -> tuple[float, float, str, str]:
    """Return (x, y, text-anchor, dominant-baseline) for placing an SVG
    ``<text>`` element on the camouflage canvas.

    A named position picks one of six standard spots; an explicit
    ``(row, col)`` is treated as a hex coordinate and the text is anchored
    at that hex's centre.
    """
    margin = max(hex_radius, canvas_h / 20.0)
    if isinstance(text_position, tuple):
        row, col = text_position
        x_offset = (hex_w / 2.0) if (row % 2 == 1) else 0.0
        cx = col * hex_w + (hex_w / 2.0) + x_offset
        cy = row * row_spacing + hex_radius
        return (cx, cy, "middle", "central")
    name = text_position.lower()
    if name in ("top", "top-center", "top_center"):
        return (canvas_w / 2.0, margin, "middle", "hanging")
    if name in ("center", "centre", "middle"):
        return (canvas_w / 2.0, canvas_h / 2.0, "middle", "central")
    if name in ("bottom", "bottom-center", "bottom_center"):
        return (canvas_w / 2.0, canvas_h - margin, "middle", "alphabetic")
    raise ValueError(f"Unknown text_position: {text_position!r}")


def _xml_escape(s: str) -> str:
    """Minimal XML-attribute / element-text escaping for SVG payloads."""
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;")
             .replace("'", "&apos;"))


# ---------------------------------------------------------------------------
# SVG output
# ---------------------------------------------------------------------------
def _rgb(c: Iterable[int]) -> str:
    r, g, b = (int(v) & 0xFF for v in c)
    return f"rgb({r},{g},{b})"


def generate_camouflage_svg(
    nx: int,
    ny: int,
    colors: Sequence[Sequence[int]],
    proportions: Sequence[float],
    correlation_length: float,
    hex_radius: float = 20.0,
    seed: int | None = None,
    output_path: str = "camouflage.svg",
    remove_singletons: bool = True,
    text: str | None = None,
    text_color: Sequence[int] | None = None,
    text_height: float = 10.0,
    text_position: tuple[int, int] | str = "center",
    text_font: str | None = None,
    text_mode: str = "hex",
    text_rotation: float = 0.0,
    text_opacity: float = 1.0,
    text_spacing_x: float = 0.0,
    text_spacing_y: float = 0.0,
) -> str:
    """Write a hexagonal camouflage SVG and return the output path.

    Parameters
    ----------
    nx, ny : int
        Canvas size, measured in hexagons (width, height).
    colors : sequence of (r, g, b) triples
        Colour palette, each component in 0..255.
    proportions : sequence of floats
        Relative frequency of each colour.  Must have the same length as
        ``colors``; automatically normalised to sum to 1.
    correlation_length : float
        Average spatial correlation length in units of hexagons.
    hex_radius : float
        Circumradius of a hexagon in SVG user units (pixels).
    seed : int, optional
        RNG seed for reproducibility.
    output_path : str
        Destination ``.svg`` file path.
    text_mode : {"hex", "text"}
        How an optional text overlay is rendered.  ``"hex"`` (default,
        original behaviour) burns the glyphs into the hex grid by
        recolouring affected hexes.  ``"text"`` adds a single SVG
        ``<text>`` element drawn on top of the camouflage, which keeps
        glyph edges crisp and respects ``text_font`` (font-family),
        ``text_height`` (height in hex units), ``text_rotation``
        (degrees), ``text_color`` and ``text_opacity`` (0..1).
    text_rotation : float
        Rotation in degrees, applied around the text's anchor point.
        Only used when ``text_mode == "text"``.
    text_opacity : float
        Fill opacity in 0..1.  Only used when ``text_mode == "text"``.
    text_spacing_x, text_spacing_y : float
        Distance between consecutive copies of the repeated text, in hex
        units, measured along the text's baseline (``x``) and
        perpendicular to it (``y``).  One hex unit equals one hex row
        pitch (``1.5 * hex_radius`` pixels), the same convention used by
        ``text_height``, so e.g. ``text_spacing_y == text_height`` gives
        copies that are stacked flush vertically with no gap.  ``0``
        (default) means "no tiling on this axis"; any positive value
        causes the text to be tiled along that axis to cover the entire
        canvas, with copies centred on the resolved anchor.  Only used
        when ``text_mode == "text"``.
    """
    if len(colors) != len(proportions):
        raise ValueError("`colors` and `proportions` must have equal length")
    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be positive")

    rng = np.random.default_rng(seed)
    # One independent correlated field per colour, then argmax with
    # calibrated per-colour offsets.  This lets blobs of any colour border
    # blobs of any other colour (rather than forming a strict quantile
    # ordering, where e.g. the darkest colour could only touch the second
    # darkest).
    color_idx = assign_colors_by_argmax(
        nx=nx,
        ny=ny,
        proportions=proportions,
        correlation_length=correlation_length,
        rng=rng,
    )

    # Clean up any 1-hex "confetti" blobs by recolouring each isolated cell
    # to the most common colour among its 6 hex neighbours.
    if remove_singletons:
        color_idx = remove_singleton_blobs(color_idx)

    # Burn optional hex-rendered text on top of the camouflage.  The text
    # colour is appended to the palette as a new entry so it cannot clash
    # with the camo colours.  When `text_mode == "text"`, the text is
    # emitted later as an SVG <text> element instead.
    colors = list(colors)
    if text and text_mode == "hex":
        if text_color is None:
            raise ValueError("text_color must be provided when `text` is set")
        mask = render_text_mask(text, int(round(text_height)), text_font)
        anchor = _resolve_text_anchor(text_position, color_idx.shape, mask.shape)
        colors.append(tuple(int(v) for v in text_color))
        color_idx = burn_text_into_grid(color_idx, mask, len(colors) - 1, anchor)
    elif text and text_mode == "text":
        if text_color is None:
            raise ValueError("text_color must be provided when `text` is set")
    elif text:
        raise ValueError(
            f"text_mode must be 'hex' or 'text' (got {text_mode!r})"
        )

    # Pointy-top hex dimensions.
    w = math.sqrt(3.0) * hex_radius        # full width
    h = 2.0 * hex_radius                   # full height
    row_spacing = 0.75 * h                 # 1.5 * r

    canvas_w = nx * w + w / 2.0            # +w/2 so odd rows fit
    canvas_h = (ny - 1) * row_spacing + h

    lines: list[str] = []
    lines.append('<?xml version="1.0" encoding="UTF-8" standalone="no"?>')
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{canvas_w:.2f}" height="{canvas_h:.2f}" '
        f'viewBox="0 0 {canvas_w:.2f} {canvas_h:.2f}" '
        f'shape-rendering="crispEdges">'
    )
    # Background = most common colour, so any sub-pixel seams are invisible.
    dominant = int(np.argmax(np.bincount(color_idx.ravel(), minlength=len(colors))))
    lines.append(f'  <rect width="100%" height="100%" fill="{_rgb(colors[dominant])}"/>')

    for row in range(ny):
        x_offset = (w / 2.0) if (row % 2 == 1) else 0.0
        cy = row * row_spacing + hex_radius
        for col in range(nx):
            cx = col * w + (w / 2.0) + x_offset
            ci = int(color_idx[row, col])
            pts = _hex_vertices(cx, cy, hex_radius)
            pts_str = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
            lines.append(
                f'  <polygon points="{pts_str}" fill="{_rgb(colors[ci])}"/>'
            )

    # Optional pure-SVG text overlay (text_mode="text").
    if text and text_mode == "text":
        # text_height is given in hex units; one hex row pitch is
        # row_spacing pixels, so this matches the on-canvas size you would
        # get from text_mode="hex" with the same numeric height.
        font_size_px = max(1.0, float(text_height) * row_spacing)
        tx, ty, anchor_attr, baseline_attr = _resolve_text_pixel_position(
            text_position, canvas_w, canvas_h, hex_radius, w, row_spacing
        )
        opacity = max(0.0, min(1.0, float(text_opacity)))

        # User-specified inter-copy spacing in hex units (see docstring),
        # converted to canvas pixels.  Zero on an axis disables tiling
        # on that axis (single copy along the centre line).
        step_x = max(0.0, float(text_spacing_x)) * row_spacing
        step_y = max(0.0, float(text_spacing_y)) * row_spacing

        # Decide how many copies to draw on each axis so that the rotated
        # tiling fully covers the canvas.  We work in the text-local
        # frame (where the grid is axis-aligned), so the canvas corners
        # have to be projected into that frame to find the required
        # extent on each side of the anchor.
        theta = math.radians(float(text_rotation))
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        corners = (
            (0.0 - tx, 0.0 - ty),
            (canvas_w - tx, 0.0 - ty),
            (0.0 - tx, canvas_h - ty),
            (canvas_w - tx, canvas_h - ty),
        )
        # canvas → local: rotate by -theta around the anchor.
        local_x = [dx * cos_t + dy * sin_t for dx, dy in corners]
        local_y = [-dx * sin_t + dy * cos_t for dx, dy in corners]
        max_local_x = max(abs(v) for v in local_x)
        max_local_y = max(abs(v) for v in local_y)

        if step_x > 0:
            n_each_side_x = int(math.ceil(max_local_x / step_x))
            x_indices = list(range(-n_each_side_x, n_each_side_x + 1))
        else:
            x_indices = [0]
        if step_y > 0:
            n_each_side_y = int(math.ceil(max_local_y / step_y))
            y_indices = list(range(-n_each_side_y, n_each_side_y + 1))
        else:
            y_indices = [0]

        # Sanity cap: if the spacing is so small that we'd emit a
        # ridiculous number of copies, refuse rather than producing a
        # multi-megabyte SVG.
        total_copies = len(x_indices) * len(y_indices)
        if total_copies > 20000:
            raise ValueError(
                f"text_spacing_x/y too small: would produce "
                f"{total_copies} text copies. Increase the spacing."
            )

        common_attrs = [
            f'fill="{_rgb(text_color)}"',
            f'fill-opacity="{opacity:.4f}"',
            f'text-anchor="{anchor_attr}"',
            f'dominant-baseline="{baseline_attr}"',
            f'font-size="{font_size_px:.2f}"',
        ]
        if text_font:
            # In text mode, text_font is interpreted as a CSS font-family
            # (e.g. "Arial", "Helvetica", "DejaVu Sans").  If the caller
            # passed a TTF path, fall back to its filename stem so SVG
            # viewers at least try a sensibly named family.
            family = text_font
            if os.path.sep in family or family.lower().endswith((".ttf", ".otf", ".ttc")):
                family = os.path.splitext(os.path.basename(family))[0]
            common_attrs.append(f'font-family="{_xml_escape(family)}"')

        # Wrap copies in a group so a single rotate transform applies to
        # the whole tiling.  Without rotation, skip the wrapper.
        rotated = bool(text_rotation)
        if rotated:
            lines.append(
                f'  <g transform="rotate({float(text_rotation):.4f} '
                f'{tx:.2f} {ty:.2f})">'
            )
        prefix = "    " if rotated else "  "
        escaped_text = _xml_escape(text)
        for j in y_indices:
            oy = j * step_y
            for i in x_indices:
                ox = i * step_x
                x = tx + ox
                y = ty + oy
                lines.append(
                    f'{prefix}<text x="{x:.2f}" y="{y:.2f}" '
                    f'{" ".join(common_attrs)}>{escaped_text}</text>'
                )
        if rotated:
            lines.append('  </g>')

    lines.append('</svg>')

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
# Each palette entry is (colors, proportions, default_correlation_length).
# The correlation length is the recommended blob scale (in hex units) and
# is used when the CLI --correlation flag is omitted.
#
# The "m05*" family is themed on the Finnish Defence Forces M05 digital
# camouflage; the colours are tuned by eye against fabric photographs and
# the variants share a correlation length of ~2.2, giving the chunky
# blob scale of the real fabric.  The non-M05 entries (woodland, desert,
# urban, navy) are generic four-tone schemes.
_PALETTES: dict[str, tuple[list[tuple[int, int, int]], list[float], float]] = {
    # Generic four-tone schemes -------------------------------------------
    "woodland": (
        # dark olive, mid green, tan, near-black
        [(56, 68, 45), (110, 120, 70), (160, 140, 90), (35, 28, 20)],
        [0.40, 0.30, 0.20, 0.10],
        3.0,
    ),
    "desert": (
        # light sand, mid sand, brown, dark brown
        [(196, 170, 120), (160, 130, 85), (120, 95, 60), (85, 70, 50)],
        [0.45, 0.30, 0.15, 0.10],
        3.0,
    ),
    "urban": (
        # light grey, mid grey, dark grey, near-black
        [(210, 210, 210), (150, 150, 150), (90, 90, 90), (35, 35, 35)],
        [0.35, 0.35, 0.20, 0.10],
        3.0,
    ),
    "navy": (
        # navy, mid blue, light blue, near-black
        [(30, 45, 75), (60, 85, 120), (120, 150, 180), (15, 20, 35)],
        [0.40, 0.30, 0.20, 0.10],
        3.0,
    ),

    # Finnish M05 family --------------------------------------------------
    # Standard M05 woodland: yellow-green highlight, cool dark olive
    # (dominant), warm muted tan, cool near-black with a bluish cast.
    "m05woodland": (
        [(138, 148, 78), (62, 78, 46), (152, 118, 86), (44, 50, 58)],
        [0.25, 0.30, 0.25, 0.20],
        2.2,
    ),
    # M05 snow: mostly white with sparse dark olive accents -- meant to
    # sit under a snow overlay or be used as the lightest M05 variant.
    "m05snow": (
        [(62, 78, 46), (255, 255, 255), (255, 255, 255), (255, 255, 255)],
        [0.2, 0.2, 0.1, 0.5],
        2.2,
    ),
    # M05 arid: tan-dominant variant with a single olive accent breaking
    # up the warm sand tones.  Equal proportions across the four colours.
    "m05arid": (
        [(216, 187, 155), (62, 78, 46), (216, 187, 155), (152, 118, 86)],
        [0.25, 0.25, 0.25, 0.25],
        2.2,
    ),
    # M05 winter woodland: pale-grey "snow-on-bark" highlight added on
    # top of the standard olive / tan / dark M05 colours.
    "m05winterwood": (
        [(218, 220, 220), (62, 78, 46), (152, 118, 86), (44, 50, 58)],
        [0.2, 0.3, 0.3, 0.2],
        2.2,
    ),
    # Force-on-force "red team" identifier: three reds and one near-black,
    # using the M05 blob scale -- intended for training/airsoft use.
    "m05badguy": (
        [(255, 0, 0), (255, 0, 0), (255, 0, 0), (44, 50, 58)],
        [0.2, 0.3, 0.3, 0.2],
        2.2,
    ),
    # Force-on-force "blue team" identifier: three blues and one
    # near-black, mirror image of the red-team palette.
    "m05goodguy": (
        [(0, 0, 255), (0, 0, 255), (0, 0, 255), (44, 50, 58)],
        [0.2, 0.3, 0.3, 0.2],
        2.2,
    ),
    # M05 night / very dark variant: three near-equal dark olives plus
    # the standard M05 near-black, for low-light scenes.
    "m05black": (
        [(21, 41, 30), (42, 44, 36), (28, 30, 24), (44, 50, 58)],
        [0.2, 0.3, 0.3, 0.2],
        2.2,
    ),
}


def _parse_color(s: str) -> tuple[int, int, int]:
    parts = s.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Colour must look like 'R,G,B' (got {s!r})"
        )
    r, g, b = (int(p) for p in parts)
    for v in (r, g, b):
        if not 0 <= v <= 255:
            raise argparse.ArgumentTypeError("Colour channels must be in 0..255")
    return r, g, b


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--width",  type=int, default=256, help="hexagons wide")
    ap.add_argument("--height", type=int, default=256, help="hexagons tall")
    ap.add_argument("--correlation", type=float, default=None,
                    help="average correlation length, in hexagons "
                         "(default: per-palette recommended value)")
    ap.add_argument("--hex-size", type=float, default=18.0,
                    help="hexagon circumradius in px")
    ap.add_argument("--palette", choices=sorted(_PALETTES), default="m05woodland",
                    help="named palette to use when --color is omitted "
                         "(default: m05woodland)")
    ap.add_argument("--color", type=_parse_color, action="append",
                    help="repeatable: an 'R,G,B' colour, pair with --proportion")
    ap.add_argument("--proportion", type=float, action="append",
                    help="repeatable: relative frequency of the matching --color")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed")
    ap.add_argument("--output", default="camouflage.svg", help="output .svg path")
    ap.add_argument("--keep-singletons", action="store_true",
                    help="skip the post-processing pass that recolours "
                         "single-hex blobs into their dominant neighbour")
    ap.add_argument("--text", type=str, default=None,
                    help="text string to overlay on the camouflage")
    ap.add_argument("--text-mode", choices=("hex", "text"), default="hex",
                    help="'hex' burns the text into the hex grid (default); "
                         "'text' draws a real SVG <text> element on top "
                         "(supports rotation, opacity, system fonts)")
    ap.add_argument("--text-color", type=_parse_color, default=None,
                    help="text colour as 'R,G,B' (required if --text is given)")
    ap.add_argument("--text-height", type=float, default=10.0,
                    help="height of the text, measured in hexagons")
    ap.add_argument("--text-position", default="center",
                    help="'top' / 'center' / 'bottom' (horizontally centred) "
                         "or 'row,col' for the top-left hex of the text")
    ap.add_argument("--text-font", default=None,
                    help="font: a TrueType path in --text-mode=hex, or a "
                         "CSS font-family name (e.g. 'Arial') in "
                         "--text-mode=text. Defaults to a system sans-serif.")
    ap.add_argument("--text-rotation", type=float, default=0.0,
                    help="rotation in degrees (only with --text-mode=text)")
    ap.add_argument("--text-opacity", type=float, default=1.0,
                    help="text opacity 0..1 (only with --text-mode=text)")
    ap.add_argument("--text-spacing-x", type=float, default=0.0,
                    help="spacing in hex units between copies of the "
                         "text along its own baseline direction; 0 means "
                         "no horizontal tiling. Any positive value tiles "
                         "the text to cover the canvas, centred on the "
                         "anchor (only with --text-mode=text)")
    ap.add_argument("--text-spacing-y", type=float, default=0.0,
                    help="spacing in hex units between copies of the "
                         "text perpendicular to its baseline; 0 means "
                         "no vertical tiling. Any positive value tiles "
                         "the text to cover the canvas, centred on the "
                         "anchor (only with --text-mode=text)")
    args = ap.parse_args(argv)

    if args.color or args.proportion:
        if not (args.color and args.proportion) or len(args.color) != len(args.proportion):
            ap.error("--color and --proportion must be given the same number of times")
        colors = args.color
        proportions = args.proportion
        palette_correlation = 1.0            # reasonable fallback for custom palettes
    else:
        colors, proportions, palette_correlation = _PALETTES[args.palette]

    correlation = args.correlation if args.correlation is not None else palette_correlation

    if args.text and args.text_color is None:
        ap.error("--text requires --text-color")

    text_position: tuple[int, int] | str
    pos = args.text_position
    if pos in ("top", "center", "centre", "middle", "bottom",
               "top-center", "bottom-center"):
        text_position = pos
    else:
        try:
            r_str, c_str = pos.split(",")
            text_position = (int(r_str), int(c_str))
        except ValueError:
            ap.error(f"--text-position must be a name or 'row,col' (got {pos!r})")

    out = generate_camouflage_svg(
        nx=args.width,
        ny=args.height,
        colors=colors,
        proportions=proportions,
        correlation_length=correlation,
        hex_radius=args.hex_size,
        seed=args.seed,
        output_path=args.output,
        remove_singletons=not args.keep_singletons,
        text=args.text,
        text_color=args.text_color,
        text_height=args.text_height,
        text_position=text_position,
        text_font=args.text_font,
        text_mode=args.text_mode,
        text_rotation=args.text_rotation,
        text_opacity=args.text_opacity,
        text_spacing_x=args.text_spacing_x,
        text_spacing_y=args.text_spacing_y,
    )
    print(f"Wrote {out}  ({args.width}x{args.height} hexes, palette={args.palette})")


if __name__ == "__main__":
    main()
