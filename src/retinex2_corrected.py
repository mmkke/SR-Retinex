#!/usr/bin/env python3
"""
Recursive Retinex (Zhang et al., ICWAPR 2011) + ISD-based illumination equalization.

"""

from __future__ import annotations

import argparse
import numpy        as np
import cv2
import json

from dataclasses    import dataclass
from pathlib        import Path
from typing         import Tuple
from pprint         import pprint

try:
    from metrics import ChromaticityMetrics
except ImportError:
    ChromaticityMetrics = None

from model_utils import load_model, ISDMapEstimator


LOG_16BIT_MAX = np.log(65535.0)


# -----------------------------
# Utilities
# -----------------------------
def read_image_float(path: str) -> np.ndarray:
    """
    Read image via OpenCV.
    Returns float32:
      - uint8  -> [0,1]
      - uint16 -> [0,1]
      - float  -> unchanged
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    else:
        img = img.astype(np.float32)

    return img


def save_image_float(path: str, img: np.ndarray, channel_order: str = "bgr") -> None:
    """Save float image to PNG (8-bit) for quick inspection."""
    if channel_order == "rgb" and img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    out = np.clip(img, 0.0, 1.0)
    out8 = (out * 255.0 + 0.5).astype(np.uint8)
    cv2.imwrite(path, out8)


def rgb_to_luma(img_rgb: np.ndarray) -> np.ndarray:
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError("Expected (H,W,3) image.")
    r, g, b = img_rgb[..., 0], img_rgb[..., 1], img_rgb[..., 2]
    return 0.299 * r + 0.587 * g + 0.114 * b


def shift2d_reflect(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift by (dy,dx) with reflect padding, keeping shape."""
    H, W = a.shape
    pad_y = abs(dy)
    pad_x = abs(dx)
    if pad_y == 0 and pad_x == 0:
        return a.copy()

    ap = np.pad(a, ((pad_y, pad_y), (pad_x, pad_x)), mode="reflect")
    y0 = pad_y + dy
    x0 = pad_x + dx
    return ap[y0:y0 + H, x0:x0 + W]

def gamma_correct_16bit(image: np.ndarray, gamma: float = 2.2, preserve_dtype: bool = True) -> np.ndarray:
    """
    Apply standard gamma correction to a 16-bit linear image.

    Parameters
    ----------
    image : np.ndarray
        Input image:
            - uint16 in [0, 65535], or
            - float in [0,1]
    gamma : float
        Gamma value (e.g. 2.2 for display gamma).
    preserve_dtype : bool
        If True and input is uint16, output will be uint16.
        Otherwise returns float32 in [0,1].

    Returns
    -------
    np.ndarray
        Gamma-corrected image.
    """

    if gamma <= 0:
        raise ValueError("Gamma must be positive.")

    # Convert to float in [0,1]
    if image.dtype == np.uint16:
        img = image.astype(np.float32) / 65535.0
        input_was_uint16 = True
    else:
        img = image.astype(np.float32)
        input_was_uint16 = False

    img = np.clip(img, 0.0, 1.0)

    # Standard gamma correction
    img_gamma = np.power(img, 1.0 / gamma)

    img_gamma = np.clip(img_gamma, 0.0, 1.0)

    if preserve_dtype and input_was_uint16:
        return (img_gamma * 65535.0 + 0.5).astype(np.uint16)
    else:
        return img_gamma.astype(np.float32)


def linear16_to_srgb(image: np.ndarray, preserve_dtype: bool = True,) -> np.ndarray:
    """
    Convert linear 16-bit image to sRGB (standard piecewise transfer).

    Parameters
    ----------
    image : np.ndarray
        Either:
            - uint16 in [0,65535], or
            - float32 in [0,1] (linear)
    preserve_dtype : bool
        If True and input is uint16, output will be uint16.
        Otherwise returns float32 in [0,1].

    Returns
    -------
    np.ndarray
        sRGB image.
    """

    # Convert to float in [0,1]
    if image.dtype == np.uint16:
        img = image.astype(np.float32) / 65535.0
        input_was_uint16 = True
    else:
        img = image.astype(np.float32)
        input_was_uint16 = False

    img = np.clip(img, 0.0, 1.0)

    # sRGB piecewise transfer
    threshold = 0.0031308

    srgb = np.where(
        img <= threshold,
        12.92 * img,
        1.055 * np.power(img, 1.0 / 2.4) - 0.055
    )

    srgb = np.clip(srgb, 0.0, 1.0)

    if preserve_dtype and input_was_uint16:
        return (srgb * 65535.0 + 0.5).astype(np.uint16)
    else:
        return srgb.astype(np.float32)

def linear16_to_log_normalized(image_01: np.ndarray) -> np.ndarray:
    """
    Convert float image in [0,1] that represents 16-bit linear into normalized log space [0,1],
    using ZERO MASKING (no eps). Zeros remain exactly zero.
    """
    img = image_01.astype(np.float32)
    img = np.clip(img, 0.0, 1.0)

    img16 = img * 65535.0
    log_img = np.zeros_like(img16, dtype=np.float32)

    mask = img16 > 0
    log_img[mask] = np.log(img16[mask]) / LOG_16BIT_MAX

    # optional safety
    log_img = np.clip(log_img, 0.0, 1.0)
    if np.any(log_img < -1e-6) or np.any(log_img > 1.0 + 1e-6):
        print(f"log_img max = {np.max(log_img)}, log_img min = {np.min(log_img)}")
        raise ValueError("log_img out of expected [0,1] range after normalization.")
    return log_img.astype(np.float32)


def log_normalized_to_linear16(log_img: np.ndarray) -> np.ndarray:
    """
    Inverse of linear16_to_log_normalized.
    Input: normalized log in [0,1]
    Output: float linear in [0,1], with zeros preserved.
    """
    x = log_img.astype(np.float32)
    x = np.clip(x, 0.0, 1.0)

    lin16 = np.zeros_like(x, dtype=np.float32)
    mask = x > 0
    lin16[mask] = np.exp(x[mask] * LOG_16BIT_MAX)

    # Back to [0,1]
    lin01 = np.clip(lin16 / 65535.0, 0.0, 1.0)
    return lin01


def load_isd_map(path: str, expected_shape_hw: Tuple[int, int]) -> np.ndarray:
    """
    Load ISD map from file. Expected to be 3-channel 16-bit (or float) with shape (H,W,3).
    Returns float32 in original units (not necessarily unit-length); you can normalize later.
    """
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"Could not read ISD map: {path}")

    if m.ndim != 3 or m.shape[2] != 3:
        raise ValueError(f"ISD map must be 3-channel (H,W,3). Got {m.shape} from {path}")

    H, W = expected_shape_hw
    if (m.shape[0], m.shape[1]) != (H, W):
        raise ValueError(f"ISD map shape {m.shape[:2]} does not match image shape {(H, W)}")

    # Convert to float32 and scale if needed
    if m.dtype == np.uint16:
        m = m.astype(np.float32) / 65535.0
    else:
        m = m.astype(np.float32)

    return m


# -----------------------------
# Retinex implementation
# -----------------------------
@dataclass
class RecursiveRetinexParams:
    beta: float = 10.0
    n_levels: int | None = None
    iters_per_level: int = 1
    weight_eps: float = 1e-6  # ONLY for weights to avoid division by zero (not for log)


def recursive_retinex(
    S: np.ndarray,
    params: RecursiveRetinexParams = RecursiveRetinexParams(),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Recursive Retinex illumination–reflectance decomposition
    using the multi-scale update formulation of Zhang et al. (ICWAPR 2011).

    This implementation operates on a single-channel intensity image and
    estimates a smooth illumination field in the log domain via iterative
    weighted neighbor interactions across multiple spatial scales.

    The model assumes:
        S(x, y) = L(x, y) * R(x, y)

    In log space:
        s = l + r

    where:
        s = log(S)
        l = log(L)  (illumination)
        r = log(R)  (reflectance)

    The illumination estimate `l` is obtained recursively by combining:
        • a data-fidelity term (agreement with original log intensity s)
        • a spatial smoothness term (agreement with neighbors at distance d)

    For each spatial scale d = 2^level:
        term1 = (β * l + s) / (1 + β)
        term2 = (β * d * l + D) / (1 + β * d)

    The final update is a weighted average of the minimum of these terms,
    where weights are inversely proportional to squared illumination differences:

        w_i = 1 / ((D - l)^2 + weight_eps)

    The algorithm proceeds from coarse to fine scales.

    Parameters
    ----------
    S : np.ndarray
        2D non-negative intensity image (H, W).
        Typically luma derived from RGB. Must be >= 0.
        Zero values are handled via zero-masked log (no epsilon added).

    params : RecursiveRetinexParams
        Configuration parameters controlling:
            beta              → smoothness vs data fidelity balance
            n_levels          → number of spatial scales
            iters_per_level   → iterations per scale
            weight_eps        → numerical stability constant

    Returns
    -------
    l_log : np.ndarray
        Log-domain illumination estimate (H, W).

    r_log : np.ndarray
        Log-domain reflectance estimate (H, W).

    L_lin : np.ndarray
        Linear-domain illumination field:
            L = exp(l_log)

    R_lin : np.ndarray
        Linear-domain reflectance field:
            R = exp(r_log)
    """
    if S.ndim != 2:
        raise ValueError("S must be 2D (H,W).")
    if np.any(S < 0):
        raise ValueError("S must be nonnegative.")

    beta = float(params.beta)
    w_eps = float(params.weight_eps)

    S = S.astype(np.float32)

    s = np.zeros_like(S, dtype=np.float32)
    mS = S > 0
    s[mS] = np.log(S[mS])

    l = s.copy()

    H, W = S.shape
    max_dim = max(H, W)

    if params.n_levels is None:
        n_levels = int(np.floor(np.log2(max_dim))) + 1
    else:
        n_levels = int(params.n_levels)

    iters_per_level = int(params.iters_per_level)

    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for level in reversed(range(n_levels)):
        d = 2 ** level
        if d <= 0:
            continue

        for _ in range(iters_per_level):
            m_list = []
            w_list = []

            term1 = (l * beta + s) / (1.0 + beta)

            for dy, dx in dirs:
                D = shift2d_reflect(l, dy * d, dx * d)

                term2 = (l * beta * d + D) / (1.0 + beta * d)
                m_i = np.minimum(term1, term2)

                w_i = 1.0 / ((D - l) ** 2 + w_eps)

                m_list.append(m_i)
                w_list.append(w_i)

            m_stack = np.stack(m_list, axis=0)
            w_stack = np.stack(w_list, axis=0)

            num = np.sum(w_stack * m_stack, axis=0)
            den = np.sum(w_stack, axis=0)

            l = num / (den + w_eps)

    r = s - l
    L = np.exp(l).astype(np.float32)
    R = np.exp(r).astype(np.float32)
    return l.astype(np.float32), r.astype(np.float32), L, R


# -----------------------------
# ISD equalization
# -----------------------------

def alpha_from_gamma_tonecurve_on_illumination(
    L: np.ndarray,                 # (H,W) Retinex illumination (linear, >0)
    *,
    gamma: float = 2.2,            # >1 brightens shadows
    L_norm_percentile: float = 99.5,
    only_brighten: bool = True,
    eps: float = 1e-6,
    boost: float = 1.0
) -> tuple[np.ndarray, dict]:
    """
    Build an alpha map by applying a gamma tone curve to a scalar illumination field
    and interpreting the required change as a log-space shift magnitude.

    This does NOT gamma-correct the RGB image. It gamma-remaps illumination intensity only.

    Returns
    -------
    alpha : (H,W) float32
        Normalized-log shift magnitude to apply along ISD.
    info : dict
        Debug outputs.
    """
    if gamma <= 0:
        raise ValueError("gamma must be > 0")

    L = L.astype(np.float32)

    # normalize illumination to [0,1]-ish
    denom = np.percentile(L, L_norm_percentile) + eps
    I = np.clip(L / denom, 0.0, 1.0).astype(np.float32)

    # gamma tone curve target
    I_tgt = np.power(I, 1.0 / gamma).astype(np.float32) * boost

    # convert to gain
    gain = (I_tgt + eps) / (I + eps)
    if only_brighten:
        gain = np.maximum(gain, 1.0)

    # convert to normalized-log alpha
    alpha = (np.log(gain) / LOG_16BIT_MAX).astype(np.float32)

    info = {"I": I, "I_tgt": I_tgt, "gain": gain, "alpha": alpha}
    return alpha, info

def equalize_illumination_along_isd(
    log_img: np.ndarray,     # (H,W,3) normalized log in [0,1]
    isd_map: np.ndarray,     # (H,W,3) illumination direction (any scale)
    L: np.ndarray,           # (H,W) Retinex illumination (linear)
    *,
    only_brighten: bool = True,
    max_abs_alpha_normlog: float = 0.4,
    eps: float = 1e-6,
    gamma = 2.2,
    soften: bool = False,
    p: float = 70.0,
    softness: float = 0.06
) -> tuple[np.ndarray, dict]:
    """
    Equalize illumination by applying a per-pixel gain derived from L,
    implemented as a shift along ISD in *normalized-log* space.

    Critical fix: alpha is converted from ln(gain) to normalized-log units by dividing by ln(65535).
    """
    if log_img.ndim != 3 or log_img.shape[2] != 3:
        raise ValueError("log_img must be (H,W,3)")
    if isd_map.shape != log_img.shape:
        raise ValueError(f"isd_map shape {isd_map.shape} must match log_img {log_img.shape}")
    if L.shape != log_img.shape[:2]:
        raise ValueError("L must match (H,W)")

    L = L.astype(np.float32)

    # normalize illumination for threshold/target selection
    L_n = L / (np.percentile(L, 99.5) + eps)
    L_n = np.clip(L_n, 1e-3, 10.0)

    alpha, dbg = alpha_from_gamma_tonecurve_on_illumination(
                                                        L, 
                                                        only_brighten=only_brighten, 
                                                        gamma=gamma)

    # Mask softening
    if soften:
        L_n = L / (np.percentile(L, 99.5) + eps)
        L_star = np.percentile(L_n, p)
        w = (L_star - L_n) / softness
        w = np.clip(w, 0.0, 1.0)
        alpha = alpha * w
    
    # Blur alpha
    alpha_blur_sigma = 1.2
    if alpha_blur_sigma is not None and alpha_blur_sigma > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=alpha_blur_sigma, sigmaY=alpha_blur_sigma)
    
    # Clamp in normalized-log units
    alpha = np.clip(alpha, -max_abs_alpha_normlog, max_abs_alpha_normlog)

    # Normalize ISD vectors
    n = np.linalg.norm(isd_map, axis=2, keepdims=True)
    unit_isd = isd_map / np.maximum(n, eps)

    log_out = log_img.astype(np.float32) + alpha[..., None] * unit_isd

    info = {
        "alpha": alpha,
        "max_abs_alpha_normlog": max_abs_alpha_normlog,
        "only_brighten": only_brighten,
        "gamma": gamma
    }

    return log_out, info


# -----------------------------
# CLI
# -----------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Recursive Retinex (Eq. 9-10) + ISD equalization.")
    p.add_argument("--input", "-i", type=str, default="data/submission_images/tang_yilin_033.tif", help="Input image path.")
    p.add_argument("--output_dir", "-o", type=str, default="retinex2_output", help="Directory to write outputs.")
    p.add_argument("--isd_map", type=str, default="data/sr_maps/tang_yilin_033_isd.tiff", help="Path to ISD map image (H,W,3), uint16 or float.")
    p.add_argument("--beta", type=float, default=1.0, help="Interaction parameter beta.")
    p.add_argument("--weight_eps", type=float, default=1e-6, help="Stability eps for weights/denominator only.")
    p.add_argument("--iters_per_level", type=int, default=1, help="Iterations per distance level.")
    p.add_argument("--n_levels", type=int, default=0, help="Number of levels. 0 means auto from image size.")
    p.add_argument("--mode", type=str, default="luma", choices=["luma", "channel"], 
                        help="How to compute intensity S for Retinex.")
    p.add_argument("--channel", type=int, default=2, help="If mode=channel, which channel index 0/1/2.")
    p.add_argument("--max_abs_alpha", type=float, default=0.4, help="Clamp for ISD shift (normalized-log units).")
    p.add_argument("--only_brighten", action="store_true", help="Only brighten (do not darken).")
    p.add_argument("--gamma", type=float, default=2.2, help="Adds a nonlinear boost to shadowed areas.")
    p.add_argument("--soften", action="store_true", help="Apply mask softening.")
    p.add_argument("--softness", type=float, default=0.06, help="")
    p.add_argument("--percentile", type=float, default=70, help="")
    p.add_argument("--use_model", action="store_true", help="Uses model to generate spectral ratio map from image")
    p.add_argument("--model_type", type=str, default="vit", choices=["vit", "resnet"])
    p.add_argument("--chatgpt", action="store_true", help="Compare results to those generated by ChatGPT")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    if ChromaticityMetrics is None:
        raise ImportError(
            "ChromaticityMetrics is unavailable. Add src/metrics.py "
            "or adjust the evaluation step."
        )

    base_dir = Path(args.output_dir)
    stem = Path(args.input).stem
    out_dir = base_dir / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    img_bgr = read_image_float(args.input)  # (H,W) or (H,W,3) in [0,1]
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError(
            f"This pipeline requires a 3-channel color image. "
            f"Got shape {img_bgr.shape} from {args.input}."
        )
    save_image_float(str(out_dir / "original_linear.png"), img_bgr)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_srgb = linear16_to_srgb(img_rgb)
    save_image_float(str(out_dir / "original_srgb.png"), img_srgb, channel_order='rgb')

    # Get ISD map
    if args.use_model:
        model, model_path = load_model(model_type=args.model_type)
        estimator = ISDMapEstimator(
            model = model,
            model_path = model_path,
            device = "cpu"
            )
        isd_map_rgb, _ = estimator.predict(img_rgb)
        # print("Model type: ", args.model_type)
        # print(f" Predicated Map | Shape: {isd_map.shape} | Dtype: {isd_map.dtype}")
        # print(f" Predicated Map | Max: {isd_map.max(axis=(0,1))} | Min: {isd_map.min(axis=(0,1))}")
    else:
        isd_map_bgr = load_isd_map(args.isd_map, expected_shape_hw=img_bgr.shape[:2])
        isd_map_rgb = isd_map_bgr[..., ::-1]  # swap channels
        # isd_map_rgb = cv2.cvtColor(isd_map_bgr, cv2.COLOR_BGR2RGB)
    save_image_float(str(out_dir / "isd_map.png"), isd_map_rgb, channel_order="rgb")

    # Build intensity S
    if args.mode == "luma":
        S = rgb_to_luma(img_rgb)
    elif args.mode == "channel":
        c = int(args.channel)
        if c < 0 or c > 2:
            raise ValueError("--channel must be 0, 1, or 2")
        S = img_rgb[..., c]   
    else:
        raise ValueError(f"Unsupported mode: {args.mode}. Argument 'mode' must be either 'luma' or 'channel.")


    params = RecursiveRetinexParams(
        beta=args.beta,
        n_levels=None if args.n_levels == 0 else args.n_levels,
        iters_per_level=args.iters_per_level,
        weight_eps=args.weight_eps,
    )

    l_log, r_log, L, R = recursive_retinex(S, params)

    # Save illumination (visualization)
    L_vis = L / (np.percentile(L, 99.5) + 1e-6)
    L_vis = np.clip(L_vis, 0, 1)
    save_image_float(str(out_dir / "illumination_L.png"), L_vis)

    # Save reflectance visualization:
    # - If we have RGB input, compute color reflectance from scalar L

    R_rgb = img_rgb / (L[..., None] + 1e-6)
    R_vis = R_rgb / (np.percentile(R_rgb, 99.5) + 1e-6)
    R_vis = np.clip(R_vis, 0, 1)
    save_image_float(str(out_dir / "reflectance_rgb.png"), R_vis, channel_order="rgb")

    # Save log-domain maps normalized for display
    l_disp = (l_log - np.min(l_log)) / (np.max(l_log) - np.min(l_log) + 1e-6)
    r_disp = (r_log - np.min(r_log)) / (np.max(r_log) - np.min(r_log) + 1e-6)
    save_image_float(str(out_dir / "illumination_log_l.png"), l_disp)
    save_image_float(str(out_dir / "reflectance_log_r.png"), r_disp)

    # ISD equalization
    log_img_rgb = linear16_to_log_normalized(img_rgb)  # normalized log in [0,1]
    log_eq_rgb, dbg = equalize_illumination_along_isd(
        log_img=log_img_rgb,
        isd_map=isd_map_rgb,
        L=L,
        only_brighten=args.only_brighten,
        max_abs_alpha_normlog=args.max_abs_alpha,
        gamma=args.gamma,
        soften=args.soften,
        softness=args.softness,
        p=args.percentile
    )

    # Convert back to linear for viewing/saving
    lin_eq_rgb = log_normalized_to_linear16(log_eq_rgb)
    save_image_float(str(out_dir / "result_equalized.png"), lin_eq_rgb, channel_order="rgb")
    lin_eq_srgb = linear16_to_srgb(lin_eq_rgb)
    save_image_float(str(out_dir / "result_equalized_srgb.png"), lin_eq_srgb, channel_order="rgb")

    # Optional: also save alpha map for debugging
    alpha = dbg["alpha"]
    alpha_disp = (alpha - alpha.min()) / (alpha.max() - alpha.min() + 1e-6)
    save_image_float(str(out_dir / "alpha.png"), alpha_disp)

    # print("Wrote:")
    # for name in [
    #     "illumination_L.png",
    #     "reflectance_rgb.png" if img_bgr is not None else "reflectance_gray.png",
    #     "illumination_log_l.png",
    #     "reflectance_log_r.png",
    #     "result_equalized.png",
    #     "alpha.png",
    # ]:
    #     p = out_dir / name
    #     if p.exists():
    #         print(f"  {p}")
    a = dbg["alpha"]
    alpha_stats = {
                    "alpha": {
                        "min": float(a.min()),
                        "max": float(a.max()),
                        "mean": float(a.mean()),
                        "p95": float(np.percentile(a, 95)),
                        "p99": float(np.percentile(a, 99)),
                    },
                    "params": {
                        "max_abs_alpha_normlog": float(dbg["max_abs_alpha_normlog"]),
                        "gamma": float(dbg["gamma"]),
                        "only_brighten": bool(dbg["only_brighten"])
                    }
                }
    pprint(alpha_stats)

    metrics = ChromaticityMetrics()
    error = metrics.evaluate(img_rgb, lin_eq_rgb)
    maps = metrics.error_maps(img_rgb, lin_eq_rgb)
    chroma_l2 = maps["chroma_l2"]
    p = np.percentile(chroma_l2, 99)
    chroma_l2 = chroma_l2 / (p + 1e-12)
    chroma_l2 = np.clip(chroma_l2, 0, 1)
    save_image_float(str(out_dir / "chroma_l2.png"), chroma_l2)
    chroma_l2_log = np.log(maps["chroma_l2"] + 1e-8)
    chroma_l2_log = (chroma_l2_log - chroma_l2_log.min()) / (chroma_l2_log.max() - chroma_l2_log.min())
    save_image_float(str(out_dir / "chroma_l2_log.png"), chroma_l2_log)
    save_image_float(str(out_dir / "angular_error_deg.png"), maps["angular_error_deg"])


    # Get chromaticity metrics
    if args.chatgpt:
        chatgpt_bgr = read_image_float(f"data/chatgpt/{stem}.png")
        chatgpt_rgb = cv2.cvtColor(chatgpt_bgr, cv2.COLOR_BGR2RGB)
        save_image_float(str(out_dir / "chatgpt_result.png"), chatgpt_rgb, channel_order="rgb")
        chatgpt_srgb = linear16_to_srgb(chatgpt_rgb)
        save_image_float(str(out_dir / "chatgpt_result_srgb.png"), chatgpt_srgb, channel_order="rgb")
        chatgpt_error = metrics.evaluate(img_rgb, chatgpt_rgb)
    else:
        chatgpt_error = None

    # export summary
    summary = {
        "config": alpha_stats,
        "metrics": error,
        "chatgpt_metrics": chatgpt_error
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=4, default=float)
               
if __name__ == "__main__":
    main()
