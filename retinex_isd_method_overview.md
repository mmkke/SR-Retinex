# Recursive Retinex (Zhang 2011) + ISD-based Illumination Equalization


## 1) Read and normalize the input image

**Function:** `read_image_float(path)`

- Loads the image with OpenCV (`cv2.imread(..., IMREAD_UNCHANGED)`).
- Converts to `float32` and normalizes:
  - `uint16` → divide by **65535** → **linear** values in **[0, 1]**
  - `uint8`  → divide by **255**   → **[0, 1]**
  - float input → kept as float32

**Output:** `img` is `(H,W)` or `(H,W,3)` float32, assumed **linear** in `[0,1]`.

---

## 2) Compute a scalar intensity image `S` for Retinex

Retinex is run on a **single-channel** intensity map `S`.

**Options:**
- `--mode luma` (recommended): `S = 0.114 B + 0.587 G + 0.299 R` (via `rgb_to_luma`)
- `--mode channel`: pick one channel (0/1/2)
- `--mode gray`: OpenCV grayscale from an 8-bit conversion (works, but less “linear-faithful”)

**Output:** `S` is `(H,W)` float32.

---

## 3) Run recursive Retinex (Eq. 9 & 10) to estimate illumination `L`

**Function:** `recursive_retinex_eq9_eq10(S, params)`

### 3a) Convert intensity to log domain with **zero masking**
You build:

- `s = log(S)` **only where** `S > 0`
- if `S == 0`, then `s == 0` (no eps injected)

This avoids `log(0)` while preserving exact zeros.

### 3b) Initialize illumination estimate
- `l = s` initially

`l` is the **log-illumination estimate** that will be iteratively smoothed.

### 3c) Multi-scale recursion (large → small distances)
For each scale:
- `d = 2**level` (large distances first)
- for each of 8 directions (N,S,E,W + diagonals):
  - shift `l` by `(dy*d, dx*d)` using **reflect padding**: `D = shift2d_reflect(l, ...)`
  - compute two terms (your Eq. 9 structure):
    - `term1 = (l*beta + s)/(1+beta)`
    - `term2 = (l*beta*d + D)/(1+beta*d)`
  - directional candidate: `m_i = min(term1, term2)`
  - directional weight (Eq. 10):
    - `w_i = 1 / ((D - l)^2 + weight_eps)`

Then average directional candidates with weights:

\[
l \leftarrow \frac{\sum_i w_i m_i}{\sum_i w_i}
\]

### 3d) Recover reflectance (log and linear)
After recursion:
- `r = s - l`  (log reflectance)
- `L = exp(l)` (linear illumination)
- `R = exp(r)` (linear reflectance, scalar)

**Outputs:**
- `l_log` (H,W): log illumination
- `r_log` (H,W): log reflectance
- `L` (H,W): linear illumination estimate
- `R` (H,W): linear reflectance estimate (scalar)

---

## 4) Save diagnostic Retinex visualizations

### Illumination visualization
- `L_vis = L / percentile(L, 99.5)` then clipped to `[0,1]`
- Saved to `illumination_L.png`

This is purely for viewing.

### Reflectance visualization (color)
If the input is color:
- `R_rgb = img_bgr / (L[...,None] + eps)`
- normalized by a high percentile for display and clipped
- saved to `reflectance_rgb.png`

This is a *useful diagnostic* “reflectance-like” view.

---

## 5) Convert original image to **normalized log space** (16-bit dynamic range)

**Function:** `linear16_to_log_normalized(img_bgr)`

Purpose: represent linear 16-bit-like values in a log domain normalized to `[0,1]`.

Steps:
1. Assume `img_bgr` is linear in `[0,1]` representing 16-bit content
2. scale to 16-bit domain: `img16 = img * 65535`
3. compute log with **zero masking**:
   - where `img16 > 0`: `log_img = log(img16) / log(65535)`
   - where `img16 == 0`: `log_img = 0`

**Output:** `log_img` is `(H,W,3)` float32 in `[0,1]`.

---

## 6) Load ISD map and match channel order

**Function:** `load_isd_map(path, expected_shape_hw)`

- Loads a 3-channel map, checks shape matches `(H,W)`
- If `uint16`, scales to `[0,1]` by `/65535`
- Returns float32

Then you swap channels:
- `isd_map = isd_map[..., ::-1]`

This converts the SR/ISD map from **RGB file order** into **BGR order**, matching `img_bgr`.

---

## 7) Compute per-pixel gain from illumination and convert to an `alpha` map

**Function:** `equalize_illumination_along_isd(log_img, isd_map, L, ...)`

### 7a) Normalize illumination for robust targeting
- `L_n = L / percentile(L, 99.5)`
- clipped to avoid tiny/huge values

This makes thresholds and percentiles more stable across images.

### 7b) Choose target illumination level `L_star`
- `median`, `p70`, or explicit `value`

Interpretation:
- this is the desired “reference illumination level” to bring darker pixels up toward.

### 7c) Compute the multiplicative gain field
- `gain = (L_star + eps) / (L_n + eps)`
- if `only_brighten`: clamp `gain >= 1`

So:
- dark pixels (low `L_n`) get `gain > 1`
- bright pixels get `gain < 1`, but can be stopped from darkening.

### 7d) Convert gain into a normalized-log shift magnitude `alpha`
Since your pixel representation is:

\[
\text{log\_img} = \frac{\ln(I_{16})}{\ln(65535)}
\]

a multiplicative gain `gain` becomes an additive shift:

\[
\alpha = \frac{\ln(\text{gain})}{\ln(65535)}
\]

This is exactly:
- `alpha = log(gain) / LOG_16BIT_MAX`

### 7e) Optional soft gating (act mostly on shadows)
If `soften=True`:
- you compute a weight `w` that ramps from 1 in shadows to 0 near/above `L_star`
- then `alpha *= w`

This helps avoid flattening midtones/highlights.

### 7f) Scale + clamp
- `alpha *= alpha_scale`
- `alpha = clip(alpha, [-max_abs_alpha_normlog, +max_abs_alpha_normlog])`

This sets the maximum correction strength in normalized-log units.

---

## 8) Shift pixels along ISD direction in normalized-log space

You normalize ISD vectors per pixel:
- `unit_isd = isd_map / ||isd_map||`

Then apply the correction:
- `log_out = log_img + alpha[...,None] * unit_isd`

Interpretation:
- You move each pixel in **log-RGB** along the estimated illumination direction.
- The step size is **alpha**, derived from Retinex’s illumination gap to the target.

---

## 9) Convert back to linear for saving/preview

**Function:** `log_normalized_to_linear16(log_out)`

Inverse of the normalized log mapping:
- `I16 = exp(log_out * log(65535))` where `log_out > 0`
- scale back to `[0,1]` by dividing by 65535

You save `result_equalized.png` from this **linear** output.
