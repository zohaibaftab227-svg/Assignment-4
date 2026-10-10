"""
Advanced Computer Vision Project - Streamlit App
------------------------------------------------
1. Semantic segmentation with a U-Net (PyTorch, trained inside the app)
2. Evaluation of the segmentation with IoU and Dice
3. Short-video motion analysis: Lucas-Kanade tracking + Farneback optical flow
4. Advanced topic review: Vision Transformers (with an interactive patch demo)
5. One application (medical imaging) and one technical challenge (domain shift),
   including a live robustness experiment

Run locally:  streamlit run app.py
"""

import io
import os
import tempfile
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

torch.set_num_threads(2)


# ----------------------------------------------------------------------
# Streamlit version compatibility for st.image width
# ----------------------------------------------------------------------
def _ver(v):
    parts = []
    for p in v.split(".")[:2]:
        digits = "".join(ch for ch in p if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


if _ver(st.__version__) >= (1, 50):
    IMG_KW = {"width": "stretch"}
else:
    IMG_KW = {"use_container_width": True}

st.set_page_config(
    page_title="Advanced Computer Vision Project",
    page_icon="🧠",
    layout="wide",
)

IMG = 64  # synthetic image size for the U-Net (must be divisible by 8)


# ======================================================================
# 1) SYNTHETIC SEGMENTATION DATA
# ======================================================================
def make_sample(rng, size=IMG, noise=0.05, contrast=1.0):
    """One grayscale image with 1-3 random shapes (circle / rectangle /
    triangle) and its ground-truth binary mask."""
    bg = rng.uniform(0.2, 0.6)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    ang = rng.uniform(0, 2 * np.pi)
    grad = (np.cos(ang) * xx + np.sin(ang) * yy) / size * rng.uniform(-0.2, 0.2)
    img = bg + grad
    mask = np.zeros((size, size), np.uint8)

    for _ in range(int(rng.integers(1, 4))):
        m = np.zeros((size, size), np.uint8)
        kind = int(rng.integers(0, 3))
        cx = int(rng.integers(12, size - 12))
        cy = int(rng.integers(12, size - 12))
        r = int(rng.integers(6, 13))
        if kind == 0:
            cv2.circle(m, (cx, cy), r, 1, -1)
        elif kind == 1:
            cv2.rectangle(m, (cx - r, cy - r), (cx + r, cy + int(r * 0.7)), 1, -1)
        else:
            pts = np.array([[cx, cy - r], [cx - r, cy + r], [cx + r, cy + r]], np.int32)
            cv2.fillPoly(m, [pts], 1)
        sign = 1 if rng.random() < 0.5 else -1
        val = float(np.clip(bg + sign * rng.uniform(0.25, 0.4), 0, 1))
        img = np.where(m > 0, val, img)
        mask |= m

    img = (img - 0.5) * contrast + 0.5
    img = img + rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 1).astype(np.float32), mask.astype(np.float32)


def make_batch(rng, n, noise=0.05, contrast=1.0, augment=False):
    xs, ys = [], []
    for _ in range(n):
        if augment:  # random noise + contrast so the model becomes robust
            nz = rng.uniform(0.03, 0.35)
            ct = rng.uniform(0.25, 1.0)
        else:
            nz, ct = noise, contrast
        x, y = make_sample(rng, noise=nz, contrast=ct)
        xs.append(x)
        ys.append(y)
    X = np.stack(xs)[:, None].astype(np.float32)
    Y = np.stack(ys)[:, None].astype(np.float32)
    return X, Y


@st.cache_data(show_spinner=False)
def get_test_set(n, noise, contrast, seed=2024):
    rng = np.random.default_rng(seed)
    return make_batch(rng, n, noise=noise, contrast=contrast, augment=False)


# ======================================================================
# 2) U-NET
# ======================================================================
class DoubleConv(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """Small U-Net: 3 down-sampling stages, bottleneck, 3 up-sampling stages
    with skip connections."""

    def __init__(self, in_ch=1, out_ch=1, base=8):
        super().__init__()
        b = base
        self.enc1 = DoubleConv(in_ch, b)
        self.enc2 = DoubleConv(b, b * 2)
        self.enc3 = DoubleConv(b * 2, b * 4)
        self.bott = DoubleConv(b * 4, b * 8)
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.dec3 = DoubleConv(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.dec2 = DoubleConv(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.dec1 = DoubleConv(b * 2, b)
        self.out = nn.Conv2d(b, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        b = self.bott(F.max_pool2d(e3, 2))
        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.out(d1)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def seg_loss(logits, target):
    """BCE + Dice loss."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum()
    dice = 1 - (2 * inter + 1) / (p.sum() + target.sum() + 1)
    return bce + dice


def train_model(steps, batch_size, base, seed, augment, progress_cb=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = UNet(base=base)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    losses = []
    model.train()
    for s in range(steps):
        X, Y = make_batch(rng, batch_size, augment=augment)
        logits = model(torch.from_numpy(X))
        loss = seg_loss(logits, torch.from_numpy(Y))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(float(loss.item()))
        if progress_cb is not None and (s % 5 == 0 or s == steps - 1):
            progress_cb((s + 1) / steps, losses[-1])
    model.eval()
    return model, losses


def predict_probs(model, X):
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(X))).numpy()[:, 0]


# ======================================================================
# 3) METRICS
# ======================================================================
def iou_score(pred, gt, eps=1e-7):
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float((inter + eps) / (union + eps))


def dice_score(pred, gt, eps=1e-7):
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    return float((2 * inter + eps) / (pred.sum() + gt.sum() + eps))


def evaluate(model, X, Y, thr):
    probs = predict_probs(model, X)
    preds = probs > thr
    gts = Y[:, 0] > 0.5
    ious = np.array([iou_score(p, g) for p, g in zip(preds, gts)])
    dices = np.array([dice_score(p, g) for p, g in zip(preds, gts)])
    return probs, ious, dices


def sample_panels(x, gt, prob, thr):
    """Row image: input | ground truth | prediction | error map."""
    pred = prob > thr
    gt_b = gt > 0.5
    g = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    inp = np.stack([g] * 3, -1)
    gt_rgb = np.stack([gt_b.astype(np.uint8) * 255] * 3, -1)
    pr_rgb = np.stack([pred.astype(np.uint8) * 255] * 3, -1)
    err = np.zeros((*gt_b.shape, 3), np.uint8)
    err[np.logical_and(pred, gt_b)] = (60, 200, 90)    # true positive  (green)
    err[np.logical_and(pred, ~gt_b)] = (230, 60, 60)   # false positive (red)
    err[np.logical_and(~pred, gt_b)] = (60, 110, 240)  # false negative (blue)
    row = np.hstack([inp, gt_rgb, pr_rgb, err])
    return cv2.resize(row, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)


# ======================================================================
# 4) VIDEO: SYNTHETIC VIDEO, TRACKING, OPTICAL FLOW
# ======================================================================
def synthetic_video(n_frames=40, w=320, h=240, seed=0):
    """RGB frames: three moving shapes on a smooth background with static
    distractor objects."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = 110 + 40 * np.sin(xx / 55.0) + 30 * np.cos(yy / 45.0)
    bg = np.stack([base, base * 0.95 + 8, base * 0.85 + 18], -1)
    bg = np.clip(bg, 0, 255).astype(np.uint8)
    cv2.rectangle(bg, (20, 170), (90, 225), (70, 90, 120), -1)
    cv2.rectangle(bg, (230, 20), (300, 70), (130, 110, 80), -1)
    cv2.line(bg, (0, 200), (w, 200), (60, 60, 60), 2)

    frames = []
    denom = max(n_frames - 1, 1)
    for t in range(n_frames):
        f = bg.copy()
        # circle: diagonal motion
        cx = int(40 + 240 * t / denom)
        cy = int(40 + 100 * t / denom)
        cv2.circle(f, (cx, cy), 22, (220, 60, 60), -1)
        # rectangle: elliptical motion
        rx = int(w / 2 + 90 * np.sin(2 * np.pi * t / n_frames))
        ry = int(h / 2 + 40 * np.cos(2 * np.pi * t / n_frames))
        cv2.rectangle(f, (rx - 20, ry - 14), (rx + 20, ry + 14), (60, 200, 90), -1)
        # triangle: moves left
        tx = int(w - 40 - 200 * t / denom)
        ty = 190
        pts = np.array([[tx, ty - 22], [tx - 20, ty + 16], [tx + 20, ty + 16]], np.int32)
        cv2.fillPoly(f, [pts], (240, 220, 70))
        f = np.clip(f.astype(np.float32) + rng.normal(0, 2.0, f.shape), 0, 255)
        frames.append(f.astype(np.uint8))
    return frames


def read_video_frames(data: bytes, max_frames: int, width: int):
    """Decode an uploaded video with OpenCV; return up to max_frames RGB frames."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    try:
        tmp.write(data)
        tmp.close()
        cap = cv2.VideoCapture(tmp.name)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stride = max(1, total // max_frames) if total > 0 else 1
        frames, i = [], 0
        while len(frames) < max_frames:
            ok, bgr = cap.read()
            if not ok:
                break
            if i % stride == 0:
                h, w = bgr.shape[:2]
                if w != width:
                    bgr = cv2.resize(bgr, (width, int(h * width / w)),
                                     interpolation=cv2.INTER_AREA)
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            i += 1
        cap.release()
        return frames
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def frames_to_gif(frames, fps=8):
    imgs = [Image.fromarray(f) for f in frames]
    buf = io.BytesIO()
    imgs[0].save(buf, format="GIF", save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0)
    return buf.getvalue()


def run_lk_tracking(frames, trail_len=15):
    """Sparse Lucas-Kanade tracking with forward-backward check and
    automatic re-detection of lost points."""
    grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    feat = dict(maxCorners=60, qualityLevel=0.15, minDistance=10, blockSize=7)
    lk = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

    p0 = cv2.goodFeaturesToTrack(grays[0], mask=None, **feat)
    tracks = [[(float(p[0, 0]), float(p[0, 1]))] for p in p0] if p0 is not None else []

    def draw(frame, tracks, idx):
        out = frame.copy()
        for t in tracks:
            pts = np.array(t[-trail_len:], np.int32).reshape(-1, 1, 2)
            k = min(5, len(t) - 1)
            speed = 0.0
            if k > 0:
                speed = float(np.hypot(t[-1][0] - t[-1 - k][0],
                                       t[-1][1] - t[-1 - k][1]) / k)
            color = (255, 60, 60) if speed > 1.0 else (80, 200, 255)
            if len(pts) > 1:
                cv2.polylines(out, [pts], False, color, 1, cv2.LINE_AA)
            cv2.circle(out, (int(t[-1][0]), int(t[-1][1])), 3, color, -1)
        cv2.putText(out, f"frame {idx}  tracks {len(tracks)}", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    out_frames = [draw(frames[0], tracks, 0)]
    counts, speeds = [len(tracks)], [0.0]

    for i in range(1, len(frames)):
        step = []
        if tracks:
            p_prev = np.float32([t[-1] for t in tracks]).reshape(-1, 1, 2)
            p_next, st_f, _ = cv2.calcOpticalFlowPyrLK(grays[i - 1], grays[i], p_prev, None, **lk)
            p_back, st_b, _ = cv2.calcOpticalFlowPyrLK(grays[i], grays[i - 1], p_next, None, **lk)
            d = np.abs(p_prev - p_back).reshape(-1, 2).max(-1)
            good = (st_f.ravel() == 1) & (st_b.ravel() == 1) & (d < 1.0)
            h, w = grays[i].shape
            kept = []
            for t, ok, pn in zip(tracks, good, p_next.reshape(-1, 2)):
                if ok and 0 <= pn[0] < w and 0 <= pn[1] < h:
                    step.append(float(np.hypot(pn[0] - t[-1][0], pn[1] - t[-1][1])))
                    t.append((float(pn[0]), float(pn[1])))
                    kept.append(t)
            tracks = kept

        if len(tracks) < 25:  # re-detect new corners away from existing tracks
            mask = np.full(grays[i].shape, 255, np.uint8)
            for t in tracks:
                cv2.circle(mask, (int(t[-1][0]), int(t[-1][1])), 10, 0, -1)
            newp = cv2.goodFeaturesToTrack(grays[i], mask=mask, **feat)
            if newp is not None:
                for p in newp[: max(0, 60 - len(tracks))]:
                    tracks.append([(float(p[0, 0]), float(p[0, 1]))])

        out_frames.append(draw(frames[i], tracks, i))
        counts.append(len(tracks))
        speeds.append(float(np.mean(step)) if step else 0.0)

    return out_frames, counts, speeds


def flow_to_rgb(flow, max_mag):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*mag.shape, 3), np.uint8)
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / max_mag * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def run_farneback(frames):
    """Dense optical flow between consecutive frames."""
    grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    flows = [
        cv2.calcOpticalFlowFarneback(grays[i - 1], grays[i], None,
                                     0.5, 3, 15, 3, 5, 1.2, 0)
        for i in range(1, len(frames))
    ]
    mags = [np.linalg.norm(f, axis=2) for f in flows]
    max_mag = max(float(np.percentile(np.stack(mags), 99.5)), 1e-3)
    out_frames = []
    for i, f in enumerate(flows, start=1):
        vis = flow_to_rgb(f, max_mag)
        out_frames.append(np.hstack([frames[i], vis]))
    mean_mag = [float(m.mean()) for m in mags]
    moving = [float((m > 1.0).mean() * 100) for m in mags]
    return out_frames, mean_mag, moving


# ======================================================================
# 5) MISC HELPERS
# ======================================================================
def read_rgb(uploaded, size=None):
    arr = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if size is not None:
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    return rgb


# ======================================================================
# SESSION STATE
# ======================================================================
if "models" not in st.session_state:
    st.session_state["models"] = {}       # name -> (model, losses, meta)
if "last_trained" not in st.session_state:
    st.session_state["last_trained"] = None
if "video_result" not in st.session_state:
    st.session_state["video_result"] = None

# ======================================================================
# SIDEBAR
# ======================================================================
st.sidebar.title("⚙️ Controls")
st.sidebar.subheader("U-Net training")
train_mode = st.sidebar.radio(
    "Training data",
    ["Standard (clean images)", "Augmented (noise + contrast)"],
    help="The augmented model sees random noise/contrast during training, "
         "which makes it more robust to domain shift (see tab 5).",
)
steps = st.sidebar.slider("Training steps", 20, 800, 300, step=10)
batch_size = st.sidebar.select_slider("Batch size", [8, 16, 32], value=16)
base_ch = st.sidebar.select_slider("Base channels", [8, 16], value=8)
seed = st.sidebar.number_input("Random seed", 0, 9999, 42)
train_clicked = st.sidebar.button("🚀 Train U-Net", type="primary", key="train_btn")

st.sidebar.markdown("---")
models = st.session_state["models"]
active_name = None
if models:
    names = list(models.keys())
    default_idx = names.index(st.session_state["last_trained"]) \
        if st.session_state["last_trained"] in names else 0
    active_name = st.sidebar.selectbox("Active model", names, index=default_idx)
else:
    st.sidebar.info("No trained model yet. Press **Train U-Net**.")

thr = st.sidebar.slider("Prediction threshold", 0.1, 0.9, 0.5, step=0.05)

# ======================================================================
# MAIN PAGE
# ======================================================================
st.title("🧠 Advanced Computer Vision Project")
st.caption("Segmentation (U-Net) · IoU / Dice · Motion analysis · Vision Transformers · "
           "Application & challenge")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1️⃣ Segmentation (U-Net)",
    "2️⃣ IoU / Dice Evaluation",
    "3️⃣ Video Motion Analysis",
    "4️⃣ Advanced Topic: ViT",
    "5️⃣ Application & Challenge",
])

# ------------------------------ TAB 1 ---------------------------------
with tab1:
    st.header("Semantic segmentation with U-Net")
    st.markdown(
        """
**Semantic segmentation** assigns a class to *every pixel*. Here the task is
binary: **object (shape) vs background**.

**U-Net** (Ronneberger et al., 2015) is an encoder-decoder network:
- the **encoder** (convolution + max-pooling) captures context while shrinking
  the image,
- the **decoder** (transposed convolution) restores the resolution,
- **skip connections** copy encoder features to the decoder so fine boundaries
  are not lost.

The model is trained **inside this app** on automatically generated images
(circles, rectangles, triangles with random brightness, gradient and noise), so
every pixel has an exact ground-truth mask and no dataset download is needed.
"""
    )

    preview = UNet(base=base_ch)
    c1, c2, c3 = st.columns(3)
    c1.metric("Input size", f"{IMG}×{IMG}×1")
    c2.metric("Parameters", f"{count_params(preview):,}")
    c3.metric("Loss", "BCE + Dice")

    if train_clicked:
        name = "Augmented U-Net" if train_mode.startswith("Augmented") else "Standard U-Net"
        bar = st.progress(0.0, text="Training...")
        t0 = time.time()

        def cb(frac, loss):
            bar.progress(min(frac, 1.0), text=f"Training... {frac * 100:.0f}%   loss = {loss:.3f}")

        model, losses = train_model(
            steps, batch_size, base_ch, int(seed),
            augment=train_mode.startswith("Augmented"), progress_cb=cb,
        )
        bar.empty()
        st.session_state["models"][name] = (
            model, losses,
            {"steps": steps, "batch": batch_size, "base": base_ch,
             "seconds": time.time() - t0},
        )
        st.session_state["last_trained"] = name
        st.rerun()

    if active_name is None:
        st.info("👈 Choose the training settings in the sidebar and press **Train U-Net**. "
                "With the default settings training takes only a few seconds.")
    else:
        model, losses, meta = models[active_name]
        st.success(f"Active model: **{active_name}**  |  {meta['steps']} steps, "
                   f"batch {meta['batch']}, base channels {meta['base']}  |  "
                   f"trained in {meta['seconds']:.1f} s")

        st.subheader("Training loss")
        st.line_chart(pd.DataFrame({"loss": losses}))

        st.subheader("Predictions on unseen test images")
        s1, s2 = st.columns(2)
        n_show = s1.slider("Samples to show", 1, 6, 4)
        sseed = s2.number_input("Sample seed", 0, 9999, 7)
        rng = np.random.default_rng(int(sseed) + 100000)
        Xs, Ys = make_batch(rng, n_show)
        probs, ious, dices = evaluate(model, Xs, Ys, thr)

        h1, h2, h3, h4 = st.columns(4)
        h1.markdown("**Input**")
        h2.markdown("**Ground truth**")
        h3.markdown("**Prediction**")
        h4.markdown("**Error map**")
        for i in range(n_show):
            st.image(sample_panels(Xs[i, 0], Ys[i, 0], probs[i], thr), **IMG_KW)
            st.caption(f"Sample {i + 1}:  IoU = {ious[i]:.3f}   |   Dice = {dices[i]:.3f}")
        st.caption("Error map colours: 🟩 true positive · 🟥 false positive · 🟦 false negative")

    with st.expander("Code used (model definition)"):
        st.code(
            '''class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=8):
        ...  # enc1..enc3 -> bottleneck -> up3/dec3 -> up2/dec2 -> up1/dec1 -> 1x1 conv

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        b  = self.bott(F.max_pool2d(e3, 2))
        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))   # skip connection
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))  # skip connection
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))  # skip connection
        return self.out(d1)
''',
            language="python",
        )

# ------------------------------ TAB 2 ---------------------------------
with tab2:
    st.header("Evaluation with IoU and Dice")
    st.markdown("Both metrics compare the predicted mask **P** with the ground-truth mask **G**.")
    f1, f2 = st.columns(2)
    f1.latex(r"\mathrm{IoU}=\frac{|P\cap G|}{|P\cup G|}")
    f2.latex(r"\mathrm{Dice}=\frac{2\,|P\cap G|}{|P|+|G|}")
    st.markdown(
        "Both range from **0** (no overlap) to **1** (perfect). They are linked by "
        "`Dice = 2·IoU / (1 + IoU)`, so Dice is always ≥ IoU and is more forgiving."
    )

    if active_name is None:
        st.info("Train a model first (sidebar).")
    else:
        model, losses, meta = models[active_name]
        e1c, e2c, e3c = st.columns(3)
        n_test = e1c.slider("Test images", 20, 200, 100, step=10)
        t_noise = e2c.slider("Test noise level", 0.0, 0.4, 0.05, step=0.01)
        t_contrast = e3c.slider("Test contrast", 0.2, 1.0, 1.0, step=0.05)

        Xt, Yt = get_test_set(n_test, t_noise, t_contrast)
        probs, ious, dices = evaluate(model, Xt, Yt, thr)
        pix_acc = float(((probs > thr) == (Yt[:, 0] > 0.5)).mean())

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mean IoU", f"{ious.mean():.3f}")
        m2.metric("Mean Dice", f"{dices.mean():.3f}")
        m3.metric("Pixel accuracy", f"{pix_acc:.3f}")
        m4.metric("Worst-case IoU", f"{ious.min():.3f}")

        st.markdown("#### Summary table")
        st.dataframe(
            pd.DataFrame({
                "Metric": ["IoU", "Dice"],
                "Mean": [ious.mean(), dices.mean()],
                "Std": [ious.std(), dices.std()],
                "Min": [ious.min(), dices.min()],
                "Median": [np.median(ious), np.median(dices)],
                "Max": [ious.max(), dices.max()],
            }).round(4),
            hide_index=True,
        )

        st.markdown("#### Distribution of per-image IoU")
        counts, edges = np.histogram(ious, bins=10, range=(0, 1))
        hist_df = pd.DataFrame(
            {"images": counts},
            index=[f"{edges[i]:.1f}-{edges[i + 1]:.1f}" for i in range(10)],
        )
        st.bar_chart(hist_df)

        st.markdown("#### Best and worst test image")
        bi, wi = int(np.argmax(ious)), int(np.argmin(ious))
        st.image(sample_panels(Xt[bi, 0], Yt[bi, 0], probs[bi], thr), **IMG_KW)
        st.caption(f"Best: IoU = {ious[bi]:.3f}, Dice = {dices[bi]:.3f}")
        st.image(sample_panels(Xt[wi, 0], Yt[wi, 0], probs[wi], thr), **IMG_KW)
        st.caption(f"Worst: IoU = {ious[wi]:.3f}, Dice = {dices[wi]:.3f}")

        with st.expander("Code used"):
            st.code(
                '''def iou_score(pred, gt, eps=1e-7):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return (inter + eps) / (union + eps)

def dice_score(pred, gt, eps=1e-7):
    inter = np.logical_and(pred, gt).sum()
    return (2 * inter + eps) / (pred.sum() + gt.sum() + eps)
''',
                language="python",
            )

# ------------------------------ TAB 3 ---------------------------------
with tab3:
    st.header("Short video: tracking and optical-flow motion analysis")
    st.markdown(
        """
- **Lucas-Kanade (sparse) tracking:** detects corner points (Shi-Tomasi) and follows
  them from frame to frame. Trails are drawn behind each point:
  🟥 red = moving point, 🟦 cyan = (almost) static point. A forward-backward
  check removes unreliable tracks.
- **Farneback (dense) optical flow:** estimates a motion vector for **every pixel**.
  In the visualisation, **colour = direction** and **brightness = speed**.
"""
    )

    source = st.radio("Video source", ["Built-in synthetic video", "Upload a short video"],
                      horizontal=True)
    v1, v2, v3 = st.columns(3)
    max_frames = v1.slider("Max frames", 10, 60, 40, step=5)
    vid_width = v2.select_slider("Frame width (px)", [240, 320, 400], value=320)
    fps = v3.slider("GIF speed (fps)", 4, 15, 8)

    up_video = None
    if source == "Upload a short video":
        up_video = st.file_uploader("Upload mp4 / avi / mov / mkv",
                                    type=["mp4", "avi", "mov", "mkv"], key="vid_up")

    if st.button("▶️ Process video", type="primary", key="video_btn"):
        frames = None
        if source == "Built-in synthetic video":
            frames = synthetic_video(max_frames, w=vid_width, h=int(vid_width * 0.75))
        elif up_video is not None:
            with st.spinner("Reading video..."):
                frames = read_video_frames(up_video.getvalue(), max_frames, vid_width)
            if not frames or len(frames) < 3:
                st.error("Could not read enough frames from this video. Try another file.")
                frames = None
        else:
            st.warning("Please upload a video first, or choose the built-in synthetic video.")

        if frames:
            with st.spinner("Tracking points (Lucas-Kanade)..."):
                lk_frames, lk_counts, lk_speed = run_lk_tracking(frames)
            with st.spinner("Computing dense optical flow (Farneback)..."):
                fb_frames, fb_mean, fb_moving = run_farneback(frames)
            st.session_state["video_result"] = dict(
                frames=frames, lk_frames=lk_frames, fb_frames=fb_frames,
                lk_counts=lk_counts, lk_speed=lk_speed,
                fb_mean=fb_mean, fb_moving=fb_moving,
                lk_gif=frames_to_gif(lk_frames, fps),
                fb_gif=frames_to_gif(fb_frames, fps),
            )

    res = st.session_state["video_result"]
    if res is not None:
        n = len(res["frames"])
        st.success(f"Processed {n} frames.")

        g1, g2 = st.columns(2)
        with g1:
            st.subheader("Lucas-Kanade tracking")
            st.image(res["lk_gif"], **IMG_KW)
        with g2:
            st.subheader("Dense optical flow (frame | flow)")
            st.image(res["fb_gif"], **IMG_KW)

        st.markdown("#### Browse single frames")
        idx = st.slider("Frame", 0, n - 1, min(n // 2, n - 1), key="frame_idx")
        b1, b2 = st.columns(2)
        b1.image(res["lk_frames"][idx], caption=f"LK tracking - frame {idx}", **IMG_KW)
        b2.image(res["fb_frames"][max(idx - 1, 0)],
                 caption=f"Dense flow - frame {max(idx, 1)} (flow from previous frame)", **IMG_KW)

        st.markdown("#### Motion statistics")
        s1, s2 = st.columns(2)
        s1.write("Mean flow magnitude per frame (pixels / frame)")
        s1.line_chart(pd.DataFrame({"mean flow (px)": res["fb_mean"]}))
        s2.write("Percentage of pixels moving faster than 1 px / frame")
        s2.line_chart(pd.DataFrame({"moving pixels (%)": res["fb_moving"]}))
        s3, s4 = st.columns(2)
        s3.write("Number of tracked points per frame (Lucas-Kanade)")
        s3.line_chart(pd.DataFrame({"tracked points": res["lk_counts"]}))
        s4.write("Mean speed of tracked points (pixels / frame)")
        s4.line_chart(pd.DataFrame({"speed (px)": res["lk_speed"]}))

        st.dataframe(
            pd.DataFrame({
                "Quantity": ["Frames processed", "Mean tracked points",
                             "Mean point speed (px/frame)",
                             "Mean dense-flow magnitude (px/frame)",
                             "Mean % moving pixels"],
                "Value": [n, round(float(np.mean(res["lk_counts"])), 1),
                          round(float(np.mean(res["lk_speed"])), 3),
                          round(float(np.mean(res["fb_mean"])), 3),
                          round(float(np.mean(res["fb_moving"])), 2)],
            }),
            hide_index=True,
        )
    else:
        st.info("Press **Process video** to run tracking and optical flow.")

    with st.expander("Code used"):
        st.code(
            '''# Sparse tracking (Lucas-Kanade)
p0 = cv2.goodFeaturesToTrack(gray0, maxCorners=60, qualityLevel=0.15, minDistance=10)
p1, st, err = cv2.calcOpticalFlowPyrLK(gray0, gray1, p0, None, winSize=(21, 21), maxLevel=3)

# Dense optical flow (Farneback)
flow = cv2.calcOpticalFlowFarneback(gray0, gray1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])   # speed and direction
''',
            language="python",
        )

# ------------------------------ TAB 4 ---------------------------------
with tab4:
    st.header("Advanced topic review: Vision Transformers (ViT)")
    st.markdown(
        """
**Idea.** The Vision Transformer (Dosovitskiy et al., 2020, *"An Image is Worth
16x16 Words"*) applies the standard Transformer encoder, originally designed for
text, directly to images. Instead of convolutions, it uses **self-attention**,
so every part of the image can interact with every other part from the very
first layer.

**Pipeline**
1. **Split** the image into non-overlapping patches (e.g. 16×16 pixels).
2. **Flatten + linearly project** each patch into an embedding vector (a "token").
3. **Add position embeddings** (the model otherwise does not know where a patch came from)
   and a learnable **[CLS] token**.
4. Pass the tokens through **L Transformer encoder blocks**
   (LayerNorm → Multi-Head Self-Attention → residual → LayerNorm → MLP → residual).
5. Use the **[CLS] output** for classification (or the patch tokens for dense tasks).
"""
    )
    a1, a2 = st.columns(2)
    a1.latex(r"N=\frac{H\cdot W}{P^{2}}")
    a2.latex(r"\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V")

    st.subheader("🔬 Interactive: how an image becomes tokens")
    up_vit = st.file_uploader("Optional: upload your own image", type=["jpg", "jpeg", "png", "bmp", "webp"],
                              key="vit_up")
    SIDE = 224
    vit_img = read_rgb(up_vit, SIDE) if up_vit is not None else None
    if vit_img is None:
        vit_img = cv2.resize(synthetic_video(5)[2], (SIDE, SIDE), interpolation=cv2.INTER_AREA)

    patch = st.select_slider("Patch size P", [8, 16, 32], value=16)
    n_side = SIDE // patch
    n_tok = n_side * n_side
    grid = vit_img.copy()
    for i in range(1, n_side):
        cv2.line(grid, (i * patch, 0), (i * patch, SIDE - 1), (255, 255, 255), 1)
        cv2.line(grid, (0, i * patch), (SIDE - 1, i * patch), (255, 255, 255), 1)

    i1, i2 = st.columns(2)
    i1.image(vit_img, caption="Input image (224×224)", **IMG_KW)
    i2.image(grid, caption=f"Split into {n_side}×{n_side} patches", **IMG_KW)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Tokens N", n_tok)
    k2.metric("Tokens + [CLS]", n_tok + 1)
    k3.metric("Values per patch", patch * patch * 3)
    k4.metric("Attention matrix entries", f"{(n_tok + 1) ** 2:,}")

    p_idx = st.slider("Inspect a patch (token index)", 0, n_tok - 1, 0)
    r, c = divmod(p_idx, n_side)
    patch_img = vit_img[r * patch:(r + 1) * patch, c * patch:(c + 1) * patch]
    pz1, pz2 = st.columns([1, 3])
    pz1.image(cv2.resize(patch_img, (128, 128), interpolation=cv2.INTER_NEAREST),
              caption=f"Patch #{p_idx} (row {r}, col {c})")
    pz2.markdown(
        f"This {patch}×{patch}×3 patch is flattened into a vector of "
        f"**{patch * patch * 3}** numbers and linearly projected to the embedding size "
        f"(768 in ViT-Base). The first values are: "
        f"`{patch_img.reshape(-1)[:12].tolist()} ...`"
    )

    st.markdown("**Cost of self-attention grows quadratically with the number of tokens**")
    res_list = [224, 384, 512, 768, 1024]
    cost = pd.DataFrame(
        {"attention entries (millions), P=16": [((r_ // 16) ** 2 + 1) ** 2 / 1e6 for r_ in res_list]},
        index=[f"{r_}px" for r_ in res_list],
    )
    st.line_chart(cost)

    st.subheader("ViT vs CNN")
    st.table(pd.DataFrame({
        "Aspect": ["Core operation", "Receptive field", "Inductive bias",
                   "Data needs", "Cost with resolution", "Strengths"],
        "CNN": ["Local convolution", "Grows slowly with depth",
                "Strong (locality, translation equivariance)",
                "Works well with small/medium data",
                "Linear in pixels",
                "Efficient, robust on small datasets"],
        "Vision Transformer": ["Global self-attention", "Global from layer 1",
                               "Weak (learns it from data)",
                               "Needs large data or pre-training",
                               "Quadratic in tokens",
                               "Scales very well, strong transfer learning"],
    }))

    st.markdown(
        """
**Important variants.**
- **DeiT** - data-efficient training with distillation, so ViT works on ImageNet-1k alone.
- **Swin Transformer** - attention inside shifted local windows and a hierarchical
  design, which reduces cost and suits detection and segmentation.
- **MAE / DINO** - *self-supervised* pre-training (masked reconstruction, self-distillation)
  that removes the need for labels.
- **SegFormer, SAM** - Transformer backbones for segmentation (SAM = Segment Anything).

**Limitations.** Quadratic attention cost at high resolution, data hunger without
pre-training, and weaker built-in notion of locality than CNNs.
"""
    )

# ------------------------------ TAB 5 ---------------------------------
with tab5:
    st.header("One application and one technical challenge")

    st.subheader("📌 Application: medical image segmentation")
    st.markdown(
        """
U-Net was originally created for **biomedical image segmentation**. Today
segmentation networks outline **tumours, organs and lesions** in MRI, CT and
ultrasound scans.

**Why it matters**
- **Radiotherapy planning:** the radiation dose must cover the tumour and spare
  healthy organs, which requires precise outlines.
- **Measuring change:** the tumour volume can be tracked automatically over several scans.
- **Workload and consistency:** manual outlining takes a radiologist from minutes up to
  hours per patient; a model gives fast and reproducible first drafts.

**Typical pipeline:** scan → normalisation → U-Net → mask → post-processing →
review by a clinician, with **IoU / Dice** as the standard quality metrics (exactly
the ones used in tab 2).
"""
    )

    st.subheader("⚠️ Technical challenge: domain shift and scarce labels")
    st.markdown(
        """
A model is trained on data from one scanner / hospital / protocol, but at deployment the
images look different (more noise, different contrast, different device). This is
**domain shift**, and accuracy can fall sharply. It is made worse because **expert
labels are expensive** and patient privacy limits data sharing.

**Common mitigations**
- **Data augmentation** (noise, contrast, intensity changes) - tested below.
- **Domain adaptation / intensity harmonisation** between scanners.
- **Self-supervised pre-training** (MAE, DINO) on large unlabeled collections.
- **Transfer learning** and **federated learning** (train across hospitals without sharing images).
- **Uncertainty estimation** and a **human-in-the-loop** review for risky cases.
"""
    )

    st.subheader("🧪 Experiment: how robust is the trained U-Net?")
    if not models:
        st.info("Train at least one model (sidebar). Train both the **Standard** and the "
                "**Augmented** model to compare them here.")
    else:
        st.caption("Each point is the mean IoU on 60 unseen synthetic test images. "
                   "The model never sees the shifted test images during training.")
        noise_levels = [0.05, 0.1, 0.2, 0.3, 0.4]
        contrast_levels = [1.0, 0.8, 0.6, 0.4, 0.25]

        noise_tbl, contrast_tbl = {}, {}
        for nm, (mdl, _, _) in models.items():
            row_n, row_c = [], []
            for lv in noise_levels:
                Xn, Yn = get_test_set(60, lv, 1.0)
                row_n.append(float(evaluate(mdl, Xn, Yn, thr)[1].mean()))
            for lv in contrast_levels:
                Xc, Yc = get_test_set(60, 0.05, lv)
                row_c.append(float(evaluate(mdl, Xc, Yc, thr)[1].mean()))
            noise_tbl[nm] = row_n
            contrast_tbl[nm] = row_c

        r1, r2 = st.columns(2)
        r1.write("Mean IoU vs **noise level**")
        r1.line_chart(pd.DataFrame(noise_tbl, index=[str(v) for v in noise_levels]))
        r2.write("Mean IoU vs **lower contrast** (1.0 = normal)")
        r2.line_chart(pd.DataFrame(contrast_tbl, index=[str(v) for v in contrast_levels]))

        if len(models) < 2:
            st.info("Tip: train the other variant too (change *Training data* in the sidebar) "
                    "to see whether augmentation reduces the drop.")

        st.markdown("**Visual example under strong shift** (noise 0.35, contrast 0.4)")
        Xh, Yh = get_test_set(4, 0.35, 0.4, seed=77)
        for nm, (mdl, _, _) in models.items():
            ph, ih, dh = evaluate(mdl, Xh, Yh, thr)
            st.write(f"**{nm}** - mean IoU on these 4 images: {ih.mean():.3f}")
            st.image(np.vstack([sample_panels(Xh[i, 0], Yh[i, 0], ph[i], thr) for i in range(2)]),
                     **IMG_KW)

st.markdown("---")
st.caption("Built with Streamlit, PyTorch and OpenCV")
