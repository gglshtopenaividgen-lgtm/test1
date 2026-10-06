#!/usr/bin/env python3
"""
================================================================================
KEYFRAME CINEMA ENGINE (COLAB FREE T4) - max-quality 10 minute film
File: bulk_generator/templates/keyframe_cinema_engine.py

Pipeline (each stage resumable, state kept in a work dir + public GitHub repo):
  1. keyframes : SDXL stills 1280x720 from a hand-written screenplay.json,
                 automatic QA gate (rejects black / flat / smeared frames).
  2. svd       : Stable Video Diffusion XT image-to-video for "hero" shots.
  3. render    : Depth-Anything-V2 depth + GPU 2.5D parallax camera moves for
                 the other shots, hero clips upscaled/interpolated to 24 fps.
  4. assemble  : single streaming pass: crossfades, grade, grain, fades,
                 generated ambient audio, x264 encode, chunked GitHub push.

No LLM runs on the GPU: the screenplay is authored offline.
Secrets are read from env / Colab userdata only. Nothing is hard-coded.
================================================================================
"""
from __future__ import annotations

import argparse
import base64
import gc
import json
import math
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import wave
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Set

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")

import numpy as np
from PIL import Image

MOVES = {"dolly_in", "dolly_out", "pan_left", "pan_right", "orbit", "crane_up", "crane_down"}
SVD_FRAMES = 25
SVD_FPS = 7
SVD_SLOWDOWN = 1.2  # hero clips are slowed 20% for a calmer, more cinematic feel
HERO_SECONDS = SVD_FRAMES / SVD_FPS * SVD_SLOWDOWN


def log(msg: str) -> None:
    print(msg, flush=True)


# ==============================================================================
# CONFIG
# ==============================================================================

def make_cfg(**overrides: Any) -> SimpleNamespace:
    cfg = dict(
        screenplay="screenplay_cosmic.json",
        workdir="/content/cinema_work",
        tier="final",                 # final = SDXL base 30 steps | draft = SDXL-Lightning 4 step
        stages="keyframes,svd,render,assemble",
        target_seconds=600.0,
        width=1920, height=1080, fps=24,
        kf_w=1280, kf_h=720,
        steps_final=30, cfg_final=6.5,
        crossfade=0.4,
        qa_retries=3,
        use_svd=True,
        svd_steps=25, svd_motion_bucket=100, svd_noise_aug=0.02,
        audio=True,
        crf=22, preset="medium",
        shots=None,                   # optional list of shot indices (smoke tests)
        smoke=False,
        repo="gglshtopenaividgen-lgtm/test1", branch="main", remote_dir="cinema",
        push=True,
        gh_token="", 
    )
    unknown = set(overrides) - set(cfg)
    if unknown:
        raise TypeError(f"unknown cfg keys: {sorted(unknown)}")
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def resolve_secret(names: List[str]) -> str:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    try:
        from google.colab import userdata  # type: ignore
        for n in names:
            try:
                v = userdata.get(n)
                if v:
                    return v
            except Exception:
                continue
    except Exception:
        pass
    return ""


def ffmpeg_bin() -> str:
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg  # type: ignore
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        raise RuntimeError("ffmpeg not found (apt install ffmpeg or pip install imageio-ffmpeg)") from e


# ==============================================================================
# SCREENPLAY
# ==============================================================================

def load_screenplay(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        sp = json.load(f)
    for k in ("style_bible", "negative", "acts", "shots"):
        if k not in sp:
            raise ValueError(f"screenplay missing key '{k}'")
    n_acts = len(sp["acts"])
    for i, s in enumerate(sp["shots"]):
        for k in ("act", "hero", "move", "subject"):
            if k not in s:
                raise ValueError(f"shot {i} missing '{k}'")
        if not (0 <= s["act"] < n_acts):
            raise ValueError(f"shot {i} has invalid act {s['act']}")
        if s["move"] not in MOVES:
            raise ValueError(f"shot {i} has invalid move '{s['move']}'")
        if len(s["subject"].split()) < 5:
            raise ValueError(f"shot {i} subject too short")
    return sp


def build_prompt(sp: Dict[str, Any], idx: int) -> str:
    shot = sp["shots"][idx]
    act = sp["acts"][shot["act"]]
    return f"{shot['subject']}, {act['palette']}, {sp['style_bible']}"


def shot_seed(sp: Dict[str, Any], idx: int, attempt: int) -> int:
    base = sp["acts"][sp["shots"][idx]["act"]]["seed_base"]
    return int(base + idx * 7919 + attempt * 104729) % (2**31 - 1)


# ==============================================================================
# QA GATE (rejects black / flat / smeared keyframes)
# ==============================================================================

def qa_metrics(img: Image.Image) -> Dict[str, float]:
    g = np.asarray(img.convert("L").resize((640, 360), Image.BILINEAR), dtype=np.float32)
    lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    hist = np.bincount(g.astype(np.uint8).ravel(), minlength=256).astype(np.float64) / g.size
    nz = hist[hist > 0]
    return {
        "sharp": float(lap.var()),
        "entropy": float(-(nz * np.log2(nz)).sum()),
        "mean": float(g.mean()),
        "std": float(g.std()),
    }


def qa_pass(m: Dict[str, float]) -> bool:
    # Deep-space frames are legitimately dark, so only catch black/white/flat failures.
    return m["std"] >= 4.0 and 1.0 <= m["mean"] <= 250.0 and m["sharp"] >= 2.0


def qa_score(m: Dict[str, float]) -> float:
    return min(m["sharp"], 400.0) / 40.0 + m["entropy"] + min(m["std"], 60.0) / 20.0


# ==============================================================================
# GITHUB STORE (REST API, public repo for reads)
# ==============================================================================

class GitHubStore:
    def __init__(self, token: str, repo: str, branch: str, remote_dir: str, enabled: bool = True):
        repo = repo.strip().replace("https://github.com/", "").rstrip("/")
        if repo.endswith(".git"):
            repo = repo[:-4]
        self.repo, self.branch = repo, branch
        self.remote_dir = remote_dir.strip("/")
        self.token = (token or "").strip()
        self.enabled = bool(enabled)
        self.headers = {"Accept": "application/vnd.github+json", "User-Agent": "cinema-engine/1.0"}
        if self.token:
            prefix = "Bearer" if self.token.startswith("github_pat_") else "token"
            self.headers["Authorization"] = f"{prefix} {self.token}"

    def _path(self, subdir: str, name: str = "") -> str:
        parts = [p for p in (self.remote_dir, subdir.strip("/"), name) if p]
        return "/".join(parts)

    def list_names(self, subdir: str) -> Set[str]:
        if not self.enabled:
            return set()
        url = f"https://api.github.com/repos/{self.repo}/contents/{self._path(subdir)}?ref={self.branch}"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=self.headers), timeout=30) as r:
                return {it["name"] for it in json.loads(r.read().decode())}
        except urllib.error.HTTPError as e:
            if e.code != 404:
                log(f"   [github] list {subdir} failed: HTTP {e.code}")
            return set()
        except Exception as e:
            log(f"   [github] list {subdir} failed: {e}")
            return set()

    def download(self, subdir: str, name: str, local: Path) -> bool:
        if not self.enabled:
            return False
        url = f"https://raw.githubusercontent.com/{self.repo}/{self.branch}/{self._path(subdir, name)}"
        try:
            local.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "cinema-engine/1.0"}), timeout=60) as r:
                data = r.read()
            if not data:
                return False
            local.write_bytes(data)
            return True
        except Exception as e:
            log(f"   [github] download {name} failed: {e}")
            return False

    def upload(self, local: Path, subdir: str, name: str, message: str) -> bool:
        if not (self.enabled and self.token):
            return False
        url = f"https://api.github.com/repos/{self.repo}/contents/{self._path(subdir, name)}"
        content = base64.b64encode(local.read_bytes()).decode()
        for attempt in range(4):
            try:
                sha = None
                try:
                    with urllib.request.urlopen(urllib.request.Request(f"{url}?ref={self.branch}", headers=self.headers), timeout=30) as r:
                        sha = json.loads(r.read().decode()).get("sha")
                except urllib.error.HTTPError as e:
                    if e.code != 404:
                        raise
                body: Dict[str, Any] = {"message": message, "content": content, "branch": self.branch}
                if sha:
                    body["sha"] = sha
                req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=self.headers, method="PUT")
                with urllib.request.urlopen(req, timeout=180):
                    return True
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:140]
                log(f"   [github] upload {name} HTTP {e.code} (try {attempt + 1}/4): {detail}")
                if e.code in (401, 403, 404) and attempt >= 1:
                    return False
            except Exception as e:
                log(f"   [github] upload {name} error (try {attempt + 1}/4): {e}")
            time.sleep(2 + attempt * 3)
        return False


# ==============================================================================
# FFMPEG HELPERS
# ==============================================================================

def read_exact(stream, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            break
        buf.extend(chunk)
    return bytes(buf)


def start_encoder(out: Path, w: int, h: int, fps: float, crf: int, preset: str,
                  vf: Optional[str] = None, audio: Optional[Path] = None) -> subprocess.Popen:
    cmd = [ffmpeg_bin(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-"]
    if audio is not None:
        cmd += ["-i", str(audio)]
    if vf:
        cmd += ["-vf", vf]
    cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p"]
    if audio is not None:
        cmd += ["-c:a", "aac", "-b:a", "160k", "-shortest"]
    cmd += ["-movflags", "+faststart", str(out)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def finish_encoder(proc: subprocess.Popen) -> None:
    try:
        proc.stdin.close()
    except Exception:
        pass
    err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg encoder failed (rc={rc}): {err[-400:]}")


def count_frames(path: Path) -> int:
    r = subprocess.run([ffmpeg_bin(), "-v", "error", "-nostats", "-progress", "pipe:1",
                        "-i", str(path), "-map", "0:v:0", "-f", "null", "-"],
                       capture_output=True, text=True)
    found = re.findall(r"frame=\s*(\d+)", r.stdout)
    if not found:
        raise RuntimeError(f"could not count frames of {path}: {r.stderr[-200:]}")
    return int(found[-1])


def decode_frames(path: Path, w: int, h: int) -> Iterable[np.ndarray]:
    proc = subprocess.Popen([ffmpeg_bin(), "-v", "error", "-i", str(path), "-f", "rawvideo",
                             "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-"],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    nbytes = w * h * 3
    try:
        while True:
            raw = read_exact(proc.stdout, nbytes)
            if len(raw) < nbytes:
                break
            yield np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        proc.wait()


# ==============================================================================
# STAGE 1: KEYFRAMES (SDXL)
# ==============================================================================

def free_gpu() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def load_sdxl(cfg):
    import torch
    from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, EulerDiscreteScheduler, StableDiffusionXLPipeline
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    log(f"[keyframes] loading SDXL ({cfg.tier}) ...")
    # The stock SDXL VAE yields NaN / black images in fp16, this fixed VAE does not.
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    if cfg.tier == "draft":
        from diffusers import UNet2DConditionModel
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        unet = UNet2DConditionModel.from_config(base, subfolder="unet").to("cuda", torch.float16)
        unet.load_state_dict(load_file(hf_hub_download("ByteDance/SDXL-Lightning", "sdxl_lightning_4step_unet.safetensors"), device="cuda"))
        pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, vae=vae, torch_dtype=torch.float16, variant="fp16")
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        steps, guidance = 4, 0.0
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(base, vae=vae, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
        steps, guidance = cfg.steps_final, cfg.cfg_final
    pipe.to("cuda")
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    return pipe, steps, guidance


def make_contact_sheet(kf_dir: Path, indices: List[int], out: Path, cols: int = 8, tw: int = 240, th: int = 135) -> Optional[Path]:
    thumbs = []
    for i in indices:
        p = kf_dir / f"shot_{i:03d}.jpg"
        if p.exists():
            thumbs.append(Image.open(p).convert("RGB").resize((tw, th), Image.LANCZOS))
    if not thumbs:
        return None
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * tw, rows * th), (0, 0, 0))
    for n, t in enumerate(thumbs):
        sheet.paste(t, ((n % cols) * tw, (n // cols) * th))
    sheet.save(out, quality=88)
    return out


def stage_keyframes(cfg, sp, indices: List[int], store: GitHubStore, wd: Path) -> None:
    kf_dir = wd / "keyframes"
    kf_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = wd / "keyframes_manifest.json"
    manifest: Dict[str, Any] = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    remote = store.list_names("keyframes")
    todo = []
    for i in indices:
        p = kf_dir / f"shot_{i:03d}.jpg"
        if p.exists():
            continue
        if p.name in remote and store.download("keyframes", p.name, p):
            log(f"[keyframes] shot {i:03d} restored from GitHub")
            continue
        todo.append(i)
    if todo:
        import torch
        pipe, steps, guidance = load_sdxl(cfg)
        t_start = time.time()
        for n, i in enumerate(todo, 1):
            t0 = time.time()
            prompt = build_prompt(sp, i)
            best = None
            for attempt in range(cfg.qa_retries + 1):
                seed = shot_seed(sp, i, attempt)
                gen = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
                img = pipe(prompt=prompt, negative_prompt=sp["negative"], width=cfg.kf_w, height=cfg.kf_h,
                           num_inference_steps=steps, guidance_scale=guidance, generator=gen).images[0]
                m = qa_metrics(img)
                ok = qa_pass(m)
                score = qa_score(m) + (1000.0 if ok else 0.0)
                if best is None or score > best[0]:
                    best = (score, img, seed, m, ok, attempt)
                if ok:
                    break
                log(f"   shot {i:03d} attempt {attempt + 1} failed QA {m}")
            _, img, seed, m, ok, attempt = best
            p = kf_dir / f"shot_{i:03d}.jpg"
            img.save(p, quality=95, subsampling=0)
            manifest[str(i)] = {"prompt": prompt, "seed": seed, "qa": m, "qa_ok": ok, "attempts": attempt + 1}
            manifest_path.write_text(json.dumps(manifest, indent=1))
            if cfg.push:
                store.upload(p, "keyframes", p.name, f"keyframe {i:03d}")
            done = time.time() - t_start
            eta = done / n * (len(todo) - n)
            log(f"[keyframes] {n}/{len(todo)} shot {i:03d} {time.time() - t0:.0f}s qa_ok={ok} | ETA {eta / 60:.0f} min")
        del pipe
        free_gpu()
    sheet = make_contact_sheet(kf_dir, indices, wd / "contact_sheet.jpg")
    if sheet and cfg.push:
        store.upload(sheet, "", "contact_sheet.jpg", "keyframe contact sheet")
        if manifest_path.exists():
            store.upload(manifest_path, "", "keyframes_manifest.json", "keyframes manifest")
    log(f"[keyframes] done. contact sheet: {sheet}")


# ==============================================================================
# STAGE 2: SVD HERO SHOTS
# ==============================================================================

def frames_to_mp4(frames: List[Image.Image], out: Path, fps: float, crf: int = 12) -> None:
    w, h = frames[0].size
    enc = start_encoder(out, w, h, fps, crf, "medium")
    try:
        for fr in frames:
            enc.stdin.write(np.asarray(fr.convert("RGB"), dtype=np.uint8).tobytes())
    finally:
        finish_encoder(enc)


def load_svd():
    import torch
    from diffusers import StableVideoDiffusionPipeline
    log("[svd] loading stabilityai/stable-video-diffusion-img2vid-xt ...")
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16")
    pipe.enable_model_cpu_offload()
    try:
        pipe.unet.enable_forward_chunking()
    except Exception:
        pass
    return pipe


def stage_svd(cfg, sp, hero_idx: List[int], store: GitHubStore, wd: Path) -> Set[int]:
    svd_dir = wd / "svd"
    svd_dir.mkdir(parents=True, exist_ok=True)
    have: Set[int] = set()
    if not cfg.use_svd or not hero_idx:
        return have
    remote = store.list_names("svd")
    todo = []
    for i in hero_idx:
        p = svd_dir / f"shot_{i:03d}.mp4"
        if p.exists() and p.stat().st_size > 1000:
            have.add(i)
        elif p.name in remote and store.download("svd", p.name, p):
            have.add(i)
            log(f"[svd] shot {i:03d} restored from GitHub")
        else:
            todo.append(i)
    if not todo:
        return have
    try:
        import torch
        pipe = load_svd()
    except Exception as e:
        log(f"[svd] could not load SVD ({type(e).__name__}: {str(e)[:200]})")
        log("[svd] If this is a 401/403: accept the license at https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt "
            "with the same account as your HF_TOKEN. Hero shots fall back to depth parallax.")
        return have
    t_start = time.time()
    for n, i in enumerate(todo, 1):
        t0 = time.time()
        try:
            kf = Image.open(wd / "keyframes" / f"shot_{i:03d}.jpg").convert("RGB").resize((1024, 576), Image.LANCZOS)
            gen = torch.Generator("cpu").manual_seed(shot_seed(sp, i, 0))
            frames = pipe(kf, num_frames=SVD_FRAMES, decode_chunk_size=2, motion_bucket_id=cfg.svd_motion_bucket,
                          noise_aug_strength=cfg.svd_noise_aug, num_inference_steps=cfg.svd_steps,
                          fps=SVD_FPS, generator=gen).frames[0]
            p = svd_dir / f"shot_{i:03d}.mp4"
            frames_to_mp4(frames, p, SVD_FPS, crf=12)
            have.add(i)
            if cfg.push:
                store.upload(p, "svd", p.name, f"svd clip {i:03d}")
            eta = (time.time() - t_start) / n * (len(todo) - n)
            log(f"[svd] {n}/{len(todo)} shot {i:03d} {time.time() - t0:.0f}s | ETA {eta / 60:.0f} min")
        except Exception as e:
            log(f"[svd] shot {i:03d} failed ({type(e).__name__}: {str(e)[:160]}); falling back to parallax")
        free_gpu()
    del pipe
    free_gpu()
    return have


# ==============================================================================
# STAGE 3: PARALLAX RENDER
# ==============================================================================

def smoothstep(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def sample_grid(move: str, e: float, base, depth):
    """Backward-warp sampling grid. base: (1,H,W,2) in [-1,1]; depth: (1,H,W,1) near=1."""
    if move in ("dolly_in", "dolly_out"):
        k = e if move == "dolly_in" else 1.0 - e
        s = 1.03 + 0.10 * k * (0.4 + 1.2 * depth)
        return base / s
    zoom = 1.10
    c = (e - 0.5) * 2.0
    grid = base / zoom
    if move in ("pan_right", "pan_left"):
        sign = 1.0 if move == "pan_right" else -1.0
        grid = grid.clone()
        grid[..., 0:1] = grid[..., 0:1] + sign * 0.045 * c * (0.3 + depth)
    elif move == "orbit":
        grid = grid.clone()
        grid[..., 0:1] = grid[..., 0:1] + 0.05 * math.sin(c * math.pi / 2.0) * (2.0 * (depth - 0.45))
    elif move in ("crane_up", "crane_down"):
        sign = 1.0 if move == "crane_up" else -1.0
        grid = grid.clone()
        grid[..., 1:2] = grid[..., 1:2] + sign * 0.045 * c * (0.3 + depth)
    return grid


def estimate_depth(depth_pipe, img: Image.Image) -> np.ndarray:
    out = depth_pipe(img)["depth"]  # PIL "L", near = bright
    d = np.asarray(out.resize(img.size, Image.BICUBIC), dtype=np.float32)
    lo, hi = np.percentile(d, 2), np.percentile(d, 98)
    return np.clip((d - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def render_parallax(img: Image.Image, depth_np: np.ndarray, move: str, seconds: float,
                    cfg, out: Path) -> int:
    import torch
    import torch.nn.functional as F
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    W, H, fps = cfg.width, cfg.height, cfg.fps
    n = max(2, int(round(seconds * fps)))
    im = img.convert("RGB").resize((W, H), Image.LANCZOS)
    I = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev)
    D = torch.from_numpy(np.asarray(Image.fromarray((depth_np * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC), dtype=np.float32) / 255.0)
    D = D.unsqueeze(0).unsqueeze(0).to(dev)
    D = F.avg_pool2d(F.avg_pool2d(D, 15, stride=1, padding=7), 15, stride=1, padding=7)
    ys = torch.linspace(-1, 1, H, device=dev)
    xs = torch.linspace(-1, 1, W, device=dev)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    base = torch.stack([gx, gy], dim=-1).unsqueeze(0)
    d_hw = D.permute(0, 2, 3, 1)
    enc = start_encoder(out, W, H, fps, 16, "medium")
    try:
        for k in range(n):
            e = smoothstep(k / (n - 1))
            grid = sample_grid(move, e, base, d_hw)
            fr = F.grid_sample(I, grid, mode="bicubic", padding_mode="border", align_corners=True)
            fr = fr + (torch.randn_like(fr) * (0.8 / 255.0))  # dither: prevents banding in dark space gradients
            arr = (fr.clamp(0, 1) * 255.0).round().byte().squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy()
            enc.stdin.write(arr.tobytes())
    finally:
        finish_encoder(enc)
    return n


def hero_to_clip(svd_mp4: Path, cfg, out: Path) -> None:
    vf = (f"setpts={SVD_SLOWDOWN}*PTS,minterpolate=fps={cfg.fps}:mi_mode=mci:mc_mode=aobmc:vsbmc=1,"
          f"scale={cfg.width}:{cfg.height}:flags=lanczos,format=yuv420p")
    r = subprocess.run([ffmpeg_bin(), "-y", "-v", "error", "-i", str(svd_mp4), "-vf", vf,
                        "-c:v", "libx264", "-preset", "medium", "-crf", "16", str(out)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"hero upscale failed: {r.stderr[-300:]}")


def plan_durations(n_total: int, hero_ok: List[bool], cfg, hero_secs: Optional[List[float]] = None) -> List[float]:
    """Seconds per shot so that (sum - crossfade overlaps) hits cfg.target_seconds.
    hero_secs: measured length of each shot's SVD clip (falls back to HERO_SECONDS)."""
    secs = hero_secs if hero_secs is not None else [HERO_SECONDS] * n_total
    n_par = n_total - sum(hero_ok)
    hero_total = sum(s for s, h in zip(secs, hero_ok) if h)
    overlaps = cfg.crossfade * max(n_total - 1, 0)
    if n_par == 0:
        return list(secs)
    par = (cfg.target_seconds + overlaps - hero_total) / n_par
    par = min(max(par, 4.0), 14.0)
    return [s if h else par for s, h in zip(secs, hero_ok)]


def stage_render(cfg, sp, indices: List[int], svd_ok: Set[int], wd: Path) -> List[Path]:
    clips_dir = wd / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    hero_ok = [bool(sp["shots"][i]["hero"]) and (i in svd_ok) for i in indices]
    # Convert hero clips first so the parallax shots can be sized from their MEASURED length.
    hero_secs: List[float] = []
    for i, is_hero in zip(indices, hero_ok):
        if not is_hero:
            hero_secs.append(HERO_SECONDS)
            continue
        out = clips_dir / f"shot_{i:03d}.mp4"
        if not (out.exists() and out.stat().st_size > 1000):
            hero_to_clip(wd / "svd" / f"shot_{i:03d}.mp4", cfg, out)
        hero_secs.append(count_frames(out) / cfg.fps)
    durs = plan_durations(len(indices), hero_ok, cfg, hero_secs)
    log(f"[render] {sum(hero_ok)} SVD shots + {len(indices) - sum(hero_ok)} parallax shots; "
        f"parallax length {max([d for d, h in zip(durs, hero_ok) if not h] or [0]):.1f}s")
    depth_pipe = None
    paths: List[Path] = []
    t_start = time.time()
    for n, (i, is_hero, dur) in enumerate(zip(indices, hero_ok, durs), 1):
        out = clips_dir / f"shot_{i:03d}.mp4"
        paths.append(out)
        if out.exists() and out.stat().st_size > 1000:
            continue
        t0 = time.time()
        if is_hero:
            pass  # already converted above
        else:
            if depth_pipe is None:
                import torch
                from transformers import pipeline
                log("[render] loading Depth-Anything-V2-Small ...")
                depth_pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf",
                                      device=0 if torch.cuda.is_available() else -1)
            img = Image.open(wd / "keyframes" / f"shot_{i:03d}.jpg").convert("RGB")
            depth = estimate_depth(depth_pipe, img)
            render_parallax(img, depth, sp["shots"][i]["move"], dur, cfg, out)
        eta = (time.time() - t_start) / n * (len(indices) - n)
        log(f"[render] {n}/{len(indices)} shot {i:03d} {'svd ' if is_hero else 'par '}{time.time() - t0:.0f}s | ETA {eta / 60:.0f} min")
    if depth_pipe is not None:
        del depth_pipe
        free_gpu()
    return paths


# ==============================================================================
# AUDIO (generated ambient drone, numpy only)
# ==============================================================================

def synth_ambient(seconds: float, path: Path, sr: int = 32000, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    n_total = int(seconds * sr)
    roots = [55.0, 43.65, 36.71, 41.20]           # A1, F1, D1, E1 : slow minor drift
    ratios = [1.0, 1.5, 2.0, 2.52, 3.0]            # root, fifth, octave, minor-ish third, octave+fifth
    gains = [0.9, 0.5, 0.45, 0.22, 0.18]
    n_sec = len(roots)
    lfo_phase = rng.uniform(0, 2 * np.pi, size=(n_sec, len(ratios)))
    lfo_rate = rng.uniform(0.02, 0.07, size=(n_sec, len(ratios)))
    wind_sr = 400
    wind_raw = rng.normal(size=int(seconds * wind_sr) + 2)
    k = np.ones(40) / 40.0
    wind_raw = np.convolve(wind_raw, k, mode="same")
    wind_raw /= max(np.abs(wind_raw).max(), 1e-6)
    block = sr * 10
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        for start in range(0, n_total, block):
            end = min(start + block, n_total)
            t = np.arange(start, end, dtype=np.float64) / sr
            pos = t / max(seconds, 1e-6) * n_sec
            lr = []
            for ch, detune in ((0, 0.0), (1, 0.35)):
                sig = np.zeros(end - start, dtype=np.float64)
                for s in range(n_sec):
                    w = np.clip(1.0 - np.abs(pos - (s + 0.5)), 0.0, 1.0)   # triangular weights, sum to 1
                    if w.max() <= 0.0:
                        continue
                    pad = np.zeros_like(t)
                    for j, (ratio, g) in enumerate(zip(ratios, gains)):
                        f = roots[s] * ratio
                        amp = 0.6 + 0.4 * np.sin(2 * np.pi * lfo_rate[s, j] * t + lfo_phase[s, j])
                        pad += g * amp * (np.sin(2 * np.pi * (f + detune) * t + ch) + 0.6 * np.sin(2 * np.pi * (f * 1.004 - detune) * t))
                    sig += w * pad
                wnd = np.interp(t, np.arange(len(wind_raw)) / wind_sr, wind_raw)
                sig += 0.35 * wnd * np.sin(2 * np.pi * 0.03 * t + ch)
                lr.append(sig)
            stereo = np.stack(lr, axis=1)
            stereo *= 0.18 / 3.0
            # fades: 3s in, 6s out
            env = np.minimum(1.0, t / 3.0) * np.minimum(1.0, np.maximum(0.0, (seconds - t) / 6.0))
            stereo *= env[:, None]
            wf.writeframes((np.clip(stereo, -1, 1) * 32767).astype("<i2").tobytes())


# ==============================================================================
# STAGE 4: ASSEMBLY
# ==============================================================================

def blend(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    return (a.astype(np.float32) * (1.0 - alpha) + b.astype(np.float32) * alpha + 0.5).astype(np.uint8)


def assemble_master(clips: List[Path], cfg, wd: Path, out: Path) -> Dict[str, Any]:
    W, H, fps = cfg.width, cfg.height, cfg.fps
    xf = max(0, int(round(cfg.crossfade * fps)))
    counts = [count_frames(c) for c in clips]
    total_frames = sum(counts) - xf * (len(clips) - 1)
    dur = total_frames / fps
    log(f"[assemble] {len(clips)} clips, {total_frames} frames = {dur:.1f}s ({dur / 60:.2f} min)")
    audio = None
    if cfg.audio:
        audio = wd / "ambient.wav"
        log("[assemble] synthesizing ambient audio ...")
        synth_ambient(dur + 0.2, audio)
    vf = (f"eq=contrast=1.04:saturation=1.06,vignette=angle=PI/7,noise=alls=2:allf=t,"
          f"fade=t=in:st=0:d=1.5,fade=t=out:st={max(dur - 2.5, 0):.2f}:d=2.5")
    enc = start_encoder(out, W, H, fps, cfg.crf, cfg.preset, vf=vf, audio=audio)
    written = 0
    try:
        tail: deque = deque()
        for ci, clip in enumerate(clips):
            head_i = 0
            prev_tail = list(tail)
            tail.clear()
            for fr in decode_frames(clip, W, H):
                if prev_tail and head_i < len(prev_tail):
                    alpha = (head_i + 1) / (len(prev_tail) + 1)
                    fr = blend(prev_tail[head_i], fr, alpha)
                    head_i += 1
                tail.append(fr.copy())
                if len(tail) > xf:
                    enc.stdin.write(tail.popleft().tobytes())
                    written += 1
            if (ci + 1) % 10 == 0:
                log(f"[assemble] clip {ci + 1}/{len(clips)} done ({written} frames written)")
        while tail:
            enc.stdin.write(tail.popleft().tobytes())
            written += 1
    finally:
        finish_encoder(enc)
    size_mb = out.stat().st_size / 1048576
    log(f"[assemble] master {out} {size_mb:.1f} MB, {written} frames")
    return {"frames": written, "seconds": written / fps, "size_mb": size_mb}


def push_master(master: Path, store: GitHubStore, cfg) -> List[str]:
    names: List[str] = []
    if not (cfg.push and store.token):
        return names
    part_size = 40 * 1024 * 1024
    data = master.read_bytes()
    parts = [data[i:i + part_size] for i in range(0, len(data), part_size)]
    for n, chunk in enumerate(parts):
        name = f"{master.stem}.mp4.part{n:02d}"
        tmp = master.parent / name
        tmp.write_bytes(chunk)
        ok = store.upload(tmp, "master", name, f"master part {n + 1}/{len(parts)}")
        tmp.unlink()
        log(f"[push] {name} {'OK' if ok else 'FAILED'}")
        if ok:
            names.append(name)
    log("[push] rejoin on Windows:  copy /b cosmic_master.mp4.part* cosmic_master.mp4")
    return names


# ==============================================================================
# ORCHESTRATOR
# ==============================================================================

def run(cfg) -> Dict[str, Any]:
    sp = load_screenplay(cfg.screenplay)
    wd = Path(cfg.workdir)
    wd.mkdir(parents=True, exist_ok=True)
    token = cfg.gh_token or resolve_secret(["GH_TOKEN", "GITHUB_TOKEN"])
    store = GitHubStore(token, cfg.repo, cfg.branch, cfg.remote_dir, enabled=cfg.push)
    if cfg.push and not token:
        log("[warn] no GitHub token found: running locally without remote resume/push")
        store.enabled = False
    indices = list(cfg.shots) if cfg.shots is not None else list(range(len(sp["shots"])))
    stages = [s.strip() for s in cfg.stages.split(",") if s.strip()]
    results: Dict[str, Any] = {}
    timings: Dict[str, float] = {}

    def timed(name, fn):
        t0 = time.time()
        try:
            r = fn()
            results[name] = "PASS"
            return r
        except Exception as e:
            results[name] = f"FAIL: {type(e).__name__}: {str(e)[:300]}"
            log(f"[{name}] FAILED: {results[name]}")
            if not cfg.smoke:
                raise
            return None
        finally:
            timings[name] = time.time() - t0

    log(f"=== Cinema engine: {len(indices)} shots, tier={cfg.tier}, target={cfg.target_seconds:.0f}s ===")
    svd_ok: Set[int] = set()
    clips: List[Path] = []
    if "keyframes" in stages:
        timed("keyframes", lambda: stage_keyframes(cfg, sp, indices, store, wd))
    if "svd" in stages:
        hero = [i for i in indices if sp["shots"][i]["hero"]]
        svd_ok = timed("svd", lambda: stage_svd(cfg, sp, hero, store, wd)) or set()
    else:
        svd_ok = {i for i in indices if (wd / "svd" / f"shot_{i:03d}.mp4").exists()}
    if "render" in stages:
        clips = timed("render", lambda: stage_render(cfg, sp, indices, svd_ok, wd)) or []
    master = wd / "out" / "cosmic_master.mp4"
    if "assemble" in stages:
        master.parent.mkdir(parents=True, exist_ok=True)
        if not clips:
            clips = [wd / "clips" / f"shot_{i:03d}.mp4" for i in indices]
        info = timed("assemble", lambda: assemble_master(clips, cfg, wd, master))
        if info and results.get("assemble") == "PASS" and not cfg.smoke:
            push_master(master, store, cfg)
        results["master"] = info
    log("=== SUMMARY ===")
    for k, v in results.items():
        log(f"  {k}: {v}" + (f"  ({timings[k]:.0f}s)" if k in timings else ""))
    if cfg.smoke:
        n_all = len(sp["shots"])
        n_hero_all = sum(1 for s in sp["shots"] if s["hero"])
        log("  smoke per-stage timings above are for the smoke subset only.")
        log(f"  full run = {n_all} shots ({n_hero_all} SVD): scale keyframes/svd/render times by shot counts.")
    return {"results": results, "timings": timings, "master": str(master)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Keyframe cinema engine")
    ap.add_argument("--screenplay", default="screenplay_cosmic.json")
    ap.add_argument("--workdir", default="/content/cinema_work")
    ap.add_argument("--tier", default="final", choices=["final", "draft"])
    ap.add_argument("--stages", default="keyframes,svd,render,assemble")
    ap.add_argument("--target-seconds", type=float, default=600.0)
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--no-svd", action="store_true")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = make_cfg(screenplay=a.screenplay, workdir=a.workdir, tier=a.tier, stages=a.stages,
                   target_seconds=a.target_seconds, push=not a.no_push, use_svd=not a.no_svd,
                   audio=not a.no_audio, smoke=a.smoke)
    run(cfg)


if __name__ == "__main__":
    main()
