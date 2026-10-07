#!/usr/bin/env python3
"""
================================================================================
🎬 AUTONOMOUS CONSISTENT CINEMA ENGINE (COLAB T4 / L4 / G4 / PRO) 🎬
File: bulk_generator/templates/consistent_cinema_engine.py

Core Architecture:
  1. Autoregressive Tail-to-Head Video Continuity:
     - Shot N+1 uses the exact tail frame of Shot N as its conditioning image.
     - Eliminates jump cuts; enforces vector, lighting, and entity continuity.
  2. Dynamic GPU Adapter (LTX-Video):
     - On L4 (24GB VRAM): Direct CUDA execution in bfloat16.
     - On T4 (15GB VRAM): Auto-engages model CPU offload + VAE slicing + CPU generator.
     - Generates 161 frames @ 24 FPS (~6.7s per shot).
  3. Live Frontier Director API Bridge (api1.flh-one.fun):
     - Outbound inspection of tail frames via Cloudflare Named Tunnel.
     - Zero reverse proxies inside Colab.
     - Automatic seamless fallback to Master Storyboard manifest.
  4. Resumable Atomic Persistence:
     - Each shot & tail frame committed to GitHub REST API immediately upon generation.
     - Reconnecting Colab skips all completed shots instantly.
  5. Cinematic Master Assembly:
     - Concatenation, procedural spatial soundscape, 2.39:1 letterbox, 35mm grain.
================================================================================
"""

import os
import gc
import json
import time
import wave
import base64
import shutil
import argparse
import subprocess
import urllib.request
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from typing import List, Dict, Any, Set, Tuple, Optional

import torch
import numpy as np
from PIL import Image

# Suppress noisy logging
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")


def log(msg: str) -> None:
    print(msg, flush=True)


# ==============================================================================
# ⚙️ CONFIGURATION & HARDWARE DEFAULTS
# ==============================================================================

def make_cfg(**overrides: Any) -> SimpleNamespace:
    cfg = dict(
        storyboard="storyboard_2min_sukuna_vs_mahoraga.json",
        workdir="/content/cinema_pro_work",
        model_id="Lightricks/LTX-Video",
        repo="gglshtopenaividgen-lgtm/test1",
        branch="main",
        remote_dir="pro_cinema",
        width=704,
        height=512,
        num_frames=161,             # 161 frames @ 24fps = ~6.7s per shot
        fps=24,
        num_inference_steps=30,      # Crisp motion & quality
        guidance_scale=3.0,          # Adherence to negative prompt
        seed=42,
        smoke=False,
        director_url="https://api1.flh-one.fun", # Public Director API tunnel
        audio=True,
        push=True,
        gh_token="",
        hf_token="",
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def resolve_secret(names: List[str]) -> str:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    try:
        from google.colab import userdata
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
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        raise RuntimeError("ffmpeg not found (run: apt install -y ffmpeg)") from e


# ==============================================================================
# 🐙 GITHUB REST API STORE (RESUMPTION & PERSISTENCE)
# ==============================================================================

class GitHubStore:
    def __init__(self, token: str, repo: str, branch: str, remote_dir: str, enabled: bool = True):
        repo_clean = repo.strip().replace("https://github.com/", "").rstrip("/")
        if repo_clean.endswith(".git"):
            repo_clean = repo_clean[:-4]
        self.repo = repo_clean
        self.branch = branch
        self.remote_dir = remote_dir.strip("/")
        self.token = (token or "").strip()
        self.enabled = bool(enabled and self.token)
        self.headers = {"Accept": "application/vnd.github+json", "User-Agent": "consistent-cinema/2.0"}
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
                log(f"   [github] list {subdir} error: HTTP {e.code}")
            return set()
        except Exception:
            return set()

    def download(self, subdir: str, name: str, local: Path) -> bool:
        if not self.enabled:
            return False
        url = f"https://raw.githubusercontent.com/{self.repo}/{self.branch}/{self._path(subdir, name)}"
        try:
            local.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(urllib.request.Request(url, headers=self.headers), timeout=60) as r:
                data = r.read()
            if not data:
                return False
            local.write_bytes(data)
            return True
        except Exception as e:
            log(f"   [github] download {name} failed: {e}")
            return False

    def upload(self, local: Path, subdir: str, name: str, message: str) -> bool:
        if not self.enabled:
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
                body = {"message": message, "content": content, "branch": self.branch}
                if sha:
                    body["sha"] = sha
                req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=self.headers, method="PUT")
                with urllib.request.urlopen(req, timeout=120):
                    return True
            except urllib.error.HTTPError as e:
                log(f"   [github] upload {name} HTTP {e.code} (try {attempt + 1}/4)")
                if e.code in (401, 403, 404) and attempt >= 1:
                    return False
            except Exception as e:
                log(f"   [github] upload {name} error (try {attempt + 1}/4): {e}")
            time.sleep(2 + attempt * 2)
        return False


# ==============================================================================
# 📡 FRONTIER DIRECTOR API BRIDGE
# ==============================================================================

def query_director_api(
    director_url: str,
    shot_index: int,
    total_shots: int,
    tail_img_path: Optional[Path] = None,
    fallback_prompt: str = ""
) -> Tuple[str, str]:
    """Sends tail frame to Antigravity Director API; falls back to storyboard prompt."""
    if not director_url:
        return fallback_prompt, "Defaulted to storyboard"
    try:
        payload: Dict[str, Any] = {
            "shot_index": shot_index,
            "total_shots": total_shots,
            "last_prompt": fallback_prompt
        }
        if tail_img_path and tail_img_path.exists():
            payload["tail_image_base64"] = base64.b64encode(tail_img_path.read_bytes()).decode("utf-8")

        req = urllib.request.Request(
            f"{director_url.rstrip('/')}/v1/director/inspect",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "consistent-cinema/2.0"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=12) as r:
            data = json.loads(r.read().decode("utf-8"))
            if data.get("success") and data.get("next_prompt"):
                notes = data.get("director_notes", "Verified continuity")
                return data["next_prompt"], notes
    except Exception as e:
        log(f"   ℹ️ [Director API] Bridge notice ({e}); using storyboard manifest.")
    return fallback_prompt, "Storyboard manifest beat"


# ==============================================================================
# 🎞️ VIDEO ENCODING & AUDIO SYNTHESIS
# ==============================================================================

def export_frames_to_mp4(frames: List[Image.Image], out_path: Path, fps: int = 24, crf: int = 18) -> None:
    """Encodes list of PIL Images to high-bitrate MP4 using FFmpeg."""
    w, h = frames[0].size
    cmd = [
        ffmpeg_bin(), "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-crf", str(crf), "-preset", "fast",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(out_path)
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for fr in frames:
            proc.stdin.write(np.asarray(fr.convert("RGB"), dtype=np.uint8).tobytes())
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.wait()


def synth_cinematic_audio(seconds: float, path: Path, sr: int = 32000, seed: int = 42) -> None:
    """Synthesizes dynamic, menacing dark anime ambient battle drone."""
    rng = np.random.default_rng(seed)
    n_total = int(seconds * sr)
    # Dark cinematic Phrygian drone progression: Eb1, C1, D1, F1
    roots = [38.89, 32.70, 36.71, 43.65]
    ratios = [1.0, 1.414, 2.0, 2.828, 4.0] # Dissonant tritones & octaves
    gains = [1.0, 0.45, 0.35, 0.20, 0.10]
    n_sec = len(roots)
    lfo_phases = rng.uniform(0, 2 * np.pi, size=(n_sec, len(ratios)))

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
                    w = np.clip(1.0 - np.abs(pos - (s + 0.5)), 0.0, 1.0)
                    if w.max() <= 0.0:
                        continue
                    pad = np.zeros_like(t)
                    for j, (ratio, g) in enumerate(zip(ratios, gains)):
                        f = roots[s] * ratio
                        amp = 0.6 + 0.4 * np.sin(2 * np.pi * 0.05 * t + lfo_phases[s, j])
                        pad += g * amp * np.sin(2 * np.pi * (f + detune) * t + ch)
                    sig += w * pad
                lr.append(sig)
            stereo = np.stack(lr, axis=1) * (0.16 / len(ratios))
            env = np.minimum(1.0, t / 2.0) * np.minimum(1.0, np.maximum(0.0, (seconds - t) / 4.0))
            stereo *= env[:, None]
            wf.writeframes((np.clip(stereo, -1, 1) * 32767).astype("<i2").tobytes())


def assemble_continuous_film(
    scene_paths: List[Path],
    output_path: Path,
    fps: int = 24,
    audio: bool = True,
    letterbox: bool = True
) -> Dict[str, Any]:
    """Seamlessly concatenates autoregressively chained shots and applies cinematic grade."""
    log("\n" + "=" * 70)
    log("🎞️ ASSEMBLING CONTINUOUS CINEMATIC FILM VIA FFMPEG...")
    log("=" * 70)

    concat_file = output_path.parent / "scenes_concat.txt"
    with open(concat_file, "w", encoding="utf-8") as f:
        for p in scene_paths:
            f.write(f"file '{p.resolve().as_posix()}'\n")

    audio_wav = output_path.parent / "ambient_soundtrack.wav"
    total_est_seconds = len(scene_paths) * (161 / fps)
    if audio:
        log("🎵 Synthesizing spatial ambient audio soundtrack...")
        synth_cinematic_audio(total_est_seconds + 1.0, audio_wav)

    # 35mm grain, contrast bump, vignette, letterbox 2.39:1
    vf_filters = [
        "eq=contrast=1.06:saturation=1.05",
        "vignette=angle=PI/8",
        "noise=alls=2:allf=t",
        "fade=t=in:st=0:d=1.5",
        f"fade=t=out:st={max(total_est_seconds - 3.0, 0):.2f}:d=3.0"
    ]
    if letterbox:
        vf_filters.append("crop=w=in_w:h=in_w/2.39,pad=w=in_w:h=in_h:x=(ow-iw)/2:y=(oh-ih)/2:color=black")

    vf = ",".join(vf_filters)
    cmd = [
        ffmpeg_bin(), "-y", "-v", "error",
        "-f", "concat", "-safe", "0", "-i", str(concat_file)
    ]
    if audio and audio_wav.exists():
        cmd += ["-i", str(audio_wav)]
    cmd += [
        "-vf", vf,
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p"
    ]
    if audio and audio_wav.exists():
        cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd += ["-movflags", "+faststart", str(output_path)]

    subprocess.run(cmd, check=True)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    log(f"🎉 MASTER FILM ASSEMBLED: {output_path} ({size_mb:.2f} MB)")
    return {"size_mb": size_mb, "path": str(output_path)}


# ==============================================================================
# 🧠 AUTOREGRESSIVE TAIL-TO-HEAD VIDEO ENGINE (DYNAMIC T4 / L4 ADAPTER)
# ==============================================================================

class ConsistentCinemaEngine:
    """Manages LTX-Video pipeline with dynamic VRAM offloading and autoregressive chaining."""

    def __init__(self, cfg: SimpleNamespace):
        from diffusers import LTXImageToVideoPipeline

        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        log(f"🎬 Initializing LTX-Video Pipeline (dtype={self.dtype})...")
        self.pipe = LTXImageToVideoPipeline.from_pretrained(
            cfg.model_id,
            torch_dtype=self.dtype,
            token=cfg.hf_token or None
        )

        if self.device == "cuda":
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            log(f"🎮 GPU Device: {torch.cuda.get_device_name(0)} ({total_vram_gb:.1f} GB VRAM)")
            if total_vram_gb < 20.0:
                log(f"⚡ VRAM is {total_vram_gb:.1f} GB (<20 GB, e.g. T4). Enabling model CPU offload & VAE slicing...")
                self.pipe.enable_model_cpu_offload()
                # Anti-Pattern 17 fix: model_cpu_offload requires CPU generator!
                self.gen_device = "cpu"
            else:
                log(f"⚡ High VRAM detected ({total_vram_gb:.1f} GB, e.g. L4/A100). Loading direct to CUDA...")
                self.pipe.to("cuda")
                self.gen_device = "cuda"
        else:
            self.gen_device = "cpu"

        if hasattr(self.pipe, "enable_vae_tiling"):
            self.pipe.enable_vae_tiling()
        if hasattr(self.pipe, "enable_vae_slicing"):
            self.pipe.enable_vae_slicing()

        log(f"✅ LTX-Video Engine Ready (Generator device: {self.gen_device}).")

    def generate_shot(
        self,
        cond_image: Image.Image,
        prompt: str,
        negative_prompt: str,
        seed: int
    ) -> Tuple[List[Image.Image], Image.Image]:
        """Renders video conditioned on cond_image; returns all frames and exact last frame."""
        import torch

        # Ensure image dimensions match model requirements (divisible by 32)
        cond_img_resized = cond_image.convert("RGB").resize((self.cfg.width, self.cfg.height), Image.LANCZOS)

        gen = torch.Generator(device=self.gen_device).manual_seed(seed)
        output = self.pipe(
            image=cond_img_resized,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=self.cfg.width,
            height=self.cfg.height,
            num_frames=self.cfg.num_frames,
            frame_rate=self.cfg.fps,
            num_inference_steps=self.cfg.num_inference_steps,
            guidance_scale=self.cfg.guidance_scale,
            generator=gen
        )
        frames = output.frames[0]
        tail_frame = frames[-1]
        return frames, tail_frame


def create_anchor_frame(prompt: str, width: int, height: int, seed: int) -> Image.Image:
    """Generates initial Shot 1 visual anchor still via SDXL with complete memory cleanup."""
    try:
        import torch
        from diffusers import AutoencoderKL, StableDiffusionXLPipeline, DPMSolverMultistepScheduler

        log("🎨 Generating Pristine Shot 1 Visual Anchor Still via SDXL...")
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        sdxl = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        if torch.cuda.is_available():
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if total_vram < 20.0:
                sdxl.enable_model_cpu_offload()
                gen = torch.Generator("cpu").manual_seed(seed)
            else:
                sdxl.to("cuda")
                gen = torch.Generator("cuda").manual_seed(seed)
        else:
            gen = torch.Generator("cpu").manual_seed(seed)

        sdxl.scheduler = DPMSolverMultistepScheduler.from_config(sdxl.scheduler.config, use_karras_sigmas=True)
        img = sdxl(prompt=prompt, width=width, height=height, num_inference_steps=28, guidance_scale=6.5, generator=gen).images[0]

        del sdxl, vae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        log("✅ Visual Anchor synthesized successfully.")
        return img
    except Exception as e:
        log(f"⚠️ SDXL anchor fallback ({e}); using procedural high-contrast dark anime anchor.")
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        arr[int(height * 0.4):, :, 0] = 18
        arr[int(height * 0.4):, :, 1] = 24
        arr[int(height * 0.4):, :, 2] = 42
        return Image.fromarray(arr)


# ==============================================================================
# 🚀 MAIN PIPELINE CONTROLLER
# ==============================================================================

def run_consistent_pipeline(cfg: SimpleNamespace) -> Dict[str, Any]:
    wd = Path(cfg.workdir)
    scenes_dir = wd / "scenes"
    tails_dir = wd / "tails"
    scenes_dir.mkdir(parents=True, exist_ok=True)
    tails_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Storyboard
    with open(cfg.storyboard, "r", encoding="utf-8") as f:
        sb = json.load(f)

    shots = sb["shots"]
    if cfg.smoke:
        shots = shots[:2]
        log("🧪 RUNNING IN SMOKE-TEST MODE (2 Shots)")

    style_bible = ""
    if "aesthetic_bible" in sb:
        style_bible = sb["aesthetic_bible"].get("visual_style", "")
        neg_prompt = sb["aesthetic_bible"].get("universal_negative_prompt", "")
    else:
        style_bible = sb.get("style_bible", "")
        neg_prompt = sb.get("negative_prompt", "")

    # 2. Setup GitHub Store
    token = cfg.gh_token or resolve_secret(["GH_TOKEN", "GITHUB_TOKEN"])
    hf_token = cfg.hf_token or resolve_secret(["HF_TOKEN"])
    cfg.gh_token = token
    cfg.hf_token = hf_token

    store = GitHubStore(token, cfg.repo, cfg.branch, cfg.remote_dir, enabled=cfg.push)
    remote_scenes = store.list_names("scenes")
    log(f"🔍 Existing scenes on GitHub: {len(remote_scenes)} found.")

    # 3. Resolve or Generate Anchor Frame (Shot 1 Condition) BEFORE loading LTX-Video!
    anchor_path = wd / "anchor_shot_001.png"
    if not anchor_path.exists():
        if "anchor_shot_001.png" in store.list_names(""):
            store.download("", "anchor_shot_001.png", anchor_path)
            log("✅ Restored anchor frame from GitHub.")
        else:
            first_prompt = f"{shots[0]['prompt']}, {style_bible}"
            anchor_img = create_anchor_frame(first_prompt, cfg.width, cfg.height, cfg.seed)
            anchor_img.save(anchor_path)
            store.upload(anchor_path, "", "anchor_shot_001.png", "feat(anchor): add initial visual anchor")

    current_cond_img = Image.open(anchor_path)
    current_tail_path: Optional[Path] = anchor_path
    completed_scene_paths: List[Path] = []

    # 4. Now initialize LTX-Video Engine (pure memory space)
    engine = ConsistentCinemaEngine(cfg)

    log("\n" + "=" * 70)
    log(f"🎬 COMMENCING AUTOREGRESSIVE CHAIN ({len(shots)} CONTINUOUS SHOTS)")
    log("=" * 70)

    start_time = time.time()

    for idx, shot in enumerate(shots, 1):
        shot_filename = f"shot_{idx:03d}.mp4"
        tail_filename = f"tail_{idx:03d}.png"
        local_scene = scenes_dir / shot_filename
        local_tail = tails_dir / tail_filename

        # Resumption Check: If shot exists locally or on GitHub, restore and continue
        if local_scene.exists() and local_tail.exists():
            log(f"⏩ [Shot {idx:03d}/{len(shots)}] Already present locally. Skipping generation.")
            completed_scene_paths.append(local_scene)
            current_cond_img = Image.open(local_tail)
            current_tail_path = local_tail
            continue
        elif shot_filename in remote_scenes:
            if store.download("scenes", shot_filename, local_scene) and store.download("tails", tail_filename, local_tail):
                log(f"⏩ [Shot {idx:03d}/{len(shots)}] Restored from GitHub. Skipping generation.")
                completed_scene_paths.append(local_scene)
                current_cond_img = Image.open(local_tail)
                current_tail_path = local_tail
                continue

        # Formulate Directional Prompt with Vector Continuity
        base_prompt = shot["prompt"]
        if style_bible and style_bible not in base_prompt:
            base_prompt = f"{base_prompt}, {style_bible}"

        # Bridge call to Director API (or storyboard fallback)
        directed_prompt, director_notes = query_director_api(
            cfg.director_url,
            shot_index=idx - 1,
            total_shots=len(shots),
            tail_img_path=current_tail_path,
            fallback_prompt=base_prompt
        )

        t0 = time.time()
        log(f"\n🎥 [Shot {idx:03d}/{len(shots)}] Rendering ({shot.get('title', f'Shot {idx}')})...")
        log(f"   Continuity Input: {'Anchor Frame' if idx == 1 else f'Tail of Shot {idx-1:03d}'}")
        log(f"   Camera Motion: {shot.get('camera_motion', 'Tracking')}")
        log(f"   Director Directive: {director_notes}")

        shot_seed = cfg.seed + (idx * 1337)
        frames, tail_frame = engine.generate_shot(
            cond_image=current_cond_img,
            prompt=directed_prompt,
            negative_prompt=neg_prompt,
            seed=shot_seed
        )
        gen_time = time.time() - t0
        log(f"   ⚡ Rendered {len(frames)} frames in {gen_time:.1f}s ({gen_time/len(frames):.2f}s/frame)")

        # Save Scene MP4 and Tail Frame
        export_frames_to_mp4(frames, local_scene, fps=cfg.fps)
        tail_frame.save(local_tail)
        completed_scene_paths.append(local_scene)

        # Update condition image and tail path for the next shot!
        current_cond_img = tail_frame
        current_tail_path = local_tail

        # Commit to GitHub
        if store.enabled:
            store.upload(local_scene, "scenes", shot_filename, f"feat(shot): add {shot_filename}")
            store.upload(local_tail, "tails", tail_filename, f"feat(tail): add {tail_filename}")

        # Memory hygiene
        del frames
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # 5. Assemble Final Continuous Master Video
    master_name = "sukuna_vs_mahoraga_2min_master.mp4"
    master_path = wd / "out" / master_name
    master_path.parent.mkdir(parents=True, exist_ok=True)
    assembly_info = assemble_continuous_film(completed_scene_paths, master_path, fps=cfg.fps, audio=cfg.audio)

    if store.enabled:
        log("🐙 Committing Master 2-Minute Film to GitHub...")
        store.upload(master_path, "", master_name, "feat(master): add compiled 2-minute consistent film")

    total_duration_min = (time.time() - start_time) / 60
    log("\n" + "=" * 70)
    log(f"🎉 2-MINUTE CONTINUOUS FILM COMPLETE IN {total_duration_min:.2f} MINUTES!")
    log(f"📁 Output: {master_path} ({assembly_info['size_mb']:.2f} MB)")
    log("=" * 70)

    return {
        "master_path": str(master_path),
        "total_minutes": total_duration_min,
        "scenes_count": len(completed_scene_paths)
    }


# ==============================================================================
# 🏁 CLI ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous Consistent Cinema Engine")
    parser.add_argument("--storyboard", type=str, default="storyboard_2min_sukuna_vs_mahoraga.json")
    parser.add_argument("--workdir", type=str, default="/content/cinema_pro_work")
    parser.add_argument("--smoke", action="store_true", help="Run 2-shot smoke test")
    parser.add_argument("--no-push", action="store_true", help="Disable GitHub commits")
    parser.add_argument("--no-audio", action="store_true", help="Disable audio track")
    args = parser.parse_args()

    cfg = make_cfg(
        storyboard=args.storyboard,
        workdir=args.workdir,
        smoke=args.smoke,
        push=not args.no_push,
        audio=not args.no_audio
    )
    run_consistent_pipeline(cfg)
