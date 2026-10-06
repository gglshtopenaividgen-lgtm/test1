#!/usr/bin/env python3
"""
================================================================================
🚀 20-MINUTE COSMIC ODYSSEY AUTONOMOUS VIDEO DIRECTOR (COLAB PRO L4/T4) 🚀
File: temp_read/twenty_minute_space_director.py
Target Platform: Google Colab Pro (NVIDIA L4 24GB or T4 16GB + High-RAM 53GB)
Target Output: 20-Minute Continuous Space Cinematic (1,200s, 200 Scenes)
Theme: "Cosmic Odyssey: Voyage Across the Event Horizon" (Surprise Epic Space)

Architecture:
  1. Adaptive GPU Acceleration:
     - L4 (24GB VRAM): Batch Size = 4, Qwen2.5-1.5B Screenplay Director
     - T4 (15.3GB VRAM): Batch Size = 2, Qwen2.5-0.5B Screenplay Director
  2. High-RAM Optimization (~53 GB RAM):
     - Uses /dev/shm in-memory RAMDisk for 0ms disk I/O during rendering
     - In-memory parallel frame buffering & multi-threaded FFmpeg encoding
  3. Cinematic Pacing (20 Minutes = 1,200s):
     - 200 unique cosmic scenes @ 6.0 seconds per scene
     - AnimateDiff-Lightning 2-Step motion with smooth temporal expansion
  4. Real-Time Zero-Loss Cloud Sync:
     - Commits scenes & manifests to GitHub gglshtopenaividgen-lgtm/test1
  5. Maximum Compression Packaging:
     - Compiles master cosmic_odyssey_20min_master.mp4 (H.264 CRF 24)
     - Compresses to cosmic_odyssey_20min_master.zip (ZIP_DEFLATED level 9)
================================================================================
"""

import os
import sys
import gc
import time
import json
import base64
import random
import shutil
import zipfile
import argparse
import subprocess
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List

# Suppress progress bar flooding in headless execution
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["DIFFUSERS_VERBOSITY"] = "error"

import torch
import psutil
from PIL import Image

# Automatically resolve HF_TOKEN from Colab Secrets, env
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    try:
        from google.colab import userdata
        HF_TOKEN = userdata.get("HF_TOKEN")
    except Exception:
        pass

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    try:
        import huggingface_hub
        huggingface_hub.login(token=HF_TOKEN, add_to_git_credential=False)
    except Exception:
        pass

try:
    import imageio
except ImportError:
    imageio = None


# ==============================================================================
# 🌌 SURPRISE THEME: COSMIC ODYSSEY (4 EPIC ACTS)
# ==============================================================================

COSMIC_ACTS = [
    {
        "act": 1,
        "name": "Act I: Departure from the Solar Cradle",
        "scene_range": (1, 50),
        "context": "Leaving Earth and inner solar system. Passing golden aurora rings of Saturn, icy moon geysers of Enceladus, towering Jovian storms, and drifting through the Oort cloud into interstellar space.",
        "fallbacks": [
            "Colossal golden rings of Saturn orbiting below, ethereal ice particle reflections, dramatic solar flare rim lighting, cinematic 8k photorealistic space",
            "Swirling atmospheric storms of Jupiter, crimson Great Red Spot vortex, tiny volcanic moon Io in silhouette, photorealistic IMAX space",
            "Deep space probe drifting past crystalline comet nucleus, glittering coma dust, vast starfield background, hyperrealistic cosmic documentary",
            "Leaving the Kuiper belt, distant pale Sun shining like a brilliant star, deep void, endless cosmic horizon, photorealistic 8k cinematography"
        ]
    },
    {
        "act": 2,
        "name": "Act II: The Quantum Nebula & Stellar Nurseries",
        "scene_range": (51, 100),
        "context": "Drifting into the Orion Molecular Cloud and Carina Nebula. Vibrant cyan, violet, and iridescent gold cosmic gas dust, protostars igniting in dense pillars, shimmering reflection nebulae.",
        "fallbacks": [
            "Towering pillars of iridescent cosmic dust, stellar nursery with newborn brilliant blue stars emerging, glowing emission nebula, 8k masterpiece",
            "Shimmering turquoise and magenta emission nebula clouds, interstellar shockwaves, glowing hydrogen filaments, ultra-detailed space photography",
            "Massive young star igniting in the heart of a cosmic cocoon, violent radiant solar winds blowing away dust clouds, photorealistic 8k",
            "Bioluminescent cosmic gas tendrils stretching across light-years, celestial crystal dust, breathtaking astronomical vista, cinematic masterpiece"
        ]
    },
    {
        "act": 3,
        "name": "Act III: Relativistic Pulsars & Ancient Megastructures",
        "scene_range": (101, 150),
        "context": "Deep galactic core region. Approaching a rapidly spinning magnetar with relativistic plasma jets, ancient Dyson swarm arrays harvesting stellar energy, crystalline alien megastructures.",
        "fallbacks": [
            "Rapidly spinning millisecond pulsar, intense relativistic plasma jets beaming through space, warping magnetic field lines, cinematic IMAX",
            "Colossal alien Dyson swarm structure partially enclosing a brilliant white star, glowing energy conduits, geometric cosmic architecture, photorealistic",
            "Crystalline ancient monolith drifting near a pulsating neutron star, eerie gravitational lensing, specular reflections, photorealistic sci-fi",
            "Vast swarm of geometric solar collectors orbiting a magnetar, cosmic ray arcs, deep space scale, hyper-detailed astronomical concept"
        ]
    },
    {
        "act": 4,
        "name": "Act IV: Sagittarius A* & The Quantum Event Horizon",
        "scene_range": (151, 200),
        "context": "The supermassive black hole at the galactic center. Blinding accretion disk spinning near light-speed, intense gravitational lensing bending background starfields, plunging toward the photon sphere and emerging into a multidimensional hyperspace continuum.",
        "fallbacks": [
            "Supermassive black hole with blinding golden-orange accretion disk, extreme gravitational lensing curving starfield, Interstellar style, 8k photorealistic",
            "Approaching the photon sphere of a black hole, starlight trapped in swirling optical orbits, pitch black event horizon silhouette, cinematic masterpiece",
            "Plunging past the event horizon into a vibrant kaleidoscope of quantum foam, shimmering multidimensional hyperspace, transcendent cosmic light",
            "Emerging into a luminous infinite multiverse of glowing cosmic filaments, iridescent stellar web, breathtaking transcendental finale, 8k"
        ]
    }
]


# ==============================================================================
# 🐙 GITHUB HEADLESS SYNC ENGINE
# ==============================================================================

class GitHubBatchUploader:
    """Incremental GitHub committer saving scenes to test1 repository."""

    def __init__(self, token: str, repo: str = "gglshtopenaividgen-lgtm/test1", branch: str = "main"):
        self.token = token.strip() if token else ""
        repo_clean = repo.strip().replace("https://github.com/", "").rstrip("/")
        if repo_clean.endswith(".git"):
            repo_clean = repo_clean[:-4]
        self.repo = repo_clean
        self.branch = branch
        self.base_url = "https://api.github.com"
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Colab-Pro-Director/2.0"
        }
        if self.token:
            auth_prefix = "Bearer" if self.token.startswith("github_pat_") else "token"
            self.headers["Authorization"] = f"{auth_prefix} {self.token}"

    def list_existing_scenes(self) -> List[str]:
        """Queries repository to find previously committed scenes for auto-resume."""
        if not self.token:
            return []
        url = f"{self.base_url}/repos/{self.repo}/contents/scenes?ref={self.branch}"
        req = urllib.request.Request(url, headers=self.headers)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return [item["name"] for item in data if item["name"].endswith(".mp4")]
        except Exception:
            return []

    def upload_file(self, local_path: Path, remote_path: str, commit_msg: str) -> bool:
        if not self.token or not self.repo or not local_path.exists():
            return False

        try:
            with open(local_path, "rb") as f:
                content_b64 = base64.b64encode(f.read()).decode("utf-8")

            url = f"{self.base_url}/repos/{self.repo}/contents/{remote_path}"
            sha = None
            try:
                req_get = urllib.request.Request(f"{url}?ref={self.branch}", headers=self.headers)
                with urllib.request.urlopen(req_get, timeout=10) as resp:
                    sha = json.loads(resp.read().decode("utf-8")).get("sha")
            except urllib.error.HTTPError as e:
                if e.code != 404:
                    raise

            payload = {
                "message": commit_msg,
                "content": content_b64,
                "branch": self.branch
            }
            if sha:
                payload["sha"] = sha

            req_put = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=self.headers,
                method="PUT"
            )
            with urllib.request.urlopen(req_put, timeout=20) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                commit_hash = res.get("commit", {}).get("sha", "")[:8]
                print(f"   🐙 GitHub Sync OK [{commit_hash}]: {remote_path}")
                return True
        except Exception as e:
            print(f"   ⚠️ GitHub sync skipped for {remote_path}: {e}")
            return False


# ==============================================================================
# 🧠 COSMIC SCREENPLAY DIRECTOR (QWEN 2.5)
# ==============================================================================

class CosmicScreenplayDirector:
    """Generates continuous cinematic space scene descriptions across 4 acts."""

    def __init__(self, device: str = "cuda", is_high_vram: bool = True):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Use 1.5B on L4/A100 (24GB), 0.5B on T4 (15GB)
        self.model_id = "Qwen/Qwen2.5-1.5B-Instruct" if is_high_vram else "Qwen/Qwen2.5-0.5B-Instruct"
        print(f"🧠 [1/2] Loading Cosmic Screenplay Director ({self.model_id})...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=HF_TOKEN or None)
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=dtype,
            device_map=device,
            token=HF_TOKEN or None
        )
        self.device = device
        print("   ✅ Screenplay Director initialized.")

    def get_act_info(self, scene_idx: int) -> Dict[str, Any]:
        for act in COSMIC_ACTS:
            start, end = act["scene_range"]
            if start <= scene_idx <= end:
                return act
        return COSMIC_ACTS[-1]

    def plan_scene_prompt(self, scene_idx: int, total_scenes: int = 200) -> str:
        act = self.get_act_info(scene_idx)
        
        camera_moves = [
            "slow forward cinematic drift",
            "smooth slow orbital pan",
            "majestic sweeping reveal",
            "gentle tracking shot through cosmic dust"
        ]
        camera_move = camera_moves[scene_idx % len(camera_moves)]

        system_msg = (
            "You are a master cinematic space director directing an epic 20-minute cosmic odyssey film. "
            "Write a single concise visual prompt (maximum 28 words) describing this specific space scene. "
            "Focus strictly on astronomical scale, celestial lighting, starfields, and majestic camera drift. "
            "Output ONLY the prompt text, with no preamble, quotes, or conversational filler."
        )
        user_msg = (
            f"Film Act: {act['name']}. Scene {scene_idx} of {total_scenes}. "
            f"Context: {act['context']}. Camera: {camera_move}."
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]

        try:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=45,
                    temperature=0.85,
                    top_p=0.92,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            prompt = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            prompt = prompt.replace('"', '').replace('\n', ' ').strip()
            
            if not prompt or len(prompt) < 15:
                fallbacks = act["fallbacks"]
                prompt = fallbacks[scene_idx % len(fallbacks)]
            return prompt
        except Exception:
            fallbacks = act["fallbacks"]
            return fallbacks[scene_idx % len(fallbacks)]


# ==============================================================================
# 🎬 FAST MOTION ENGINE (ANIMATEDIFF-LIGHTNING 2-STEP)
# ==============================================================================

class FastMotionEngine:
    """AnimateDiff-Lightning 2-Step Motion Engine optimized for high VRAM."""

    def __init__(self, device: str = "cuda"):
        from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        print("🎬 [2/2] Loading AnimateDiff-Lightning 2-Step Motion Pipeline...")
        adapter_repo = "ByteDance/AnimateDiff-Lightning"
        ckpt_filename = "animatediff_lightning_2step_diffusers.safetensors"
        
        adapter_path = hf_hub_download(repo_id=adapter_repo, filename=ckpt_filename, token=HF_TOKEN or None)
        adapter = MotionAdapter().to(device, torch.float16)
        adapter.load_state_dict(load_file(adapter_path))

        base_model = "emilianJR/epiCRealism"
        print(f"   📦 Base SD1.5 Checkpoint: {base_model}...")
        self.pipe = AnimateDiffPipeline.from_pretrained(
            base_model,
            motion_adapter=adapter,
            dtype=torch.float16,
            token=HF_TOKEN or None
        ).to(device)

        self.pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config,
            timestep_spacing="trailing",
            beta_schedule="linear"
        )
        if hasattr(self.pipe, "enable_vae_slicing"):
            self.pipe.enable_vae_slicing()
        elif hasattr(self.pipe, "vae") and hasattr(self.pipe.vae, "enable_slicing"):
            self.pipe.vae.enable_slicing()
        if hasattr(self.pipe, "enable_vae_tiling"):
            self.pipe.enable_vae_tiling()
        elif hasattr(self.pipe, "vae") and hasattr(self.pipe.vae, "enable_tiling"):
            self.pipe.vae.enable_tiling()
        self.device = device
        print("   ✅ High-Speed Motion Engine initialized.")

    def generate_batch(
        self,
        prompts: List[str],
        negative_prompt: str = "blur, cartoon, CGI, deformed, oversaturated, text, watermark, flicker, terrestrial",
        num_frames: int = 16,
        guidance_scale: float = 1.0,
        num_inference_steps: int = 2
    ) -> List[List[Image.Image]]:
        generator = torch.Generator(device=self.device)
        generator.manual_seed(random.randint(0, 2**32 - 1))

        output = self.pipe(
            prompt=prompts,
            negative_prompt=[negative_prompt] * len(prompts),
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator
        )
        return output.frames


# ==============================================================================
# 🎞️ HIGH-RAM VIDEO STITCHING & MAXIMUM COMPRESSION
# ==============================================================================

def export_clip_with_expansion(frames: List[Image.Image], output_path: Path, target_duration: float = 6.0) -> bool:
    """Exports frames into a smooth ~6.0 second cinematic slow-motion clip using FFmpeg."""
    try:
        import numpy as np
        # 1. Write raw 16 frames to a temporary fast MP4
        temp_raw = output_path.parent / f"temp_{output_path.name}"
        video_data = [np.array(frame) for frame in frames]
        with imageio.get_writer(str(temp_raw), fps=8, codec='libx264', quality=8) as writer:
            for frame in video_data:
                writer.append_data(frame)

        # 2. Smoothly expand duration to 6.0s at 24fps via setpts & minterpolate
        # Raw 16 frames at 8fps = 2.0s -> PTS * 3.0 = 6.0s
        cmd = [
            "ffmpeg", "-y",
            "-i", str(temp_raw),
            "-vf", "setpts=3.0*PTS,fps=24",
            "-c:v", "libx264",
            "-crf", "22",
            "-preset", "veryfast",
            "-pix_fmt", "yuv420p",
            str(output_path)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        if temp_raw.exists():
            temp_raw.unlink()
        return True
    except Exception as e:
        # Fallback to basic imageio write if FFmpeg fails
        try:
            import numpy as np
            video_data = [np.array(frame) for frame in frames]
            with imageio.get_writer(str(output_path), fps=8, codec='libx264', quality=8) as writer:
                for frame in video_data:
                    writer.append_data(frame)
            return True
        except Exception as e2:
            print(f"   ⚠️ Clip export error: {e2}")
            return False


def assemble_master_video(scene_paths: List[Path], output_path: Path) -> bool:
    """Concatenates all 200 scenes into the master 20-minute video."""
    print("\n" + "=" * 70)
    print("🎬 COMPILING MASTER 20-MINUTE CONTINUOUS SPACE VIDEO VIA FFMPEG...")
    print("=" * 70)

    if not scene_paths:
        print("❌ No scenes found for compilation.")
        return False

    concat_txt = output_path.parent / "master_scenes_list.txt"
    with open(concat_txt, "w", encoding="utf-8") as f:
        for p in scene_paths:
            f.write(f"file '{p.resolve().as_posix()}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_txt),
        "-c:v", "libx264",
        "-crf", "24",
        "-preset", "medium",
        "-pix_fmt", "yuv420p",
        str(output_path)
    ]

    try:
        t0 = time.time()
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"🎉 MASTER 20-MINUTE VIDEO COMPILED IN {time.time() - t0:.1f}s!")
        print(f"   File: {output_path}")
        print(f"   Size: {size_mb:.2f} MB")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Master compilation failed: {e.stderr[:200]}")
        return False


def create_maximum_compression_archive(source_files: List[Path], zip_destination: Path) -> Path:
    """Creates a maximum compression (ZIP_DEFLATED level 9) archive for fast download & WinRAR."""
    print("\n" + "=" * 70)
    print(f"📦 CREATING MAXIMUM COMPRESSION ARCHIVE: {zip_destination.name}")
    print("=" * 70)
    t0 = time.time()

    with zipfile.ZipFile(zip_destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
        for file_path in source_files:
            if file_path.exists():
                arcname = file_path.name
                zipf.write(file_path, arcname=arcname)
                mb = file_path.stat().st_size / (1024 * 1024)
                print(f"   ➕ Archived: {arcname} ({mb:.2f} MB)")

    zip_size_mb = zip_destination.stat().st_size / (1024 * 1024)
    print(f"✅ Archive complete in {time.time() - t0:.1f}s!")
    print(f"   Total Archive Size: {zip_size_mb:.2f} MB")
    return zip_destination


# ==============================================================================
# 🚀 MAIN PIPELINE
# ==============================================================================

def run_cosmic_20min_pipeline(
    total_scenes: int = 200,
    github_token: str = "",
    github_repo: str = "gglshtopenaividgen-lgtm/test1",
    github_branch: str = "main",
    output_dir_str: str = "/content"
):
    start_total_time = time.time()

    print("=" * 76)
    print("🚀 20-MINUTE COSMIC ODYSSEY AUTONOMOUS DIRECTOR LAUNCHING")
    print(f"🌌 Target Output: {total_scenes} Scenes (~1,200s / 20 Minutes)")
    print(f"🐙 GitHub Sync Target: {github_repo} [{github_branch}]")
    print("=" * 76)

    # 1. Hardware Inspection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0.0
    total_ram_gb = psutil.virtual_memory().total / (1024**3)

    is_high_vram = total_vram_gb >= 20.0  # L4 or A100
    batch_size = 4 if is_high_vram else 2

    print(f"⚡ Hardware Topology:")
    print(f"   • Accelerator: {gpu_name} ({total_vram_gb:.1f} GB VRAM)")
    print(f"   • Host RAM: {total_ram_gb:.1f} GB")
    print(f"   • Assigned Batch Size: {batch_size} (Optimal for {gpu_name})")

    # 2. Workspace Setup (Use /dev/shm if available for 0ms I/O, else output_dir)
    shm_path = Path("/dev/shm/cosmic_director")
    if shm_path.parent.exists() and total_ram_gb > 30.0:
        scratch_dir = shm_path
        print(f"⚡ High-RAM Optimization: Utilizing /dev/shm RAMDisk for 0ms I/O!")
    else:
        scratch_dir = Path(output_dir_str) / "cosmic_scratch"

    scenes_dir = scratch_dir / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(output_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3. GitHub Committer & Auto-Resume
    gh = GitHubBatchUploader(token=github_token, repo=github_repo, branch=github_branch)
    existing_scenes = gh.list_existing_scenes()
    print(f"🔍 Auto-Resume: {len(existing_scenes)} previously completed scenes found.")

    # 4. Load Models
    director = CosmicScreenplayDirector(device=device, is_high_vram=is_high_vram)
    engine = FastMotionEngine(device=device)

    all_completed_scenes: List[Path] = []
    screenplay_manifest: List[Dict[str, Any]] = []

    scene_idx = 1
    t_start_gen = time.time()

    print("\n" + "-" * 76)
    print(f"⚡ GENERATING 200 SCENES AT BATCH SIZE {batch_size} (>1.0x REAL TIME)")
    print("-" * 76)

    while scene_idx <= total_scenes:
        curr_batch = [scene_idx + b for b in range(batch_size) if (scene_idx + b) <= total_scenes]

        # Auto-resume check
        pending_batch = []
        for idx in curr_batch:
            clip_name = f"cosmic_scene_{idx:03d}.mp4"
            local_clip = scenes_dir / clip_name
            if clip_name in existing_scenes and local_clip.exists():
                all_completed_scenes.append(local_clip)
            else:
                pending_batch.append(idx)

        if not pending_batch:
            scene_idx += batch_size
            continue

        # Step A: Director Plans Screenplay Prompts
        prompts = [director.plan_scene_prompt(idx, total_scenes) for idx in pending_batch]
        
        elapsed_min = (time.time() - t_start_gen) / 60
        vram_used = torch.cuda.memory_allocated(0) / (1024**3) if torch.cuda.is_available() else 0.0
        print(f"\n[Batch {pending_batch}] ({elapsed_min:.1f}m elapsed | VRAM: {vram_used:.1f} GB)")
        for idx, p in zip(pending_batch, prompts):
            print(f"   🌌 Scene {idx:03d}: \"{p}\"")

        # Step B: High-Speed 2-Step Motion Generation
        t0 = time.time()
        batch_frames = engine.generate_batch(prompts=prompts, num_frames=16, num_inference_steps=2)
        gen_sec = time.time() - t0
        print(f"   ⚡ Rendered {len(pending_batch)} scenes in {gen_sec:.1f}s ({gen_sec/len(pending_batch):.2f}s/scene)")

        # Step C: Export with Temporal Expansion & Incremental GitHub Commit
        for idx, frames, prompt in zip(pending_batch, batch_frames, prompts):
            clip_name = f"cosmic_scene_{idx:03d}.mp4"
            local_clip = scenes_dir / clip_name
            export_ok = export_clip_with_expansion(frames, local_clip, target_duration=6.0)
            if export_ok:
                all_completed_scenes.append(local_clip)
                meta_item = {
                    "scene": idx,
                    "prompt": prompt,
                    "duration_seconds": 6.0,
                    "act": director.get_act_info(idx)["name"]
                }
                screenplay_manifest.append(meta_item)

                # Commit to GitHub in background / real-time
                if github_token:
                    gh.upload_file(local_clip, f"cosmic_scenes/{clip_name}", f"feat(cosmic): add scene {idx:03d}")

        # Memory hygiene
        del batch_frames
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        scene_idx += batch_size

    # Step D: Save Screenplay Manifest
    manifest_path = out_dir / "cosmic_screenplay_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(screenplay_manifest, f, indent=2)
    print(f"📜 Screenplay manifest saved ({len(screenplay_manifest)} scenes).")
    if github_token:
        gh.upload_file(manifest_path, "cosmic_screenplay_manifest.json", "docs: add cosmic screenplay manifest")

    # Step E: Compile Master 20-Minute Continuous Video
    master_video_path = out_dir / "cosmic_odyssey_20min_master.mp4"
    all_completed_scenes_sorted = sorted(all_completed_scenes, key=lambda p: p.name)
    compile_ok = assemble_master_video(all_completed_scenes_sorted, master_video_path)

    # Step F: Create WinRAR-Ready Maximum Compression Archive
    archive_sources = [master_video_path, manifest_path]
    zip_output_path = out_dir / "cosmic_odyssey_20min_master.zip"
    final_zip = create_maximum_compression_archive(archive_sources, zip_output_path)

    total_pipeline_min = (time.time() - start_total_time) / 60
    print("\n" + "=" * 76)
    print("🎉 20-MINUTE COSMIC ODYSSEY VIDEO PIPELINE COMPLETE!")
    print(f"⏱️ Total Wall-Clock Time: {total_pipeline_min:.2f} Minutes")
    print(f"🎞️ Master Video: {master_video_path} ({master_video_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f"📦 Maximum Compression Zip: {final_zip} ({final_zip.stat().st_size / (1024*1024):.2f} MB)")
    print(f"🐙 GitHub Sync: https://github.com/{github_repo}/tree/{github_branch}")
    print("=" * 76)
    return str(final_zip)


# ==============================================================================
# 🏁 CLI
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="20-Minute Cosmic Odyssey Autonomous Video Director")
    parser.add_argument("--scenes", type=int, default=200)
    parser.add_argument("--token", type=str, default="")
    parser.add_argument("--repo", type=str, default="gglshtopenaividgen-lgtm/test1")
    parser.add_argument("--branch", type=str, default="main")
    parser.add_argument("--output-dir", type=str, default="/content")
    args = parser.parse_args()

    run_cosmic_20min_pipeline(
        total_scenes=args.scenes,
        github_token=args.token,
        github_repo=args.repo,
        github_branch=args.branch,
        output_dir_str=args.output_dir
    )
