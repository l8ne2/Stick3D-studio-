import asyncio
import base64
import io
import json
import math
import os
import re
import subprocess
import tempfile
import threading
import time
import urllib.parse
import uuid
import wave
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import edge_tts
import imageio_ffmpeg
import numpy as np
import requests
from deep_translator import MyMemoryTranslator
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from gradio_client import Client
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = BASE_DIR.parent
PROJECTS_DIR = BASE_DIR / "projects"
STATIC_DIR = BASE_DIR / "static"
KEYS_PATH = BASE_DIR / "keys.json"
CUSTOM_REF_PATH = BASE_DIR / "custom_style_ref.jpg"
DEFAULT_REF_PATH = STATIC_DIR / "style_ref_default.jpg"

PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
W, H = 1080, 1920
FPS = 30
SR = 44100

VOICES = [
    {"id": "fr-FR-RemyMultilingualNeural", "label": "Rémy (FR) — Narrateur Shorts dynamique"},
    {"id": "fr-FR-HenriNeural", "label": "Henri (FR) — Voix posée & cinéma"},
    {"id": "fr-FR-VivienneMultilingualNeural", "label": "Vivienne (FR) — Narratrice expressive"},
    {"id": "fr-FR-DeniseNeural", "label": "Denise (FR) — Claire & rythmée"},
    {"id": "fr-CA-AntoineNeural", "label": "Antoine (QC) — Conteur chaleureux"},
]

JOBS: Dict[str, dict] = {}
TRANSLATE_CACHE: Dict[str, str] = {}

app = FastAPI(title="Stick3D Shorts Studio — Qualité Ultra HD Gratuite")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_keys() -> dict:
    if KEYS_PATH.exists():
        try:
            return json.loads(KEYS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"gemini_key": "", "hf_token": ""}


def save_keys(data: dict):
    KEYS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_active_ref_path() -> Optional[Path]:
    if CUSTOM_REF_PATH.exists():
        return CUSTOM_REF_PATH
    if DEFAULT_REF_PATH.exists():
        return DEFAULT_REF_PATH
    return None


def get_bold_font(size: int = 56):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/calibrib.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


# --------------------- AUTOMATIC FR -> EN TRANSLATOR & MASTER PROMPT ENGINE ---------------------
def translate_to_en(text: str) -> str:
    clean = re.sub(r"\s+", " ", (text or "")).strip()
    if not clean:
        return ""
    if clean in TRANSLATE_CACHE:
        return TRANSLATE_CACHE[clean]
    # Check if text looks French (accents or common French words)
    looks_fr = bool(re.search(r"[éèêëàâîïôùûçœ]|(\b(le|la|les|un|une|des|dans|avec|sur|pour|il|elle|son|sa|est|sont)\b)", clean.lower()))
    if not looks_fr:
        return clean
    try:
        en = MyMemoryTranslator(source="fr-FR", target="en-GB").translate(clean[:490])
        if en and "MYMEMORY WARNING" not in en.upper():
            TRANSLATE_CACHE[clean] = en
            return en
    except Exception:
        pass
    return clean


def build_master_visual_prompt(raw_prompt: str, style_key: str, character_sheet: str = "", custom_style: str = "") -> str:
    action_en = translate_to_en(raw_prompt)
    char_en = translate_to_en(character_sheet) if character_sheet.strip() else ""
    char_clause = f"RECURRING MAIN CHARACTER DESIGN: {char_en}. " if char_en else ""

    if style_key == "stickman_west_3d":
        return (
            "Vertical 9:16 cinema frame, single full-bleed shot, no subtitles, no UI text. "
            "Who Framed Roger Rabbit mixed-media style: flat 2D hand-drawn cartoon stickman character composited seamlessly into a raw photorealistic 3D Unreal Engine 5 Wild West movie set. "
            "2D STICKMAN ANATOMY: minimal Henry-Stickmin style character with a smooth round pure-white ball head, thick bold black marker outlines, "
            "only two expressive black cartoon eyes, thick black arched eyebrows, and an exaggerated comic mouth/toothy grin on the plain white sphere face, "
            "thin solid-black rubber-hose stick-line arms and solid-black stick-line legs, chunky white cartoon hands with bold black outlines, simple flat cel-shaded short-sleeve colored clothing. "
            f"{char_clause}"
            "PHOTOREALISTIC 3D ENVIRONMENT & PROPS: gritty 3D Wild West textures (weathered timber beams, rough stone, rusty iron, warm glowing oil lanterns, volumetric dust rays); "
            "hats, helmets, tools, and props held by the 2D stickman have realistic 3D materials, with warm golden rim-lighting and real cast shadows grounding the 2D character in the 3D scene. "
            f"SCENE ACTION: {action_en}"
        )
    elif style_key == "stickman_modern_3d":
        return (
            "Vertical 9:16 cinema frame, single full-bleed shot, no subtitles, no UI text. "
            "Mixed-media 2D-in-3D style: flat 2D hand-drawn cartoon stickman character composited into a photorealistic 3D Unreal Engine 5 modern environment. "
            "2D STICKMAN ANATOMY: smooth round pure-white ball head, thick bold black ink outlines, expressive cartoon eyes, eyebrows, and big comic mouth on plain white circle face, "
            "thin solid-black stick-line arms and solid-black stick-line legs, chunky white cartoon hands with black outlines, simple flat cel-shaded short-sleeve outfit. "
            f"{char_clause}"
            "PHOTOREALISTIC 3D WORLD: realistic 3D architecture, lighting, depth of field, realistic 3D props in the 2D hands, dramatic rim-light and cast shadows. "
            f"SCENE ACTION: {action_en}"
        )
    elif style_key == "pixar_3d":
        return f"Vertical 9:16 3D animated feature film frame, expressive stylized 3D characters, rich Octane Render textures, warm rim lighting, shallow depth of field. {char_clause}SCENE: {action_en}"
    elif style_key == "comic_noir":
        return f"Vertical 9:16 bold graphic novel frame, heavy black ink outlines, dramatic cinematic lighting, vivid colors. {char_clause}SCENE: {action_en}"
    else:
        prefix = translate_to_en(custom_style)
        return f"Vertical 9:16 portrait frame. {prefix} {char_clause}SCENE: {action_en}"


# --------------------- MULTI-ENGINE FREE ULTRA-HD IMAGE GENERATOR ---------------------
def generate_with_gemini_free(master_prompt: str, api_key: str, out_path: Path) -> Optional[str]:
    """Uses free Google AI Studio key (gemini-2.5-flash-image) + exact style reference image-1.png!"""
    ref_path = get_active_ref_path()
    parts = []
    if ref_path and ref_path.exists():
        b64_ref = base64.b64encode(ref_path.read_bytes()).decode("ascii")
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": b64_ref}})
        parts.append({
            "text": (
                "Use the attached reference image ONLY as the exact visual style guide (match the exact 2D white-circle-head stickman anatomy, "
                "thick black ink outlines, solid-black stick arms/legs, white cartoon hands, and photorealistic 3D movie-set background lighting and props). "
                f"Generate a brand-new vertical 9:16 scene for: {master_prompt}"
            )
        })
    else:
        parts.append({"text": master_prompt})

    for model_name in ["gemini-2.5-flash-image", "gemini-2.0-flash-exp-image-generation"]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"] if "2.0" in model_name else ["IMAGE"],
            },
        }
        if "2.5" in model_name:
            payload["generationConfig"]["imageConfig"] = {"aspectRatio": "9:16"}

        try:
            r = requests.post(url, params={"key": api_key}, json=payload, timeout=45)
            if r.status_code == 200:
                data = r.json()
                for cand in data.get("candidates", []):
                    for p in cand.get("content", {}).get("parts", []):
                        inline = p.get("inlineData") or p.get("inline_data")
                        if inline and inline.get("data"):
                            raw_bytes = base64.b64decode(inline["data"])
                            im = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                            im = im.resize((768, 1365), Image.Resampling.LANCZOS)
                            im.save(out_path, format="JPEG", quality=92)
                            return f"Google {model_name} (Style Ref)"
        except Exception:
            pass
    return None


def generate_with_hf_flux(master_prompt: str, hf_token: str, out_path: Path, seed: int) -> Optional[str]:
    """Uses 12B FLUX.1-dev / FLUX.1-merged / FLUX.1-schnell on Hugging Face (works with or without free token)."""
    # 1) If user provided a free HF token, also try fast HF Serverless Inference Router first or ZeroGPU with auth!
    if hf_token:
        try:
            r = requests.post(
                "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell",
                headers={"Authorization": f"Bearer {hf_token}"},
                json={"inputs": master_prompt, "parameters": {"width": 768, "height": 1344, "num_inference_steps": 4}},
                timeout=35,
            )
            if r.status_code == 200 and len(r.content) > 8000 and "image" in r.headers.get("content-type", ""):
                im = Image.open(io.BytesIO(r.content)).convert("RGB")
                im.save(out_path, format="JPEG", quality=92)
                return "FLUX.1-schnell (HF Router)"
        except Exception:
            pass

        try:
            c_abid = Client("abidlabs/black-forest-labs-FLUX.1-schnell", token=hf_token, verbose=False)
            res_abid = c_abid.predict(param_0=master_prompt, api_name="/predict")
            p_abid = res_abid["path"] if isinstance(res_abid, dict) else res_abid
            im = Image.open(p_abid).convert("RGB").resize((768, 1365), Image.Resampling.LANCZOS)
            im.save(out_path, format="JPEG", quality=92)
            return "FLUX.1-schnell (HF Serverless)"
        except Exception:
            pass

    spaces = [("multimodalart/FLUX.1-merged", 8), ("black-forest-labs/FLUX.1-schnell", 4)]
    if hf_token:
        spaces.insert(0, ("black-forest-labs/FLUX.1-dev", 22))

    for space_id, steps in spaces:
        try:
            client = Client(space_id, token=hf_token if hf_token else None, verbose=False)
            if "schnell" in space_id:
                res = client.predict(
                    prompt=master_prompt,
                    seed=seed % 2147483647,
                    randomize_seed=False,
                    width=768,
                    height=1344,
                    num_inference_steps=steps,
                    api_name="/infer",
                )
            else:
                res = client.predict(
                    prompt=master_prompt,
                    seed=seed % 2147483647,
                    randomize_seed=False,
                    width=768,
                    height=1344,
                    guidance_scale=3.5,
                    num_inference_steps=steps,
                    api_name="/infer",
                )
            out_file = res[0] if isinstance(res, (list, tuple)) else res
            if isinstance(out_file, dict):
                out_file = out_file.get("path")
            im = Image.open(out_file).convert("RGB")
            im.save(out_path, format="JPEG", quality=92)
            return space_id.split("/")[-1]
        except Exception:
            continue
    return None


def generate_with_horde_sdxl(master_prompt: str, out_path: Path) -> Optional[str]:
    """Unlimited anonymous free community GPU cluster running AlbedoBase XL (SDXL)."""
    try:
        horde_headers = {"apikey": "0000000000", "Client-Agent": "Stick3DStudio:2.0:free", "Content-Type": "application/json"}
        payload = {
            "prompt": master_prompt[:900],
            "models": ["AlbedoBase XL (SDXL)", "SDXL 1.0"],
            "params": {"width": 576, "height": 1024, "steps": 24, "cfg_scale": 6.5, "n": 1},
            "nsfw": False,
            "r2": True,
        }
        sub = requests.post("https://aihorde.net/api/v2/generate/async", json=payload, headers=horde_headers, timeout=15)
        if sub.status_code in (200, 202):
            req_id = sub.json().get("id")
            for _ in range(26):
                time.sleep(2.5)
                st = requests.get(f"https://aihorde.net/api/v2/generate/status/{req_id}", headers=horde_headers, timeout=12)
                if st.status_code == 200:
                    data = st.json()
                    if data.get("done") and data.get("generations"):
                        img_field = data["generations"][0]["img"]
                        if img_field.startswith("http"):
                            ir = requests.get(img_field, timeout=15)
                            im = Image.open(io.BytesIO(ir.content)).convert("RGB")
                        else:
                            im = Image.open(io.BytesIO(base64.b64decode(img_field))).convert("RGB")
                        im = im.resize((768, 1365), Image.Resampling.LANCZOS)
                        im.save(out_path, format="JPEG", quality=91)
                        return "AI-Horde AlbedoBase-XL (Gratuit)"
    except Exception:
        pass
    return None


def generate_free_ai_image(raw_prompt: str, style_key: str, character_sheet: str, custom_style: str, out_path: Path, seed: Optional[int] = None) -> str:
    if seed is None:
        seed = int(time.time() * 1000) % 999999

    keys = load_keys()
    gemini_key = (keys.get("gemini_key") or os.environ.get("GEMINI_API_KEY") or "").strip()
    hf_token = (keys.get("hf_token") or os.environ.get("HF_TOKEN") or "").strip()

    master_prompt = build_master_visual_prompt(raw_prompt, style_key, character_sheet, custom_style)

    # 1) Free Google AI Studio Gemini 2.5 Flash Image + exact style reference image-1.png!
    if gemini_key:
        prov = generate_with_gemini_free(master_prompt, gemini_key, out_path)
        if prov:
            return prov

    # 2) 12B FLUX.1 on Hugging Face (FLUX.1-dev / FLUX.1-merged / FLUX.1-schnell)
    prov = generate_with_hf_flux(master_prompt, hf_token, out_path, seed)
    if prov:
        return prov

    raise RuntimeError(
        "Le quota public anonyme de cette adresse IP partagée est atteint sur FLUX.1. "
        "Pour générer en qualité identique à votre image modèle (0€, sans carte bancaire), cliquez sur « ⚡ Boost HD Gratuit » en haut et collez "
        "soit une Clé Google AI Studio gratuite (aistudio.google.com/apikey), soit un Token Hugging Face gratuit (huggingface.co/settings/tokens) !"
    )


# --------------------- DEMO PROJECT BOOTSTRAP ---------------------
def ensure_demo_project():
    demo_dir = PROJECTS_DIR / "demo-mine-dor"
    meta_path = demo_dir / "project.json"
    demo_dir.mkdir(parents=True, exist_ok=True)
    kf_dir = WORKSPACE_DIR / "keyframes"
    demo_video = WORKSPACE_DIR / "film_complet.mp4"

    if meta_path.exists():
        p = json.loads(meta_path.read_text(encoding="utf-8"))
        if "character_sheet" not in p:
            p["character_sheet"] = "Mineur stickman en chemise rouge à manches courtes, bretelles noires et casque de mineur 3D en cuir usé avec lampe frontale allumée"
            meta_path.write_text(json.dumps(p, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    shots = [
        {
            "id": "s1",
            "narration": "Il a chargé son fusil avec de l'or et a tiré dans les murs de la mine. Son pote regarde l'or disparaître dans la roche.",
            "prompt": "Dans le tunnel sombre d'une mine d'or éclairé par une lanterne suspendue, le mineur tire une gerbe étincelante de poussière d'or contre la paroi rocheuse pendant que son ami au bonnet jaune et pull vert écarquille les yeux sur son tabouret.",
            "camera": "zoom_in",
            "sfx": "gunshot",
            "image": "/api/demo-asset/scene1_02_tir_or.png" if (kf_dir / "scene1_02_tir_or.png").exists() else None,
        },
        {
            "id": "s2",
            "narration": "Alors on tire de l'or dans une mine d'or ? Tu vois, ça s'appelle saler une mine.",
            "prompt": "Gros plan rapproché : le mineur pose son fusil fumant sur l'épaule devant la paroi rocheuse incrustée d'or brillant et fait un grand clin d'œil malicieux pendant que son ami au bonnet jaune reste bouche bée derrière lui.",
            "camera": "crash_zoom",
            "sfx": "magic",
            "image": "/api/demo-asset/scene1_03_clin_doeil.png" if (kf_dir / "scene1_03_clin_doeil.png").exists() else None,
        },
        {
            "id": "s3",
            "narration": "Il a creusé pendant 10 ans sans rien trouver d'autre que de la pierre. Et demain, un homme riche vient l'acheter. Alors il a dépensé ses 100 derniers dollars en poussière d'or.",
            "prompt": "Au soleil devant l'entrée en bois de la mine, le mineur brandit avec enthousiasme un journal titré MILLIONAIRE DUE TOMORROW à côté de son ami au bonnet jaune tout sourire.",
            "camera": "pan_right",
            "sfx": "kaching",
            "image": "/api/demo-asset/scene2_02_journal.png" if (kf_dir / "scene2_02_journal.png").exists() else None,
        },
        {
            "id": "s4",
            "narration": "Le lendemain matin, le riche débarque en costume blanc et se cogne la tête sur la première poutre. Mais il a amené un expert, et l'expert n'est pas stupide.",
            "prompt": "Devant la mine marquée SURE THING MINE et une voiture ancienne blanche, un grand riche moustachu en costume blanc et haut-de-forme blanc se cogne la tête contre la poutre d'entrée à côté d'un petit géologue en veste violette et monocle.",
            "camera": "shake_boom",
            "sfx": "bonk",
            "image": "/api/demo-asset/scene3_01_arrivee_choc.png" if (kf_dir / "scene3_01_arrivee_choc.png").exists() else None,
        },
        {
            "id": "s5",
            "narration": "Il casse un morceau du mur, le rince dans une batée, et des paillettes d'or tourbillonnent au fond. Mais sous sa loupe, chaque paillette est plate. Quelqu'un a tiré cet or dans le mur.",
            "prompt": "Très gros plan macro à travers une loupe en laiton montrant une paillette d'or complètement plate et écrasée sur la pointe d'une lame de couteau, tenue par le géologue à monocle.",
            "camera": "crash_zoom",
            "sfx": "suspense",
            "image": "/api/demo-asset/scene4_03_loupe_macro.png" if (kf_dir / "scene4_03_loupe_macro.png").exists() else None,
        },
        {
            "id": "s6",
            "narration": "Mais le riche éclate de rire : Pourquoi quelqu'un tirerait de l'or dans un mur ? C'est complètement fou ! Et son pote hoche la tête : Complètement fou. Alors le riche paie.",
            "prompt": "Le mineur et son ami au bonnet jaune s'enfuient à toute vitesse sur les rails vers la sortie ensoleillée avec une mallette débordant de billets verts pendant que le riche en costume blanc salue derrière eux.",
            "camera": "zoom_in",
            "sfx": "laugh",
            "image": "/api/demo-asset/scene5_03_mallette_fuite.png" if (kf_dir / "scene5_03_mallette_fuite.png").exists() else None,
        },
        {
            "id": "s7",
            "narration": "Et pendant que les deux filent avec l'argent, il reste derrière, allume un bâton de dynamite de son costume, le jette comme un pétard et court. Et quand la poussière retombe, il y a un filon de vrai or épais comme le bras, trois mètres plus loin que là où l'autre a abandonné.",
            "prompt": "Dans le tunnel de la mine, au premier plan un mètre ruban jaune posé au sol à côté du casque abandonné du mineur et d'une pioche mène deux mètres plus loin à un immense filon d'or pur brillant dans la roche devant le riche qui danse de joie.",
            "camera": "pull_back",
            "sfx": "explosion",
            "image": "/api/demo-asset/scene6_04_metre_ruban_chapeau.png" if (kf_dir / "scene6_04_metre_ruban_chapeau.png").exists() else None,
        },
    ]

    project = {
        "id": "demo-mine-dor",
        "title": "Démo : La Mine d'Or Salée (Complet)",
        "style": "stickman_west_3d",
        "character_sheet": "Mineur stickman en chemise rouge à manches courtes, bretelles noires et casque de mineur 3D en cuir usé avec lampe frontale allumée",
        "custom_style": "",
        "voice": "fr-FR-RemyMultilingualNeural",
        "speed": 1.06,
        "subtitles": True,
        "video_url": "/api/demo-video" if demo_video.exists() else None,
        "updated_at": int(time.time()),
        "shots": shots,
    }
    meta_path.write_text(json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8")


ensure_demo_project()


# --------------------- MODELS ---------------------
class KeysPayload(BaseModel):
    gemini_key: str = ""
    hf_token: str = ""


class SplitRequest(BaseModel):
    script: str
    style: str = "stickman_west_3d"


class ProjectSave(BaseModel):
    title: str
    style: str = "stickman_west_3d"
    character_sheet: str = ""
    custom_style: str = ""
    voice: str = "fr-FR-RemyMultilingualNeural"
    speed: float = 1.06
    subtitles: bool = True
    shots: List[dict]


class GenImageReq(BaseModel):
    prompt: str
    style: str = "stickman_west_3d"
    character_sheet: str = ""
    custom_style: str = ""
    seed: Optional[int] = None


class VoicePreviewReq(BaseModel):
    voice: str = "fr-FR-RemyMultilingualNeural"
    speed: float = 1.06
    text: str = "Il a chargé son fusil avec de l'or et a tiré dans les murs de la mine !"


# --------------------- SMART FRENCH SCRIPT SPLITTER ---------------------
def guess_sfx_and_camera(text: str, idx: int, total: int):
    low = text.lower()
    if any(w in low for w in ["explos", "dynamite", "pétard", "boum", "bombe", "décoll"]):
        return "shake_boom", "explosion"
    if any(w in low for w in ["tiré", "fusil", "pistolet", "balle", "coup de feu"]):
        return "shake_boom", "gunshot"
    if any(w in low for w in ["cogne", "choc", "poutre", "tombe", "écrase", "frappe", "clac"]):
        return "crash_zoom", "bonk"
    if any(w in low for w in ["dollar", "argent", "billet", "riche", "paie", "million", "achèt", "achet", "banque"]):
        return "pan_right", "kaching"
    if any(w in low for w in ["rire", "rit ", "rigole", "fou"]):
        return "crash_zoom", "laugh"
    if any(w in low for w in ["loupe", "étrange", "mystère", "secret", "soupçon", "plate"]):
        return "crash_zoom", "suspense"
    if any(w in low for w in ["or ", "d'or", "filon", "brill", "trésor", "magie", "diamant"]):
        return ("pull_back" if idx == total - 1 else "zoom_in"), "magic"
    cams = ["zoom_in", "pan_right", "crash_zoom", "pull_back"]
    return cams[idx % len(cams)], "none"


@app.post("/api/split-script")
def api_split_script(req: SplitRequest):
    raw = req.script.strip()
    if not raw:
        raise HTTPException(400, "Le script est vide.")

    sentences = [s.strip() for s in re.split(r"(?<=[.!?…])\s+|\n+", raw) if s.strip()]
    chunks: List[str] = []
    cur: List[str] = []
    cur_words = 0
    for s in sentences:
        w = len(s.split())
        if cur and (cur_words + w > 34 or len(cur) >= 3):
            chunks.append(" ".join(cur))
            cur = [s]
            cur_words = w
        else:
            cur.append(s)
            cur_words += w
    if cur:
        chunks.append(" ".join(cur))

    shots = []
    for idx, ch in enumerate(chunks[:12]):
        cam, sfx = guess_sfx_and_camera(ch, idx, len(chunks[:12]))
        shots.append(
            {
                "id": f"shot_{uuid.uuid4().hex[:6]}",
                "narration": ch,
                "prompt": ch,
                "camera": cam,
                "sfx": sfx,
                "image": None,
            }
        )
    return {"shots": shots}


# --------------------- EDGE-TTS VOICEOVER + WORD TIMESTAMPS ---------------------
async def synth_edge_tts(text: str, voice: str, speed: float, mp3_path: Path):
    pct = int(round((speed - 1.0) * 100))
    rate_str = f"{pct:+d}%"
    comm = edge_tts.Communicate(text, voice=voice, rate=rate_str)
    words = []
    with open(mp3_path, "wb") as f:
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                start_s = chunk["offset"] / 10_000_000.0
                dur_s = chunk["duration"] / 10_000_000.0
                words.append({"text": chunk["text"], "start": start_s, "end": start_s + dur_s})
    return words


# --------------------- PROCEDURAL SFX & RENDER ENGINE ---------------------
def render_camera(img, cx, cy, zoom, angle_deg=0.0):
    ih, iw = img.shape[:2]
    base_scale = max(W / iw, H / ih)
    scale = base_scale * zoom
    half_vw = (W / scale) * 0.5
    half_vh = (H / scale) * 0.5
    px = np.clip(cx * iw, half_vw * 0.92, iw - half_vw * 0.92) if iw > 2 * half_vw else iw * 0.5
    py = np.clip(cy * ih, half_vh * 0.92, ih - half_vh * 0.92) if ih > 2 * half_vh else ih * 0.5
    M = cv2.getRotationMatrix2D((float(px), float(py)), float(angle_deg), float(scale))
    M[0, 2] += (W * 0.5) - px
    M[1, 2] += (H * 0.5) - py
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def draw_subtitle_overlay(frame_bgr: np.ndarray, caption_text: str, font) -> np.ndarray:
    if not caption_text:
        return frame_bgr
    pil_im = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_im)
    words = caption_text.split()
    lines, cur = [], []
    for w in words:
        if cur and len(" ".join(cur + [w])) > 22:
            lines.append(" ".join(cur))
            cur = [w]
        else:
            cur.append(w)
    if cur:
        lines.append(" ".join(cur))
    lines = lines[:3]

    y_base = H - 340 - (len(lines) - 1) * 38
    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font, stroke_width=6)
        tw = bbox[2] - bbox[0]
        x = (W - tw) // 2
        y = y_base + i * 74
        draw.text(
            (x, y),
            line,
            font=font,
            fill=(255, 226, 70) if i == len(lines) - 1 else (255, 255, 255),
            stroke_width=6,
            stroke_fill=(0, 0, 0),
        )
    return cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)


def build_sfx_track(kind: str, n_samples: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    buf = np.zeros((n_samples, 2), dtype=np.float32)
    dur = n_samples / SR
    t_full = np.arange(n_samples, dtype=np.float32) / SR
    amb = 0.018 * np.sin(2 * np.pi * 65.0 * t_full)
    buf[:, 0] += amb
    buf[:, 1] += amb

    def add_mono(start_s, sig, gain=0.5):
        i0 = int(round(start_s * SR))
        if 0 <= i0 < n_samples:
            n = min(len(sig), n_samples - i0)
            buf[i0 : i0 + n, 0] += sig[:n] * gain
            buf[i0 : i0 + n, 1] += sig[:n] * gain

    def chime(freq, d=0.45):
        n = int(SR * d)
        t = np.arange(n, dtype=np.float32) / SR
        return ((np.sin(2 * np.pi * freq * t) + 0.4 * np.sin(4.02 * np.pi * freq * t)) * np.exp(-t * 6.5)).astype(np.float32)

    nw = min(n_samples, int(SR * 0.20))
    tw = np.linspace(0, 1, nw, dtype=np.float32)
    wh = np.convolve(rng.standard_normal(nw).astype(np.float32), np.ones(10, dtype=np.float32) / 10.0, mode="same")
    wh *= np.clip(np.sin(np.pi * tw), 0.0, 1.0) ** 1.6
    add_mono(0.0, wh, 0.25)

    hit_t = min(max(0.35, dur * 0.28), max(0.2, dur - 0.8))
    if kind == "gunshot":
        nb = int(SR * 0.75)
        tb = np.arange(nb, dtype=np.float32) / SR
        noise = rng.standard_normal(nb).astype(np.float32)
        boom = 0.7 * np.sin(2 * np.pi * (130 * np.exp(-tb * 7) + 40) * tb) * np.exp(-tb * 6) + 0.6 * noise * np.exp(-tb * 10)
        add_mono(hit_t, boom.astype(np.float32), 0.65)
    elif kind == "explosion":
        nb = int(SR * 1.8)
        tb = np.arange(nb, dtype=np.float32) / SR
        noise = np.convolve(rng.standard_normal(nb).astype(np.float32), np.ones(18, dtype=np.float32) / 18.0, mode="same")
        boom = 0.8 * np.sin(2 * np.pi * (95 * np.exp(-tb * 4) + 32) * tb) * np.exp(-tb * 2.5) + 0.7 * noise * np.exp(-tb * 2.0)
        add_mono(hit_t, boom.astype(np.float32), 0.75)
    elif kind == "bonk":
        nb = int(SR * 0.25)
        tb = np.arange(nb, dtype=np.float32) / SR
        bonk = np.sin(2 * np.pi * (380 * np.exp(-tb * 9) + 145) * tb) * np.exp(-tb * 14)
        add_mono(hit_t, bonk.astype(np.float32), 0.65)
        add_mono(hit_t + 0.22, chime(1318.5, 0.22), 0.22)
        add_mono(hit_t + 0.42, chime(1046.5, 0.25), 0.22)
    elif kind == "kaching":
        add_mono(hit_t, chime(1568.0, 0.14), 0.38)
        add_mono(hit_t + 0.11, chime(3136.0, 0.65), 0.45)
    elif kind == "magic":
        for i, f in enumerate([1046.5, 1318.5, 1568.0, 2093.0, 2637.0, 3136.0]):
            add_mono(hit_t + i * 0.11, chime(f, 0.45), 0.20)
    elif kind == "laugh":
        for i, f in enumerate([196.0, 185.0, 174.6, 164.8, 196.0]):
            add_mono(hit_t + i * 0.28, chime(f, 0.22), 0.30)
    elif kind == "suspense":
        add_mono(hit_t, chime(130.81, 0.65), 0.40)
        add_mono(hit_t + 0.38, chime(123.47, 0.95), 0.45)
    return buf


def render_full_project_job(job_id: str, project_id: str, gen_missing_only: bool = False):
    try:
        proj_dir = PROJECTS_DIR / project_id
        meta_path = proj_dir / "project.json"
        proj = json.loads(meta_path.read_text(encoding="utf-8"))
        shots = proj.get("shots", [])
        if not shots:
            raise RuntimeError("Aucun plan dans le projet.")

        style_key = proj.get("style", "stickman_west_3d")
        char_sheet = proj.get("character_sheet", "")
        custom_style = proj.get("custom_style", "")
        voice = proj.get("voice", "fr-FR-RemyMultilingualNeural")
        speed = float(proj.get("speed", 1.06))
        use_subs = bool(proj.get("subtitles", True))
        font = get_bold_font(56)

        total_steps = len(shots) * 2 + 2
        step = 0

        img_paths: List[Path] = []
        for idx, sh in enumerate(shots):
            step += 1
            JOBS[job_id] = {
                "status": "running",
                "progress": int(step * 100 / total_steps),
                "message": f"Génération image Ultra-HD du plan {idx + 1}/{len(shots)}...",
            }
            shot_img_path = proj_dir / f"{sh['id']}.jpg"
            existing_url = sh.get("image")

            resolved_existing = None
            if existing_url and existing_url.startswith("/api/demo-asset/"):
                cand = WORKSPACE_DIR / "keyframes" / existing_url.split("/api/demo-asset/")[-1]
                if cand.exists():
                    resolved_existing = cand
            elif shot_img_path.exists() and existing_url:
                resolved_existing = shot_img_path

            if resolved_existing and gen_missing_only:
                img_paths.append(resolved_existing)
            else:
                generate_free_ai_image(
                    sh.get("prompt") or sh.get("narration", ""),
                    style_key,
                    char_sheet,
                    custom_style,
                    shot_img_path,
                    seed=1000 + idx * 37 + int(time.time()) % 500,
                )
                sh["image"] = f"/api/projects/{project_id}/shot-image/{sh['id']}?t={int(time.time())}"
                img_paths.append(shot_img_path)
                meta_path.write_text(json.dumps(proj, ensure_ascii=False, indent=2), encoding="utf-8")

        shot_audio_mixes = []
        shot_durations = []
        shot_words_list = []

        for idx, sh in enumerate(shots):
            step += 1
            JOBS[job_id] = {
                "status": "running",
                "progress": int(step * 100 / total_steps),
                "message": f"Voix off française du plan {idx + 1}/{len(shots)}...",
            }
            narration = (sh.get("narration") or "").strip()
            if not narration:
                dur = 3.0
                n_samp = int(dur * SR)
                sfx = build_sfx_track(sh.get("sfx", "none"), n_samp, seed=idx + 10)
                shot_audio_mixes.append(sfx * 0.6)
                shot_durations.append(dur)
                shot_words_list.append([])
                continue

            mp3_tmp = proj_dir / f"{sh['id']}_vo.mp3"
            wav_tmp = proj_dir / f"{sh['id']}_vo.wav"
            words = asyncio.run(synth_edge_tts(narration, voice, speed, mp3_tmp))
            subprocess.run(
                [FFMPEG, "-y", "-i", str(mp3_tmp), "-ar", str(SR), "-ac", "1", "-sample_fmt", "s16", str(wav_tmp)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            with wave.open(str(wav_tmp), "rb") as wf:
                pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
            mp3_tmp.unlink(missing_ok=True)
            wav_tmp.unlink(missing_ok=True)

            nz = np.where(np.abs(pcm) > 0.008)[0]
            if len(nz) > 10:
                trim_start_s = max(0, nz[0] - int(0.06 * SR)) / SR
                pcm = pcm[max(0, nz[0] - int(0.06 * SR)) : min(len(pcm), nz[-1] + int(0.08 * SR))]
                for w_item in words:
                    w_item["start"] = max(0.0, w_item["start"] - trim_start_s) + 0.12
                    w_item["end"] = max(0.05, w_item["end"] - trim_start_s) + 0.12
            peak = float(np.max(np.abs(pcm))) if len(pcm) else 1.0
            if peak > 1e-4:
                pcm = pcm * (0.88 / peak)

            padded = np.concatenate([np.zeros(int(0.12 * SR), dtype=np.float32), pcm, np.zeros(int(0.24 * SR), dtype=np.float32)])
            dur = max(2.2, round((len(padded) / SR) * FPS) / float(FPS))
            target_n = int(round(dur * SR))
            if len(padded) < target_n:
                padded = np.pad(padded, (0, target_n - len(padded)))
            else:
                padded = padded[:target_n]

            sfx = build_sfx_track(sh.get("sfx", "none"), target_n, seed=idx + 50)
            win = int(0.08 * SR)
            env = np.convolve(np.abs(padded), np.ones(win, dtype=np.float32) / win, mode="same")
            duck = np.clip(0.82 - 4.0 * env, 0.24, 0.82).astype(np.float32)
            mix = np.zeros_like(sfx)
            mix[:, 0] = padded * 0.86 + sfx[:, 0] * duck
            mix[:, 1] = padded * 0.86 + sfx[:, 1] * duck
            mix = np.tanh(mix * 1.08) * 0.92

            shot_audio_mixes.append(mix.astype(np.float32))
            shot_durations.append(dur)
            phrases = []
            if words:
                grp = []
                for wi in words:
                    grp.append(wi)
                    if len(grp) >= 5 or wi["text"].endswith((".", "!", "?", ",", ":")):
                        phrases.append({"text": " ".join(x["text"] for x in grp), "start": grp[0]["start"], "end": grp[-1]["end"] + 0.12})
                        grp = []
                if grp:
                    phrases.append({"text": " ".join(x["text"] for x in grp), "start": grp[0]["start"], "end": grp[-1]["end"] + 0.18})
            else:
                phrases = [{"text": narration, "start": 0.1, "end": dur - 0.1}]
            shot_words_list.append(phrases)

        JOBS[job_id] = {"status": "running", "progress": 90, "message": "Montage vidéo 1080×1920 (9:16) & encodage MP4..."}

        full_audio = np.concatenate(shot_audio_mixes, axis=0)
        master_wav = proj_dir / "master_audio.wav"
        pcm16 = np.clip(full_audio * 32767.0, -32768, 32767).astype(np.int16)
        with wave.open(str(master_wav), "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(SR)
            wf.writeframes(pcm16.tobytes())

        out_mp4 = proj_dir / "output_short.mp4"
        cmd = [
            FFMPEG, "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{W}x{H}", "-pix_fmt", "bgr24", "-r", str(FPS), "-i", "-",
            "-i", str(master_wav),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-shortest",
            "-movflags", "+faststart",
            str(out_mp4),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        for idx, sh in enumerate(shots):
            bgr = cv2.imread(str(img_paths[idx]), cv2.IMREAD_COLOR)
            dur = shot_durations[idx]
            nf = int(round(dur * FPS))
            cam = sh.get("camera", "zoom_in")
            phrases = shot_words_list[idx]

            for f in range(nf):
                t = f / FPS
                u = float(np.clip(t / max(0.1, dur), 0.0, 1.0))
                eu = u * u * (3.0 - 2.0 * u)
                ang = 0.0
                if cam == "crash_zoom":
                    if t < 0.42:
                        p = (t / 0.42) - 1.0
                        eb = p * p * (2.8 * p + 1.8) + 1.0
                        zoom = 1.02 + 0.24 * eb
                        ang = -1.8 * (1.0 - t / 0.42)
                    else:
                        zoom = 1.26 + 0.04 * u
                    cx, cy = 0.50, 0.48
                elif cam == "pan_right":
                    cx, cy, zoom = 0.42 + 0.16 * eu, 0.50, 1.16
                elif cam == "shake_boom":
                    hit = max(0.0, t - min(max(0.35, dur * 0.28), max(0.2, dur - 0.8)))
                    shk = math.exp(-hit * 6.0) if t >= min(max(0.35, dur * 0.28), max(0.2, dur - 0.8)) else 0.0
                    cx = 0.50 + 0.03 * shk * math.sin(hit * 52.0)
                    cy = 0.50 + 0.025 * shk * math.cos(hit * 44.0)
                    zoom = 1.06 + 0.14 * eu
                    ang = 3.2 * shk * math.sin(hit * 38.0)
                elif cam == "pull_back":
                    cx, cy, zoom = 0.52 - 0.02 * eu, 0.44 + 0.06 * eu, 1.28 - 0.25 * eu
                else:
                    cx, cy, zoom = 0.50, 0.50 - 0.02 * eu, 1.02 + 0.20 * eu

                frame = render_camera(bgr, cx, cy, zoom, ang)
                if use_subs and phrases:
                    active_cap = ""
                    for ph in phrases:
                        if ph["start"] <= t <= ph["end"]:
                            active_cap = ph["text"]
                            break
                    if active_cap:
                        frame = draw_subtitle_overlay(frame, active_cap, font)

                if idx == len(shots) - 1 and t > dur - 0.25:
                    fade = max(0.0, (dur - t) / 0.25)
                    frame = (frame.astype(np.float32) * fade).astype(np.uint8)

                proc.stdin.write(frame.tobytes())

        proc.stdin.close()
        _, err = proc.communicate()
        master_wav.unlink(missing_ok=True)
        if proc.returncode != 0:
            raise RuntimeError(err.decode("utf-8", errors="ignore")[-500:])

        proj["video_url"] = f"/api/projects/{project_id}/video?t={int(time.time())}"
        proj["updated_at"] = int(time.time())
        meta_path.write_text(json.dumps(proj, ensure_ascii=False, indent=2), encoding="utf-8")

        JOBS[job_id] = {
            "status": "done",
            "progress": 100,
            "message": "Vidéo MP4 9:16 générée avec succès !",
            "video_url": proj["video_url"],
            "project": proj,
        }
    except Exception as e:
        JOBS[job_id] = {"status": "error", "progress": 100, "message": f"Erreur : {str(e)}"}


# --------------------- API ROUTES ---------------------
@app.get("/api/config")
def api_config():
    keys = load_keys()
    return {
        "voices": VOICES,
        "has_gemini_key": bool(keys.get("gemini_key")),
        "has_hf_token": bool(keys.get("hf_token")),
        "style_ref_url": f"/api/style-reference?t={int(time.time())}" if get_active_ref_path() else None,
    }


@app.post("/api/keys")
def update_keys(payload: KeysPayload):
    cur = load_keys()
    if payload.gemini_key.strip() != "***":
        cur["gemini_key"] = payload.gemini_key.strip()
    if payload.hf_token.strip() != "***":
        cur["hf_token"] = payload.hf_token.strip()
    save_keys(cur)
    return {"has_gemini_key": bool(cur.get("gemini_key")), "has_hf_token": bool(cur.get("hf_token"))}


@app.get("/api/style-reference")
def get_style_ref():
    p = get_active_ref_path()
    if not p:
        raise HTTPException(404, "Aucune image de référence")
    return FileResponse(p, media_type="image/jpeg")


@app.post("/api/style-reference")
async def upload_style_ref(file: UploadFile = File(...)):
    raw = await file.read()
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    im.save(CUSTOM_REF_PATH, format="JPEG", quality=92)
    return {"style_ref_url": f"/api/style-reference?t={int(time.time())}"}


@app.get("/api/projects")
def list_projects():
    items = []
    for d in sorted(PROJECTS_DIR.iterdir()):
        mp = d / "project.json"
        if mp.exists():
            try:
                p = json.loads(mp.read_text(encoding="utf-8"))
                items.append({"id": p["id"], "title": p["title"], "updated_at": p.get("updated_at", 0)})
            except Exception:
                pass
    items.sort(key=lambda x: x["updated_at"], reverse=True)
    return {"projects": items}


@app.post("/api/projects")
def create_project():
    pid = f"short-{uuid.uuid4().hex[:8]}"
    pdir = PROJECTS_DIR / pid
    pdir.mkdir(parents=True, exist_ok=True)
    proj = {
        "id": pid,
        "title": "Nouveau Short Stickman 3D",
        "style": "stickman_west_3d",
        "character_sheet": "Stickman en chemise rouge à manches courtes, bretelles noires et chapeau 3D en cuir usé",
        "custom_style": "",
        "voice": "fr-FR-RemyMultilingualNeural",
        "speed": 1.06,
        "subtitles": True,
        "video_url": None,
        "updated_at": int(time.time()),
        "shots": [
            {
                "id": f"shot_{uuid.uuid4().hex[:6]}",
                "narration": "Dans une vieille banque du Far West, le shérif garde le coffre-fort avec un grand sourire confiant.",
                "prompt": "Dans une banque en bois du Far West éclairée par des lanternes chaudes, le personnage fait un clin d'œil malicieux devant une lourde porte de coffre-fort en acier 3D entrouverte remplie de lingots d'or brillants.",
                "camera": "crash_zoom",
                "sfx": "kaching",
                "image": None,
            },
            {
                "id": f"shot_{uuid.uuid4().hex[:6]}",
                "narration": "Mais juste derrière le mur en bois, un complice creuse un tunnel au mauvais endroit et tombe dans le saloon voisin !",
                "prompt": "Dans un saloon 3D réaliste du Far West avec tonneaux et bouteilles, un stickman surgit d'un trou dans le plancher en bois avec une pelle 3D à la main et un air totalement surpris.",
                "camera": "shake_boom",
                "sfx": "bonk",
                "image": None,
            },
        ],
    }
    (pdir / "project.json").write_text(json.dumps(proj, ensure_ascii=False, indent=2), encoding="utf-8")
    return proj


@app.get("/api/projects/{project_id}")
def get_project(project_id: str):
    mp = PROJECTS_DIR / project_id / "project.json"
    if not mp.exists():
        raise HTTPException(404, "Projet introuvable")
    return json.loads(mp.read_text(encoding="utf-8"))


@app.put("/api/projects/{project_id}")
def save_project(project_id: str, payload: ProjectSave):
    pdir = PROJECTS_DIR / project_id
    mp = pdir / "project.json"
    if not mp.exists():
        raise HTTPException(404, "Projet introuvable")
    existing = json.loads(mp.read_text(encoding="utf-8"))
    existing.update(payload.model_dump())
    existing["updated_at"] = int(time.time())
    mp.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    return existing


@app.post("/api/projects/{project_id}/shots/{shot_id}/generate-image")
def api_gen_shot_image(project_id: str, shot_id: str, req: GenImageReq):
    pdir = PROJECTS_DIR / project_id
    mp = pdir / "project.json"
    if not mp.exists():
        raise HTTPException(404, "Projet introuvable")
    proj = json.loads(mp.read_text(encoding="utf-8"))
    out_img = pdir / f"{shot_id}.jpg"
    try:
        provider = generate_free_ai_image(req.prompt, req.style, req.character_sheet, req.custom_style, out_img, seed=req.seed)
    except Exception as e:
        raise HTTPException(429, str(e))
    img_url = f"/api/projects/{project_id}/shot-image/{shot_id}?t={int(time.time())}"
    for sh in proj["shots"]:
        if sh["id"] == shot_id:
            sh["prompt"] = req.prompt
            sh["image"] = img_url
    proj["updated_at"] = int(time.time())
    mp.write_text(json.dumps(proj, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"image_url": img_url, "provider": provider}


@app.post("/api/projects/{project_id}/shots/{shot_id}/upload-image")
async def api_upload_shot_image(project_id: str, shot_id: str, file: UploadFile = File(...)):
    pdir = PROJECTS_DIR / project_id
    mp = pdir / "project.json"
    if not mp.exists():
        raise HTTPException(404, "Projet introuvable")
    raw = await file.read()
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    out_img = pdir / f"{shot_id}.jpg"
    im.save(out_img, format="JPEG", quality=92)
    proj = json.loads(mp.read_text(encoding="utf-8"))
    img_url = f"/api/projects/{project_id}/shot-image/{shot_id}?t={int(time.time())}"
    for sh in proj["shots"]:
        if sh["id"] == shot_id:
            sh["image"] = img_url
    mp.write_text(json.dumps(proj, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"image_url": img_url}


@app.get("/api/projects/{project_id}/shot-image/{shot_id}")
def get_shot_image(project_id: str, shot_id: str):
    p = PROJECTS_DIR / project_id / f"{shot_id}.jpg"
    if not p.exists():
        raise HTTPException(404, "Image non trouvée")
    return FileResponse(p, media_type="image/jpeg")


@app.get("/api/demo-asset/{filename}")
def get_demo_asset(filename: str):
    safe = Path(filename).name
    p = WORKSPACE_DIR / "keyframes" / safe
    if not p.exists():
        raise HTTPException(404, "Asset introuvable")
    return FileResponse(p, media_type="image/png")


@app.get("/api/demo-video")
def get_demo_video():
    p = WORKSPACE_DIR / "film_complet.mp4"
    if not p.exists():
        raise HTTPException(404, "Vidéo démo introuvable")
    return FileResponse(p, media_type="video/mp4", filename="demo_mine_dor.mp4")


@app.post("/api/voice-preview")
async def api_voice_preview(req: VoicePreviewReq):
    tmp = Path(tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name)
    await synth_edge_tts(req.text[:240], req.voice, req.speed, tmp)
    data = tmp.read_bytes()
    tmp.unlink(missing_ok=True)
    return StreamingResponse(io.BytesIO(data), media_type="audio/mpeg")


@app.post("/api/projects/{project_id}/render")
def start_render(project_id: str, gen_missing_only: bool = True):
    mp = PROJECTS_DIR / project_id / "project.json"
    if not mp.exists():
        raise HTTPException(404, "Projet introuvable")
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    JOBS[job_id] = {"status": "running", "progress": 3, "message": "Démarrage du moteur Ultra-HD..."}
    t = threading.Thread(target=render_full_project_job, args=(job_id, project_id, gen_missing_only), daemon=True)
    t.start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "Job introuvable")
    return JOBS[job_id]


@app.get("/api/projects/{project_id}/video")
def get_project_video(project_id: str):
    p = PROJECTS_DIR / project_id / "output_short.mp4"
    if not p.exists():
        raise HTTPException(404, "Vidéo non générée")
    return FileResponse(p, media_type="video/mp4", filename=f"{project_id}.mp4")


@app.get("/api/download-studio-zip")
def download_studio_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in [
            "app.py",
            "requirements.txt",
            "LANCER_WINDOWS.bat",
            "LANCER_MAC_LINUX.sh",
            "LISEZMOI.md",
            "static/index.html",
            "static/style_ref_default.jpg",
        ]:
            fp = BASE_DIR / rel
            if fp.exists():
                zf.write(fp, arcname=f"Stick3D_Studio_Gratuit/{rel}")
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="Stick3D_Studio_Gratuit_PC.zip"'},
    )


@app.get("/", response_class=HTMLResponse)
def index_page():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")
