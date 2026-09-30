from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import polars as pl
import plotly.graph_objects as go
import requests
from functools import lru_cache
from dash import Dash, Input, Output, State, dcc, html, dash_table, no_update, ctx, ALL

# ============================================================
# MPLADS AI MONITOR — DASH FRONTEND
# ============================================================
# Architecture:
#   Dash/Plotly UI -> FastAPI -> processed MPLADS analytical data
#
# This file intentionally does NOT recompute risk, ML, feature
# engineering, or business logic. Those remain in backend/.
# ============================================================

API_BASE = os.getenv("MPLADS_API_URL", "http://127.0.0.1:8000").rstrip("/")
DASH_HOST = os.getenv("MPLADS_DASH_HOST", "127.0.0.1")
DASH_PORT = int(os.getenv("MPLADS_DASH_PORT", "8050"))
REQUEST_TIMEOUT = int(os.getenv("MPLADS_DASH_TIMEOUT", "90"))
APP_VERSION = "10.0.0-CLEAN-ANALYTICS"

# Living Seal integration: pure-CSS sidebar emblem + seal-to-Team-Info trigger.
QUEUE_LIMIT = 500

RISK_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
RISK_RANGES = {
    "LOW": "0–25",
    "MEDIUM": ">25–50",
    "HIGH": ">50–75",
    "CRITICAL": ">75–100",
}

FONT_HEAD = "Bahnschrift, Segoe UI, Arial, sans-serif"
FONT_BODY = "Segoe UI, Arial, sans-serif"
FONT_MONO = "Cascadia Mono, Consolas, monospace"


# ============================================================
# TEAM / BRAND CONFIGURATION — EDIT THIS BLOCK ONLY
# ============================================================
# The JMI logo resolver is local-first:
#   1) dashboard/assets/jmi_logo.png / jmi-logo.png / common image variants
#   2) MPLADS_JMI_LOGO_SRC if explicitly configured
#   3) official JMI logo page as a best-effort cache fallback
#   4) clean CSS/text badge — never a broken image
# ============================================================

TEAM_NAME = "Team Fresh Minds"
TEAM_ID = "120613"
GITHUB_REPOSITORY_URL = "https://github.com/abdulhammad13/mplads-ai-monitor"

# Official JMI logo page. Defined BEFORE the resolver is called.
JMI_LOGO_PAGE_URL = "https://www.jmi.ac.in/About-Jamia/Profile/History/Jamia%27s-Logo"


def _ensure_jmi_logo_asset() -> str:
    """Resolve a usable JMI logo without ever breaking dashboard startup."""
    assets_dir = Path(__file__).resolve().parent / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Local-first: the user's actual project contains jmi_logo.png.
    local_candidates = (
        "jmi_logo.png",
        "jmi-logo.png",
        "jmi_logo.webp",
        "jmi-logo.webp",
        "jmi_logo.jpg",
        "jmi-logo.jpg",
        "jmi_logo.jpeg",
        "jmi-logo.jpeg",
    )
    for candidate in local_candidates:
        path = assets_dir / candidate
        if path.is_file() and path.stat().st_size > 256:
            return f"/assets/{candidate}"

    configured = os.getenv("MPLADS_JMI_LOGO_SRC", "").strip()
    if configured:
        if configured.startswith("/assets/"):
            local_path = assets_dir / Path(configured).name
            if local_path.is_file() and local_path.stat().st_size > 256:
                return configured
        elif configured.startswith(("http://", "https://")):
            return configured

    # Best-effort official-page discovery. Network failure is harmless.
    try:
        page = requests.get(
            JMI_LOGO_PAGE_URL,
            timeout=12,
            headers={"User-Agent": "MPLADS-AI-Monitor/6.9"},
        )
        page.raise_for_status()

        tags = re.findall(r"<img\b[^>]*>", page.text, flags=re.I)
        candidates = []

        for tag in tags:
            alt_m = re.search(
                r"\balt=[\"]([^\"]*)[\"]|\balt=[']([^']*)[']",
                tag,
                flags=re.I,
            )
            src_m = re.search(
                r"\b(?:src|data-src)=[\"]([^\"]+)[\"]|\b(?:src|data-src)=[']([^']+)[']",
                tag,
                flags=re.I,
            )
            if not src_m:
                continue

            src_value = next((g for g in src_m.groups() if g), "").strip()
            if not src_value:
                continue

            alt_value = " ".join(
                g for g in (alt_m.groups() if alt_m else ()) if g
            )
            lowered = src_value.casefold()
            score = (
                4 * ("logo" in alt_value.casefold())
                + 3 * ("logo" in lowered)
                + 1 * lowered.endswith((".png", ".jpg", ".jpeg", ".webp", ".svg"))
            )
            if score:
                candidates.append((score, urljoin(JMI_LOGO_PAGE_URL, src_value)))

        candidates.sort(key=lambda item: item[0], reverse=True)

        for _, image_url in candidates[:8]:
            try:
                image_response = requests.get(
                    image_url,
                    timeout=12,
                    headers={"User-Agent": "MPLADS-AI-Monitor/6.9"},
                )
                image_response.raise_for_status()

                content_type = image_response.headers.get("content-type", "").lower()
                if "svg" in content_type or image_url.casefold().endswith(".svg"):
                    ext = ".svg"
                elif "webp" in content_type or image_url.casefold().endswith(".webp"):
                    ext = ".webp"
                elif "jpeg" in content_type or "jpg" in content_type or image_url.casefold().endswith((".jpg", ".jpeg")):
                    ext = ".jpg"
                else:
                    ext = ".png"

                target = assets_dir / f"jmi_logo{ext}"
                target.write_bytes(image_response.content)

                if target.is_file() and target.stat().st_size > 256:
                    return f"/assets/{target.name}"
            except requests.RequestException:
                continue
            except OSError:
                continue

    except requests.RequestException:
        pass
    except OSError:
        pass

    return ""


JMI_LOGO_SRC = _ensure_jmi_logo_asset()

TEAM_MEMBERS = [
    {
        "name": "Abdul Hammad",
        "role": "AI / ML Architect",
        "degree": "B.Tech '30 · JMI",
        "bio": "Designs and trains the anomaly-detection core — engineering features, tuning the Isolation Forest, and turning raw MPLADS records into explainable risk signals the dashboard can defend.",
        "photo": "/assets/team/abdul_hammad.png",
    },
    {
        "name": "MD Faisal Raza",
        "role": "R&D Pipeline Engineer",
        "degree": "B.Tech '30 · JMI",
        "bio": "Builds and hardens the analytical pipeline — wiring data flows, benchmarking rule thresholds, and iterating on model behaviour until the risk engine performs consistently across states and categories.",
        "photo": "/assets/team/md_faisal_raza.png",
    },
    {
        "name": "Zishan Afroz",
        "role": "Innovation & Validation Lead",
        "degree": "B.Tech '30 · JMI",
        "bio": "Drives methodology research and validation — stress-testing scoring logic, auditing feature choices, and ensuring every signal in the pipeline survives real-world edge cases before it reaches the analyst.",
        "photo": "/assets/team/zishan_afroz.png",
    },
    {
        "name": "Zarish Parveen",
        "role": "Presentation & Deck Designer",
        "degree": "B.Tech '30 · JMI",
        "bio": "Crafts the visual narrative of the project — designing slide hierarchies, data-backed visuals, and the presentation rhythm that makes complex MPLADS analytics land in a single glance.",
        "photo": "/assets/team/zarish_parveen.png",
    },
    {
        "name": "Mobashra Fatima",
        "role": "Presentation & Narrative Designer",
        "degree": "B.Tech '30 · JMI",
        "bio": "Shapes the spoken and written narrative — scripting demo flow, structuring the pitch story arc, and making sure every reviewer walks away understanding what the system actually proves.",
        "photo": "/assets/team/mobashra_fatima.png",
    },
    {
        "name": "Nasiba Hoda",
        "role": "Product Pitch & Storytelling Lead",
        "degree": "B.Tech '30 · JMI",
        "bio": "Owns the product voice — translating technical capability into a compelling pitch, framing the problem MPLADS AI Monitor solves, and landing the message with judges, mentors, and stakeholders.",
        "photo": "/assets/team/nasiba_hoda.png",
    },
]

# User-supplied MPLADS emblem reference, embedded directly so the
# footer works without requiring another file.
MPLADS_EMBLEM_SRC = "data:image/webp;base64,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"


def _member_initials(name: str) -> str:
    parts = [p for p in str(name).split() if p]
    if not parts:
        return "TF"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()


def model_logo(class_name: str = "model-logo") -> html.Div:
    """Unique vector-style MPLADS AI model mark; CSS-rendered, crisp at any DPI."""
    return html.Div(
        [
            html.Span(className="model-logo-orbit"),
            html.Span(className="model-logo-core"),
            html.Span(className="model-logo-bar bar-a"),
            html.Span(className="model-logo-bar bar-b"),
            html.Span(className="model-logo-bar bar-c"),
            html.Span("AI", className="model-logo-ai"),
        ],
        className=class_name,
        title="MPLADS AI Monitor",
    )


def model_logo_interactive() -> html.Div:
    """Footer-only interactive model logo with animated cosmic background stage."""
    return html.Div(
        [
            # ── Animated cosmic stage ──────────────────────────────────────────
            html.Div(
                [
                    # Spinning aurora cones
                    html.Span(className="mlg-aurora mlg-a1"),
                    html.Span(className="mlg-aurora mlg-a2"),
                    html.Span(className="mlg-aurora mlg-a3"),
                    # Pulsing orbital rings
                    html.Span(className="mlg-ring mlg-r1"),
                    html.Span(className="mlg-ring mlg-r2"),
                    html.Span(className="mlg-ring mlg-r3"),
                    # Floating star particles
                    html.Span(className="mlg-star s1"),
                    html.Span(className="mlg-star s2"),
                    html.Span(className="mlg-star s3"),
                    html.Span(className="mlg-star s4"),
                    html.Span(className="mlg-star s5"),
                    html.Span(className="mlg-star s6"),
                    # Neural connection arcs
                    html.Span(className="mlg-arc arc1"),
                    html.Span(className="mlg-arc arc2"),
                    # The actual logo mark — floats at center
                    html.Div(
                        [
                            html.Span(className="model-logo-orbit"),
                            html.Span(className="model-logo-core"),
                            html.Span(className="model-logo-bar bar-a"),
                            html.Span(className="model-logo-bar bar-b"),
                            html.Span(className="model-logo-bar bar-c"),
                            html.Span("AI", className="model-logo-ai"),
                        ],
                        className="model-logo mlg-mark",
                    ),
                    # Click-ripple burst (CSS-activated)
                    html.Span(className="mlg-ripple"),
                    # Hover label
                    html.Span("MPLADS AI MONITOR", className="mlg-hint"),
                ],
                className="model-logo-stage",
                id="model-logo-interactive",
                n_clicks=0,
                title="Click me",
            ),
        ],
        className="model-logo-interactive-wrap",
    )


TEAM_PHOTO_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
TEAM_PHOTO_CANONICAL_STEMS = {
    "Abdul Hammad": "abdul_hammad",
    "MD Faisal Raza": "md_faisal_raza",
    "Zishan Afroz": "zishan_afroz",
    "Zarish Parveen": "zarish_parveen",
    "Mobashra Fatima": "mobashra_fatima",
    "Nasiba Hoda": "nasiba_hoda",
}


def _normalise_photo_stem(value: str) -> str:
    """Canonical filename stem: first_lastname, lowercase, ASCII-safe."""
    stem = Path(str(value or "").replace("\\", "/")).stem
    stem = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_").casefold()
    return re.sub(r"_+", "_", stem)


def _canonical_member_photo_stem(name: str) -> str:
    canonical = TEAM_PHOTO_CANONICAL_STEMS.get(str(name or "").strip())
    if canonical:
        return canonical
    parts = [p for p in re.split(r"\s+", str(name or "").strip()) if p]
    return _normalise_photo_stem("_".join(parts))


def _photo_url_with_version(path: Path, assets_root: Path) -> str:
    rel = path.relative_to(assets_root).as_posix()
    # Cache-bust only when the underlying file changes. This prevents stale,
    # previously downscaled portraits from surviving a replacement upload.
    try:
        version = path.stat().st_mtime_ns
    except OSError:
        version = "1"
    return f"/assets/{rel}?v={version}"


def _find_best_team_photo(member: dict[str, Any]) -> Path | None:
    """Find the best available portrait using the project's naming convention.

    Preferred convention:
        dashboard/assets/team/first_lastname.PNG|JPG|JPEG|WEBP

    The resolver also accepts an exact configured filename and legacy variants,
    but never creates a broken-image element. The largest/highest-priority file
    is chosen when more than one extension exists.
    """
    assets_root = Path(__file__).resolve().parent / "assets"
    team_root = assets_root / "team"
    team_root.mkdir(parents=True, exist_ok=True)

    name = str(member.get("name") or "Team Member")
    canonical = _canonical_member_photo_stem(name)
    configured = str(member.get("photo") or "").strip()
    configured_path = Path(configured.removeprefix("/assets/").replace("\\", "/"))
    configured_stem = _normalise_photo_stem(configured_path.name or canonical)
    stems = [configured_stem, canonical]

    # Known legacy spellings kept only for compatibility with older assets.
    if canonical == "mobashra_fatima":
        stems += ["moobashra_fatima"]
    if canonical == "md_faisal_raza":
        stems += ["faisal_raza"]

    candidates: list[Path] = []
    try:
        for child in team_root.iterdir():
            if not child.is_file() or child.suffix.casefold() not in TEAM_PHOTO_EXTENSIONS:
                continue
            child_stem = _normalise_photo_stem(child.name)
            if child_stem in stems or child_stem.endswith("@2x") and child_stem[:-3] in stems or child_stem.endswith("_2x") and child_stem[:-3] in stems:
                candidates.append(child)
    except OSError:
        return None

    if not candidates:
        # Exact configured path first if it happens to exist with a supported ext.
        base = team_root / Path(configured).name
        for ext in TEAM_PHOTO_EXTENSIONS:
            p = base.with_suffix(ext)
            if p.is_file() and p.stat().st_size > 1024:
                candidates.append(p)

    if not candidates:
        return None

    # Prefer @2x / _2x assets, then extension quality, then larger file size.
    def rank(p: Path) -> tuple[int, int, int]:
        stem = p.stem.casefold()
        retina = int(stem.endswith("@2x") or stem.endswith("_2x"))
        ext_score = {".png": 4, ".jpg": 3, ".jpeg": 3, ".webp": 2}.get(p.suffix.casefold(), 0)
        try:
            size = p.stat().st_size
        except OSError:
            size = 0
        return retina, ext_score, size

    return max(candidates, key=rank)


def _resolve_team_photo(photo: str, member: dict[str, Any] | None = None) -> str:
    """Return a cache-busted, Dash-served photo URL or an empty string."""
    assets_root = Path(__file__).resolve().parent / "assets"
    team_root = assets_root / "team"

    if member is not None:
        best = _find_best_team_photo(member)
        return _photo_url_with_version(best, assets_root) if best else ""

    raw = str(photo or "").strip()
    if not raw:
        return ""
    relative = raw.removeprefix("/assets/").replace("\\", "/").lstrip("/")
    candidate = (assets_root / relative).resolve()
    try:
        candidate.relative_to(assets_root.resolve())
    except ValueError:
        return ""
    if candidate.is_file() and candidate.stat().st_size > 1024 and candidate.suffix.casefold() in TEAM_PHOTO_EXTENSIONS:
        return _photo_url_with_version(candidate, assets_root)
    stem = candidate.with_suffix("")
    for ext in TEAM_PHOTO_EXTENSIONS:
        variant = Path(f"{stem}{ext}")
        if variant.is_file() and variant.stat().st_size > 1024:
            return _photo_url_with_version(variant, assets_root)
    return ""


def team_member_avatar(member: dict[str, Any]) -> html.Div:
    """Render a stable, non-distorting team portrait or a clean initials fallback."""
    best = _find_best_team_photo(member)
    assets_root = Path(__file__).resolve().parent / "assets"

    if best:
        photo_url = _photo_url_with_version(best, assets_root)
        return html.Div(
            html.Img(
                src=photo_url,
                alt=f"{member.get('name', 'Team member')} portrait",
                className="team-member-photo",
                draggable="false",
            ),
            className="team-member-avatar team-member-avatar-photo",
        )

    return html.Div(
        [
            html.Div(
                _member_initials(member.get("name", "")),
                className="team-member-avatar-initials-main",
            ),
            html.Div(
                "TEAM MEMBER",
                className="team-member-avatar-caption",
            ),
        ],
        className="team-member-avatar team-member-avatar-initials",
    )


def team_member_card(member: dict[str, Any]) -> html.Div:
    if JMI_LOGO_SRC:
        jmi_badge = html.Div(
            html.Img(
                src=JMI_LOGO_SRC,
                alt="Jamia Millia Islamia",
                className="team-member-jmi-logo",
            ),
            className="team-member-jmi-badge team-member-jmi-badge-logo",
            title="Jamia Millia Islamia",
        )
    else:
        jmi_badge = html.Div(
            [html.Span("JMI", className="team-member-jmi-fallback-main"), html.Span("Jamia", className="team-member-jmi-fallback-sub")],
            className="team-member-jmi-badge team-member-jmi-badge-fallback",
            title="Jamia Millia Islamia",
        )

    initials = _member_initials(member.get("name", ""))
    role = str(member.get("role", "Team Role"))
    degree = str(member.get("degree", "B.Tech '30 · JMI"))
    return html.Div(
        [
            html.Div(
                [
                    html.Div(team_member_avatar(member), className="team-member-photo-frame"),
                    html.Div([html.Span(initials, className="team-member-photo-index")], className="team-member-photo-corner"),
                ],
                className="team-member-visual",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(str(member.get("name", "Team Member")), className="team-member-name"),
                            jmi_badge,
                        ],
                        className="team-member-name-row",
                    ),
                    html.Div(role, className="team-member-role"),
                    html.Div(degree, className="team-member-degree"),
                    html.Div(str(member.get("bio", "")), className="team-member-bio"),
                    html.Div(
                        [html.Span("FRESH MINDS", className="team-member-tag"), html.Span("JMI", className="team-member-tag team-member-tag-gold")],
                        className="team-member-tags",
                    ),
                ],
                className="team-member-content",
            ),
        ],
        className="team-member-card",
    )

def team_info_modal() -> html.Div:
    """Premium Team Info modal with stable responsive card geometry."""
    return html.Div(
        [
            html.Div(
                id="team-info-backdrop",
                n_clicks=0,
                className="team-modal-backdrop",
                title="Close Team Info",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    model_logo("model-logo model-logo-modal"),
                                    html.Div(
                                        [
                                            html.Div("TEAM FRESH MINDS", className="team-modal-kicker"),
                                            html.H2(
                                                "The people behind the model",
                                                id="team-info-title",
                                                className="team-modal-title",
                                            ),
                                            html.Div(
                                                "Six-member engineering team · B.Tech '30 · Jamia Millia Islamia",
                                                className="team-modal-subtitle",
                                            ),
                                        ],
                                        className="team-modal-heading-copy",
                                    ),
                                ],
                                className="team-modal-heading",
                            ),
                            html.Div(
                                [
                                    html.A(
                                        [
                                            html.Span("↗", className="team-modal-github-icon"),
                                            html.Span("VIEW SOURCE", className="team-modal-github-label"),
                                        ],
                                        href=GITHUB_REPOSITORY_URL,
                                        target="_blank",
                                        rel="noopener noreferrer",
                                        className="team-modal-github",
                                        title="Open the MPLADS AI Monitor GitHub repository",
                                        style={
                                            "display": "inline-flex",
                                            "alignItems": "center",
                                            "gap": "4px",
                                            "height": "18px",
                                            "minHeight": "18px",
                                            "padding": "0 7px",
                                            "borderRadius": "999px",
                                            "border": "1px solid rgba(154, 112, 214, 0.22)",
                                            "background": "linear-gradient(90deg, rgba(121, 191, 235, 0.045), rgba(186, 135, 229, 0.07))",
                                            "color": "#B99AE6",
                                            "font": '700 6px/1 "Cascadia Mono", Consolas, monospace',
                                            "letterSpacing": "0.11em",
                                            "textDecoration": "none",
                                            "whiteSpace": "nowrap",
                                            "boxSizing": "border-box",
                                            "opacity": "0.92",
                                        },
                                    ),
                                    html.Button(
                                        "×",
                                        id="team-info-close",
                                        n_clicks=0,
                                        className="team-modal-close",
                                        title="Close Team Info",
                                        **{"aria-label": "Close Team Info"},
                                    ),
                                ],
                                className="team-modal-actions",
                            ),
                        ],
                        className="team-modal-top",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span("MPLADS AI MONITOR", className="team-modal-pill"),
                                    html.Span(
                                        "HUMAN-IN-THE-LOOP",
                                        className="team-modal-pill team-modal-pill-alt",
                                    ),
                                ],
                                className="team-modal-pills",
                            ),
                            html.Div(
                                [team_member_card(member) for member in TEAM_MEMBERS[:6]],
                                className="team-members-grid",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Span("", style={
                                                "display": "inline-block",
                                                "width": "14px",
                                                "height": "1px",
                                                "borderRadius": "999px",
                                                "background": "linear-gradient(90deg, rgba(114, 207, 239, .75), rgba(188, 140, 229, .32))",
                                                "boxShadow": "0 0 7px rgba(114, 207, 239, .12)",
                                            }),
                                            html.Span("TEAM CAPABILITIES", style={
                                                "font": '800 5.2px/1 "Cascadia Mono", Consolas, monospace',
                                                "letterSpacing": ".18em",
                                                "color": "#7DAFC2",
                                                "textTransform": "uppercase",
                                            }),
                                        ],
                                        className="team-capabilities-kicker",
                                        style={
                                            "display": "flex",
                                            "alignItems": "center",
                                            "gap": "6px",
                                            "margin": "0 0 6px 0",
                                            "height": "8px",
                                        },
                                    ),
                                    html.Div(
                                        [
                                            html.Div([
                                                html.Span("", style={"width":"5px","height":"5px","borderRadius":"50%","background":"#52D2F2","boxShadow":"0 0 8px rgba(82,210,242,.42)","flex":"0 0 5px"}),
                                                html.Span("ANOMALY DETECTION"),
                                            ], className="team-capability-chip", style={"display":"inline-flex","alignItems":"center","gap":"7px","height":"25px","padding":"0 10px","borderRadius":"999px","boxSizing":"border-box","border":"1px solid rgba(82,210,242,.22)","background":"linear-gradient(135deg, rgba(82,210,242,.105), rgba(82,210,242,.018))","color":"#9DE5F4","font":"800 7.2px/1 Cascadia Mono, Consolas, monospace","letterSpacing":".11em","whiteSpace":"nowrap","boxShadow":"inset 0 1px 0 rgba(255,255,255,.045), 0 4px 12px rgba(10,32,42,.18)"}),
                                            html.Div([
                                                html.Span("", style={"width":"5px","height":"5px","borderRadius":"50%","background":"#80B9E1","boxShadow":"0 0 8px rgba(128,185,225,.38)","flex":"0 0 5px"}),
                                                html.Span("DATA ENGINEERING"),
                                            ], className="team-capability-chip", style={"display":"inline-flex","alignItems":"center","gap":"7px","height":"25px","padding":"0 10px","borderRadius":"999px","boxSizing":"border-box","border":"1px solid rgba(128,185,225,.20)","background":"linear-gradient(135deg, rgba(128,185,225,.09), rgba(128,185,225,.016))","color":"#B5D2E3","font":"800 7.2px/1 Cascadia Mono, Consolas, monospace","letterSpacing":".11em","whiteSpace":"nowrap","boxShadow":"inset 0 1px 0 rgba(255,255,255,.045), 0 4px 12px rgba(13,27,40,.18)"}),
                                            html.Div([
                                                html.Span("", style={"width":"5px","height":"5px","borderRadius":"50%","background":"#B985E7","boxShadow":"0 0 8px rgba(185,133,231,.42)","flex":"0 0 5px"}),
                                                html.Span("VALIDATION"),
                                            ], className="team-capability-chip", style={"display":"inline-flex","alignItems":"center","gap":"7px","height":"25px","padding":"0 10px","borderRadius":"999px","boxSizing":"border-box","border":"1px solid rgba(185,133,231,.22)","background":"linear-gradient(135deg, rgba(185,133,231,.10), rgba(185,133,231,.018))","color":"#D0BFE6","font":"800 7.2px/1 Cascadia Mono, Consolas, monospace","letterSpacing":".11em","whiteSpace":"nowrap","boxShadow":"inset 0 1px 0 rgba(255,255,255,.045), 0 4px 12px rgba(26,16,40,.18)"}),
                                            html.Div([
                                                html.Span("", style={"width":"5px","height":"5px","borderRadius":"50%","background":"#7CC9A5","boxShadow":"0 0 8px rgba(124,201,165,.40)","flex":"0 0 5px"}),
                                                html.Span("PRESENTATION"),
                                            ], className="team-capability-chip", style={"display":"inline-flex","alignItems":"center","gap":"7px","height":"25px","padding":"0 10px","borderRadius":"999px","boxSizing":"border-box","border":"1px solid rgba(124,201,165,.20)","background":"linear-gradient(135deg, rgba(124,201,165,.085), rgba(124,201,165,.016))","color":"#B9DDCF","font":"800 7.2px/1 Cascadia Mono, Consolas, monospace","letterSpacing":".11em","whiteSpace":"nowrap","boxShadow":"inset 0 1px 0 rgba(255,255,255,.045), 0 4px 12px rgba(13,36,27,.18)"}),
                                            html.Div([
                                                html.Span("", style={"width":"5px","height":"5px","borderRadius":"50%","background":"#DDB76E","boxShadow":"0 0 8px rgba(221,183,110,.40)","flex":"0 0 5px"}),
                                                html.Span("PRODUCT STORY"),
                                            ], className="team-capability-chip", style={"display":"inline-flex","alignItems":"center","gap":"7px","height":"25px","padding":"0 10px","borderRadius":"999px","boxSizing":"border-box","border":"1px solid rgba(221,183,110,.20)","background":"linear-gradient(135deg, rgba(221,183,110,.085), rgba(221,183,110,.014))","color":"#E5D2A2","font":"800 7.2px/1 Cascadia Mono, Consolas, monospace","letterSpacing":".11em","whiteSpace":"nowrap","boxShadow":"inset 0 1px 0 rgba(255,255,255,.045), 0 4px 12px rgba(42,31,12,.18)"}),
                                        ],
                                        className="team-capabilities-row",
                                        style={
                                            "display": "flex",
                                            "alignItems": "center",
                                            "flexWrap": "wrap",
                                            "gap": "7px",
                                            "minHeight": "25px",
                                        },
                                    ),
                                ],
                                className="team-capabilities",
                                style={
                                    "marginTop": "9px",
                                    "padding": "8px 10px 9px",
                                    "borderRadius": "14px",
                                    "border": "1px solid rgba(124, 207, 232, .10)",
                                    "background": "radial-gradient(180px 55px at 0% 0%, rgba(82,210,242,.055), transparent 72%), radial-gradient(180px 55px at 100% 100%, rgba(185,133,231,.045), transparent 72%), linear-gradient(120deg, rgba(255,255,255,.016), rgba(255,255,255,.006))",
                                    "boxShadow": "inset 0 1px 0 rgba(255,255,255,.026), 0 8px 20px rgba(0,0,0,.10)",
                                    "boxSizing": "border-box",
                                },
                            ),
                        ],
                        className="team-modal-scroll",
                    ),
                ],
                className="team-modal-dialog",
                role="dialog",
                **{
                    "aria-modal": "true",
                    "aria-labelledby": "team-info-title",
                    "tabIndex": "-1",
                },
            ),
        ],
        id="team-info-modal",
        className="team-modal is-hidden",
    )


def seal_viewer() -> html.Div:
    """Fullscreen magnified view for the sidebar MPLADS seal."""
    return html.Div(
        [
            html.Div(
                id="seal-viewer-backdrop",
                n_clicks=0,
                className="seal-viewer-backdrop",
                title="Close seal preview",
            ),
            html.Div(
                [
                    html.Button(
                        "×",
                        id="seal-viewer-close",
                        n_clicks=0,
                        className="seal-viewer-close",
                        title="Close seal preview",
                        **{"aria-label": "Close seal preview"},
                    ),
                    html.Div(
                        [
                            html.Div(
                                "PROGRAMME IDENTITY",
                                className="seal-viewer-eyebrow",
                            ),
                            html.Div(
                                [
                                    html.Span("MPLADS", className="seal-viewer-title-main"),
                                    html.Span("AI MONITOR", className="seal-viewer-title-accent"),
                                ],
                                className="seal-viewer-title",
                            ),
                            html.Div(
                                "EXPLAINABLE MONITORING • ANOMALY DETECTION • PUBLIC WORKS INTELLIGENCE",
                                className="seal-viewer-subtitle",
                            ),
                            html.Div(
                                [
                                    html.Span(className="seal-viewer-status-dot"),
                                    html.Span("SYSTEM MARK", className="seal-viewer-status-text"),
                                    html.Span("·", className="seal-viewer-status-sep"),
                                    html.Span("MPLADS", className="seal-viewer-status-muted"),
                                ],
                                className="seal-viewer-status",
                            ),
                        ],
                        className="seal-viewer-heading",
                    ),
                    html.Div(
                        html.Div(
                            [
                                html.Span(className="brand-seal-aura"),
                                html.Span(className="brand-seal-chakra"),
                                html.Span(className="brand-seal-ring ring-a"),
                                html.Span(className="brand-seal-ring ring-b"),
                                html.Span(className="brand-seal-dome"),
                                html.Span(className="brand-seal-plinth"),
                                html.Span(
                                    [
                                        html.Span(className="seal-fig fig-saffron"),
                                        html.Span(className="seal-fig fig-blue"),
                                        html.Span(className="seal-fig fig-green"),
                                    ],
                                    className="brand-seal-triad",
                                ),
                            ],
                            className="brand-seal seal-viewer-seal",
                            title="MPLADS AI Monitor",
                        ),
                        className="seal-viewer-seal-wrap",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span("MONITOR", className="seal-viewer-chip"),
                                    html.Span("DETECT", className="seal-viewer-chip"),
                                    html.Span("EXPLAIN", className="seal-viewer-chip"),
                                    html.Span("REVIEW", className="seal-viewer-chip"),
                                ],
                                className="seal-viewer-chips",
                            ),
                            html.Div(
                                "A visual signature for intelligent, explainable MPLADS monitoring.",
                                className="seal-viewer-caption",
                            ),
                        ],
                        className="seal-viewer-bottom",
                    ),
                    html.Div(
                        [
                            html.Span("CLICK OUTSIDE", className="seal-viewer-hint-strong"),
                            html.Span("or", className="seal-viewer-hint-or"),
                            html.Span("PRESS ESC", className="seal-viewer-hint-strong"),
                            html.Span("to return to the dashboard.", className="seal-viewer-hint-tail"),
                        ],
                        className="seal-viewer-hint",
                    ),
                ],
                className="seal-viewer-card",
                role="dialog",
                **{
                    "aria-modal": "true",
                    "aria-label": "MPLADS AI Monitor seal preview",
                    "tabIndex": "-1",
                },
            ),
        ],
        id="seal-viewer",
        className="seal-viewer is-hidden",
    )







# ============================================================
# SAFE / DATA HELPERS
# ============================================================

def api_get(endpoint: str, params: dict[str, Any] | None = None) -> Any:
    response = requests.get(
        f"{API_BASE}{endpoint}",
        params=params or {},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def api_post(endpoint: str, payload: dict[str, Any]) -> Any:
    url = endpoint if endpoint.startswith("http") else f"{API_BASE}{endpoint}"
    response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()




def _json_compact(value: Any, *, max_list: int = 60, max_string: int = 5000, depth: int = 0, max_depth: int = 5) -> Any:
    """Make dashboard context JSON-safe without silently dropping the schema.

    Scalar values and dictionary keys are retained. Large row collections are
    sampled from the front so the LLM receives shape + representative evidence
    without exceeding a practical request size. The full analytical payload is
    still attempted separately when it fits the configured budget.
    """
    if depth > max_depth:
        return str(value)[:max_string]
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value if not isinstance(value, str) else value[:max_string]
    if isinstance(value, dict):
        return {
            str(k): _json_compact(v, max_list=max_list, max_string=max_string, depth=depth + 1, max_depth=max_depth)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        items = list(value)
        out = [
            _json_compact(v, max_list=max_list, max_string=max_string, depth=depth + 1, max_depth=max_depth)
            for v in items[:max_list]
        ]
        if len(items) > max_list:
            out.append({"_truncated": True, "original_count": len(items), "shown": max_list})
        return out
    return str(value)[:max_string]
















_STATE_ALIASES = {
    "up": "Uttar Pradesh", "u p": "Uttar Pradesh", "uttar pradesh": "Uttar Pradesh",
    "mp": "Madhya Pradesh", "madhya pradesh": "Madhya Pradesh",
    "ap": "Andhra Pradesh", "andhra pradesh": "Andhra Pradesh",
    "uk": "Uttarakhand", "uttarakhand": "Uttarakhand",
    "wb": "West Bengal", "west bengal": "West Bengal",
    "tn": "Tamil Nadu", "tamil nadu": "Tamil Nadu",
    "hp": "Himachal Pradesh", "himachal pradesh": "Himachal Pradesh",
    "jk": "Jammu and Kashmir", "jammu and kashmir": "Jammu and Kashmir",
    "j&k": "Jammu and Kashmir",
    "odisha": "Odisha", "orissa": "Odisha",
    "dl": "Delhi", "delhi": "Delhi", "nct delhi": "Delhi",
    "ts": "Telangana", "telangana": "Telangana",
    "tg": "Telangana",
    "jharkhand": "Jharkhand", "jhar": "Jharkhand",
    "chhattisgarh": "Chhattisgarh", "cg": "Chhattisgarh",
    "uttaranchal": "Uttarakhand",
    "bengal": "West Bengal",
    "kerala": "Kerala", "karnataka": "Karnataka", "tamil nadu": "Tamil Nadu",
    "maharashtra": "Maharashtra", "gujarat": "Gujarat", "rajasthan": "Rajasthan",
    "bihar": "Bihar", "punjab": "Punjab", "haryana": "Haryana",
    "assam": "Assam", "goa": "Goa", "sikkim": "Sikkim", "tripura": "Tripura",
    "manipur": "Manipur", "meghalaya": "Meghalaya", "mizoram": "Mizoram",
    "nagaland": "Nagaland", "arunachal": "Arunachal Pradesh", "arunachal pradesh": "Arunachal Pradesh",
}



















def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(safe_float(value, float(default))))
    except (TypeError, ValueError):
        return default


def pct(value: Any) -> str:
    return "—" if value is None else f"{safe_float(value):,.1f}%"


def inr(value: Any) -> str:
    amount = safe_float(value, float("nan"))
    if not math.isfinite(amount):
        return "—"
    sign = "-" if amount < 0 else ""
    amount = abs(amount)
    if amount >= 1e7:
        return f"{sign}₹{amount / 1e7:,.2f} Cr"
    if amount >= 1e5:
        return f"{sign}₹{amount / 1e5:,.2f} L"
    return f"{sign}₹{amount:,.0f}"


def clean_options(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return sorted(result, key=str.casefold)


def option_values(values: list[str]) -> list[dict[str, str]]:
    return [{"label": value, "value": value} for value in values]


def build_filter_params(
    state: str | None,
    district: str | None,
    mp: str | None,
    constituency: str | None,
    category: str | None,
    status: str | None,
    risk: str | None,
    completion: str | None,
    search: str | None,
    min_sanction: float | None,
    max_sanction: float | None,
    risk_range: list[float] | None,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    if risk_range:
        params["min_risk"] = risk_range[0]
        params["max_risk"] = risk_range[1]

    values = {
        "state": state,
        "district": district,
        "mp": mp,
        "constituency": constituency,
        "work_category": category,
        "work_status": status,
        "risk_category": risk,
        "completion_status": completion,
    }

    for key, value in values.items():
        if value and value != "All":
            params[key] = value

    if search and search.strip():
        params["search"] = search.strip()

    if min_sanction and min_sanction > 0:
        params["min_sanction"] = min_sanction

    if max_sanction and max_sanction > 0:
        params["max_sanction"] = max_sanction

    return params


def to_polars(records: Any) -> pl.DataFrame:
    """Build a Polars frame from heterogeneous API records safely.

    Work-queue payloads can mix nulls, ints, floats and strings in the same
    column. Default schema inference is too short for that, so we:
    1. Prefer from_dicts with full-schema inference
    2. Fall back to string-coerced rows if needed
    """
    if not isinstance(records, list) or not records:
        return pl.DataFrame()
    rows = [dict(item) for item in records if isinstance(item, dict)]
    if not rows:
        return pl.DataFrame()
    try:
        return pl.from_dicts(rows, infer_schema_length=None)
    except Exception:
        try:
            return pl.DataFrame(rows, infer_schema_length=None)
        except Exception:
            # Last resort: stringify values so the UI never crashes.
            safe_rows = [
                {str(k): ("" if v is None else str(v)) for k, v in row.items()}
                for row in rows
            ]
            try:
                return pl.from_dicts(safe_rows, infer_schema_length=None)
            except Exception:
                return pl.DataFrame()


def records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        return payload["items"]
    if isinstance(payload, list):
        return payload
    return []


# ============================================================
# PLOTLY VISUAL SYSTEM
# ============================================================

def base_figure(title: str, height: int = 380) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title={"text": title, "font": {"family": FONT_HEAD, "size": 17}},
        height=height,
        margin={"l": 14, "r": 18, "t": 58, "b": 34},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": FONT_BODY, "color": "#DDE6ED"},
        hoverlabel={"font": {"family": FONT_BODY}},
        legend={
            "orientation": "h",
            "y": 1.02,
            "x": 0,
            "bgcolor": "rgba(0,0,0,0)",
        },
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#202A34",
        zeroline=False,
        linecolor="#2B3642",
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#202A34",
        zeroline=False,
        linecolor="#2B3642",
    )
    return fig


def empty_figure(message: str, height: int = 380) -> go.Figure:
    fig = base_figure("", height)
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"family": FONT_BODY, "size": 14, "color": "#81909E"},
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def risk_figure(data: Any, title: str) -> go.Figure:
    frame = to_polars(data)
    if frame.is_empty() or not {"risk_category", "count"}.issubset(frame.columns):
        return empty_figure("Risk distribution unavailable")

    rows = frame.to_dicts()
    rows.sort(
        key=lambda row: RISK_ORDER.index(str(row.get("risk_category")))
        if str(row.get("risk_category")) in RISK_ORDER
        else 99
    )

    fig = base_figure(title, 390)
    fig.add_trace(
        go.Bar(
            x=[safe_int(r.get("count")) for r in rows],
            y=[str(r.get("risk_category")) for r in rows],
            orientation="h",
            text=[f"{safe_float(r.get('pct')):.1f}%" for r in rows],
            textposition="outside",
            hovertemplate="%{y}<br>%{x:,} works<br>%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        yaxis={"categoryorder": "array", "categoryarray": RISK_ORDER},
        xaxis_title="Works",
        yaxis_title="",
    )
    return fig


def reasons_figure(data: Any) -> go.Figure:
    frame = to_polars(data)
    if frame.is_empty() or not {"reason", "count"}.issubset(frame.columns):
        return empty_figure("Risk-signal frequency unavailable")

    frame = frame.with_columns(
        pl.col("count").cast(pl.Float64, strict=False).fill_null(0)
    ).sort("count")

    rows = frame.to_dicts()
    fig = base_figure("Detected risk-signal frequency", 390)
    fig.add_trace(
        go.Bar(
            x=[safe_float(r.get("count")) for r in rows],
            y=[str(r.get("reason")) for r in rows],
            orientation="h",
            hovertemplate="%{y}<br>%{x:,.0f} works<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="Works", yaxis_title="")
    return fig


def time_figure(data: Any) -> go.Figure:
    frame = to_polars(data)
    if frame.is_empty() or "financial_year" not in frame.columns:
        return empty_figure("Time-series data unavailable")

    rows = frame.to_dicts()
    x = [row.get("financial_year") for row in rows]

    fig = base_figure("Financial flow across financial years", 400)

    if "sanctioned_amount" in frame.columns:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=[safe_float(r.get("sanctioned_amount")) for r in rows],
                mode="lines+markers",
                name="Sanctioned",
                hovertemplate="FY %{x}<br>Sanctioned: ₹%{y:,.0f}<extra></extra>",
            )
        )

    if "total_expenditure" in frame.columns:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=[safe_float(r.get("total_expenditure")) for r in rows],
                mode="lines+markers",
                name="Recorded expenditure",
                hovertemplate="FY %{x}<br>Expenditure: ₹%{y:,.0f}<extra></extra>",
            )
        )

    fig.update_layout(
        yaxis_title="Amount (₹)",
        hovermode="x unified",
    )
    return fig


def state_figure(data: Any) -> go.Figure:
    frame = to_polars(data)
    required = {"state", "high_or_critical_rate_pct"}
    if frame.is_empty() or not required.issubset(frame.columns):
        return empty_figure("State analytics unavailable", 500)

    frame = (
        frame.with_columns(
            pl.col("high_or_critical_rate_pct")
            .cast(pl.Float64, strict=False)
            .fill_null(0)
        )
        .sort("high_or_critical_rate_pct", descending=True)
        .head(20)
        .sort("high_or_critical_rate_pct")
    )

    rows = frame.to_dicts()
    fig = base_figure("High / critical signal rate by state", 510)
    fig.add_trace(
        go.Bar(
            x=[safe_float(r.get("high_or_critical_rate_pct")) for r in rows],
            y=[str(r.get("state", "Unknown")) for r in rows],
            orientation="h",
            customdata=[
                [safe_int(r.get("works")), safe_int(r.get("completed_works"))]
                for r in rows
            ],
            hovertemplate=(
                "%{y}<br>High/Critical: %{x:.1f}%"
                "<br>Works: %{customdata[0]:,}"
                "<br>Completed: %{customdata[1]:,}<extra></extra>"
            ),
        )
    )
    fig.update_layout(xaxis_title="Rate (%)", yaxis_title="")
    return fig


def sector_figure(data: Any) -> go.Figure:
    frame = to_polars(data)
    if frame.is_empty():
        return empty_figure("Sector analytics unavailable")

    label_col = (
        "sector"
        if "sector" in frame.columns
        else "work_category"
        if "work_category" in frame.columns
        else None
    )
    value_col = (
        "sanctioned_amount"
        if "sanctioned_amount" in frame.columns
        else "works"
        if "works" in frame.columns
        else None
    )

    if not label_col or not value_col:
        return empty_figure("Sector analytics unavailable")

    frame = (
        frame.with_columns(
            pl.col(value_col).cast(pl.Float64, strict=False).fill_null(0)
        )
        .sort(value_col, descending=True)
        .head(12)
    )

    rows = frame.to_dicts()
    fig = base_figure("Portfolio by sector / work category", 400)
    fig.add_trace(
        go.Bar(
            x=[safe_float(r.get(value_col)) for r in rows],
            y=[str(r.get(label_col, "Unknown")) for r in rows],
            orientation="h",
            hovertemplate="%{y}<br>%{x:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title="Amount (₹)" if value_col == "sanctioned_amount" else "Works",
        yaxis_title="",
    )
    return fig


def financial_scatter(data: list[dict[str, Any]]) -> go.Figure:
    frame = to_polars(data)
    required = {"sanction_amount", "total_expenditure"}
    if frame.is_empty() or not required.issubset(frame.columns):
        return empty_figure("Sanction / expenditure fields unavailable", 420)

    rows = frame.to_dicts()
    valid = []

    for row in rows:
        x = safe_float(row.get("sanction_amount"), float("nan"))
        y = safe_float(row.get("total_expenditure"), float("nan"))
        if math.isfinite(x) and math.isfinite(y):
            valid.append((x, y, row))

    if not valid:
        return empty_figure("No numeric financial observations", 420)

    max_value = max(
        max(x for x, _, _ in valid),
        max(y for _, y, _ in valid),
        1.0,
    )

    fig = base_figure("Sanction vs recorded expenditure · current queue", 430)

    fig.add_trace(
        go.Scatter(
            x=[x for x, _, _ in valid],
            y=[y for _, y, _ in valid],
            mode="markers",
            marker={"size": 9, "opacity": 0.78},
            customdata=[
                [
                    row.get("work_uid"),
                    row.get("risk_category"),
                    safe_float(row.get("final_risk_score")),
                ]
                for _, _, row in valid
            ],
            hovertemplate=(
                "Work %{customdata[0]}"
                "<br>Band: %{customdata[1]}"
                "<br>Risk: %{customdata[2]:.1f}"
                "<br>Sanction: ₹%{x:,.0f}"
                "<br>Expenditure: ₹%{y:,.0f}<extra></extra>"
            ),
            name="Works",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[0, max_value],
            y=[0, max_value],
            mode="lines",
            name="1:1 reference",
            line={"dash": "dash"},
        )
    )

    fig.update_layout(
        xaxis_title="Sanction amount (₹)",
        yaxis_title="Recorded expenditure (₹)",
    )
    return fig


def execution_figure(data: list[dict[str, Any]]) -> go.Figure:
    frame = to_polars(data)
    if frame.is_empty():
        return empty_figure("Execution data unavailable")

    fields = {
        "days_open_since_sanction": "Open age",
        "days_since_last_expenditure": "Days since expenditure",
        "days_sanction_to_complete": "Sanction → completion",
    }

    available = [field for field in fields if field in frame.columns]
    if not available:
        return empty_figure("Execution-duration fields unavailable")

    fig = base_figure("Execution-duration distribution · current queue", 410)

    for field in available:
        values = (
            frame.get_column(field)
            .cast(pl.Float64, strict=False)
            .drop_nulls()
            .to_list()
        )
        if values:
            fig.add_trace(
                go.Box(
                    y=values,
                    name=fields[field],
                    boxpoints="outliers",
                    hovertemplate=f"{fields[field]}<br>%{{y:.0f}} days<extra></extra>",
                )
            )

    fig.update_layout(yaxis_title="Days")
    return fig


def component_figure(components: dict[str, Any]) -> go.Figure:
    mapping = [
        ("ml_anomaly_percentile", "ML anomaly"),
        ("financial_risk_score", "Financial"),
        ("execution_risk_score", "Execution"),
        ("duplicate_risk_score", "Similarity"),
        ("data_integrity_risk_score", "Integrity"),
    ]

    pairs = [
        (label, safe_float(components.get(key)))
        for key, label in mapping
        if components.get(key) is not None
    ]

    if not pairs:
        return empty_figure("Component scores unavailable", 300)

    fig = base_figure("Risk decomposition", 310)
    fig.add_trace(
        go.Bar(
            x=[value for _, value in pairs],
            y=[label for label, _ in pairs],
            orientation="h",
            text=[f"{value:.1f}" for _, value in pairs],
            textposition="outside",
            hovertemplate="%{y}: %{x:.1f}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis={"range": [0, 100], "title": "Score (0–100)"},
        yaxis_title="",
    )
    return fig


# ============================================================
# COMPREHENSIVE ANALYTICS CHARTS
# ============================================================

def _numeric_values(frame: pl.DataFrame, column: str) -> list[float]:
    if column not in frame.columns:
        return []
    return [
        safe_float(v, float("nan"))
        for v in frame.get_column(column).cast(pl.Float64, strict=False).to_list()
        if math.isfinite(safe_float(v, float("nan")))
    ]


def risk_basis_figure(analytics: dict[str, Any]) -> go.Figure:
    final = to_polars(analytics.get("risk_final"))
    rule = to_polars(analytics.get("risk_rule"))
    ml = to_polars(analytics.get("risk_ml"))
    if final.is_empty() and rule.is_empty() and ml.is_empty():
        return empty_figure("Risk-basis comparison unavailable")

    def by_band(frame: pl.DataFrame) -> dict[str, float]:
        if frame.is_empty() or "risk_category" not in frame.columns:
            return {}
        return {str(r.get("risk_category")): safe_float(r.get("pct")) for r in frame.to_dicts()}

    datasets = [
        ("Final composite", by_band(final)),
        ("Rule-based", by_band(rule)),
        ("ML anomaly", by_band(ml)),
    ]
    fig = base_figure("Risk-band composition across scoring layers", 420)
    for name, values in datasets:
        if not values:
            continue
        fig.add_trace(go.Bar(
            x=RISK_ORDER,
            y=[values.get(b, 0.0) for b in RISK_ORDER],
            name=name,
            hovertemplate=f"{name}<br>%{{x}}<br>%{{y:.1f}}%<extra></extra>",
        ))
    fig.update_layout(
        barmode="group",
        xaxis_title="Risk band",
        yaxis_title="Share of filtered works (%)",
        yaxis={"range": [0, 100]},
    )
    return fig


def risk_score_histogram(data: Any) -> go.Figure:
    frame = to_polars(data)
    if frame.is_empty() or "final_risk_score" not in frame.columns:
        return empty_figure("Risk-score distribution unavailable")
    vals = _numeric_values(frame, "final_risk_score")
    if not vals:
        return empty_figure("No numeric risk scores in current queue")
    bins = [i for i in range(0, 101, 10)]
    counts = [0] * 10
    for v in vals:
        idx = min(9, max(0, int(v // 10)))
        counts[idx] += 1
    labels = [f"{i}–{i+10}" for i in range(0, 100, 10)]
    fig = base_figure("Final-risk score distribution · priority queue", 390)
    fig.add_trace(go.Bar(x=labels, y=counts, name="Works", hovertemplate="Score %{x}<br>%{y:,} works<extra></extra>"))
    fig.update_layout(xaxis_title="Final risk score", yaxis_title="Works")
    return fig


def completion_risk_figure(data: Any) -> go.Figure:
    frame = to_polars(data)
    required = {"final_risk_score", "utilization_pct"}
    if frame.is_empty() or not required.issubset(frame.columns):
        return empty_figure("Risk vs utilisation relationship unavailable")
    rows=[]
    for r in frame.to_dicts():
        x=safe_float(r.get("utilization_pct"), float("nan")); y=safe_float(r.get("final_risk_score"), float("nan"))
        if math.isfinite(x) and math.isfinite(y):
            rows.append((x,y,r))
    if not rows:
        return empty_figure("No numeric observations for risk vs utilisation")
    fig=base_figure("Risk score vs utilisation", 430)
    fig.add_trace(go.Scatter(
        x=[r[0] for r in rows], y=[r[1] for r in rows], mode="markers", name="Works",
        customdata=[[r[2].get("work_uid"), r[2].get("risk_category")] for r in rows],
        hovertemplate="Work %{customdata[0]}<br>Band: %{customdata[1]}<br>Utilisation: %{x:.1f}%<br>Risk: %{y:.1f}<extra></extra>",
        marker={"size":8,"opacity":0.65},
    ))
    fig.update_layout(xaxis_title="Recorded expenditure / sanction (%)", yaxis_title="Final risk score (0–100)")
    return fig


def priority_risk_figure(data: Any) -> go.Figure:
    frame=to_polars(data)
    required={"final_risk_score","priority_score"}
    if frame.is_empty() or not required.issubset(frame.columns):
        return empty_figure("Priority vs risk relationship unavailable")
    rows=[]
    for r in frame.to_dicts():
        x=safe_float(r.get("final_risk_score"),float("nan")); y=safe_float(r.get("priority_score"),float("nan"))
        if math.isfinite(x) and math.isfinite(y): rows.append((x,y,r))
    if not rows: return empty_figure("No numeric priority observations")
    fig=base_figure("Risk score vs investigation priority",430)
    fig.add_trace(go.Scatter(
        x=[r[0] for r in rows], y=[r[1] for r in rows], mode="markers", name="Works",
        customdata=[[r[2].get("work_uid"),r[2].get("risk_category")] for r in rows],
        hovertemplate="Work %{customdata[0]}<br>Band: %{customdata[1]}<br>Risk: %{x:.1f}<br>Priority: %{y:.1f}<extra></extra>",
        marker={"size":8,"opacity":0.65},
    ))
    fig.update_layout(xaxis_title="Final risk score", yaxis_title="Priority score")
    return fig


def risk_reason_rate_figure(data: Any) -> go.Figure:
    frame=to_polars(data)
    if frame.is_empty() or "reason" not in frame.columns: return empty_figure("Risk-signal rates unavailable")
    value="pct_of_filtered_works" if "pct_of_filtered_works" in frame.columns else "pct_of_all_works" if "pct_of_all_works" in frame.columns else None
    if not value: return empty_figure("Risk-signal rates unavailable")
    frame=frame.with_columns(pl.col(value).cast(pl.Float64,strict=False).fill_null(0)).sort(value).tail(12)
    rows=frame.to_dicts()
    fig=base_figure("Risk-signal prevalence",430)
    fig.add_trace(go.Bar(x=[safe_float(r.get(value)) for r in rows], y=[str(r.get("reason")) for r in rows], orientation="h", hovertemplate="%{y}<br>%{x:.1f}% of filtered works<extra></extra>"))
    fig.update_layout(xaxis_title="Share of filtered works (%)",yaxis_title="")
    return fig


def sector_completion_figure(data: Any) -> go.Figure:
    frame=to_polars(data)
    required={"work_category","completion_rate_pct"}
    if frame.is_empty() or not required.issubset(frame.columns): return empty_figure("Sector completion data unavailable")
    frame=frame.with_columns(pl.col("completion_rate_pct").cast(pl.Float64,strict=False).fill_null(0)).sort("completion_rate_pct").tail(12)
    rows=frame.to_dicts()
    fig=base_figure("Completion rate by work category",430)
    fig.add_trace(go.Bar(x=[safe_float(r.get("completion_rate_pct")) for r in rows],y=[str(r.get("work_category")) for r in rows],orientation="h",hovertemplate="%{y}<br>Completion: %{x:.1f}%<extra></extra>"))
    fig.update_layout(xaxis_title="Completed works (%)",yaxis_title="")
    return fig


def sector_risk_figure(data: Any) -> go.Figure:
    frame=to_polars(data)
    required={"work_category","high_or_critical_rate_pct"}
    if frame.is_empty() or not required.issubset(frame.columns): return empty_figure("Sector risk data unavailable")
    frame=frame.with_columns(pl.col("high_or_critical_rate_pct").cast(pl.Float64,strict=False).fill_null(0)).sort("high_or_critical_rate_pct").tail(12)
    rows=frame.to_dicts()
    fig=base_figure("High / critical signal rate by work category",430)
    fig.add_trace(go.Bar(x=[safe_float(r.get("high_or_critical_rate_pct")) for r in rows],y=[str(r.get("work_category")) for r in rows],orientation="h",hovertemplate="%{y}<br>High/Critical: %{x:.1f}%<extra></extra>"))
    fig.update_layout(xaxis_title="Rate (%)",yaxis_title="")
    return fig


def state_exposure_figure(data: Any) -> go.Figure:
    frame=to_polars(data)
    required={"state","sanctioned_amount"}
    if frame.is_empty() or not required.issubset(frame.columns): return empty_figure("State financial exposure unavailable")
    frame=frame.with_columns(pl.col("sanctioned_amount").cast(pl.Float64,strict=False).fill_null(0)).sort("sanctioned_amount").tail(15)
    rows=frame.to_dicts()
    fig=base_figure("Sanctioned amount by state",480)
    fig.add_trace(go.Bar(x=[safe_float(r.get("sanctioned_amount")) for r in rows],y=[str(r.get("state")) for r in rows],orientation="h",hovertemplate="%{y}<br>Sanctioned: ₹%{x:,.0f}<extra></extra>"))
    fig.update_layout(xaxis_title="Sanctioned amount (₹)",yaxis_title="")
    return fig


def state_completion_figure(data: Any) -> go.Figure:
    frame=to_polars(data)
    required={"state","completion_rate_pct"}
    if frame.is_empty() or not required.issubset(frame.columns): return empty_figure("State completion data unavailable")
    frame=frame.with_columns(pl.col("completion_rate_pct").cast(pl.Float64,strict=False).fill_null(0)).sort("completion_rate_pct").tail(15)
    rows=frame.to_dicts()
    fig=base_figure("Completion rate by state",480)
    fig.add_trace(go.Bar(x=[safe_float(r.get("completion_rate_pct")) for r in rows],y=[str(r.get("state")) for r in rows],orientation="h",hovertemplate="%{y}<br>Completion: %{x:.1f}%<extra></extra>"))
    fig.update_layout(xaxis_title="Completion rate (%)",yaxis_title="")
    return fig


def time_performance_figure(data: Any) -> go.Figure:
    frame=to_polars(data)
    if frame.is_empty() or "financial_year" not in frame.columns: return empty_figure("Financial-year performance unavailable")
    rows=frame.to_dicts(); fig=base_figure("Completion and high/critical rates over time",420)
    if "completion_rate_pct" in frame.columns:
        fig.add_trace(go.Scatter(x=[r.get("financial_year") for r in rows],y=[safe_float(r.get("completion_rate_pct")) for r in rows],mode="lines+markers",name="Completion rate",hovertemplate="FY %{x}<br>Completion: %{y:.1f}%<extra></extra>"))
    if "high_or_critical_rate_pct" in frame.columns:
        fig.add_trace(go.Scatter(x=[r.get("financial_year") for r in rows],y=[safe_float(r.get("high_or_critical_rate_pct")) for r in rows],mode="lines+markers",name="High/Critical rate",hovertemplate="FY %{x}<br>High/Critical: %{y:.1f}%<extra></extra>"))
    fig.update_layout(xaxis_title="Financial year",yaxis_title="Rate (%)",yaxis={"range":[0,100]},hovermode="x unified")
    return fig


def time_risk_figure(data: Any) -> go.Figure:
    frame=to_polars(data)
    if frame.is_empty() or not {"financial_year","median_risk"}.issubset(frame.columns): return empty_figure("Median-risk trend unavailable")
    rows=frame.to_dicts(); fig=base_figure("Median final risk by financial year",400)
    fig.add_trace(go.Scatter(x=[r.get("financial_year") for r in rows],y=[safe_float(r.get("median_risk")) for r in rows],mode="lines+markers",name="Median risk",hovertemplate="FY %{x}<br>Median risk: %{y:.1f}<extra></extra>"))
    fig.update_layout(xaxis_title="Financial year",yaxis_title="Median final risk (0–100)",yaxis={"range":[0,100]})
    return fig


def financial_gap_figure(data: Any) -> go.Figure:
    frame=to_polars(data)
    required={"sanction_amount","total_expenditure"}
    if frame.is_empty() or not required.issubset(frame.columns): return empty_figure("Financial-gap relationship unavailable")
    rows=[]
    for r in frame.to_dicts():
        s=safe_float(r.get("sanction_amount"),float("nan")); e=safe_float(r.get("total_expenditure"),float("nan"))
        if math.isfinite(s) and math.isfinite(e) and s>0: rows.append((s,e,r))
    if not rows: return empty_figure("No valid financial observations")
    fig=base_figure("Sanction, expenditure and financial position",430)
    fig.add_trace(go.Scatter(x=[r[0] for r in rows],y=[r[1] for r in rows],mode="markers",name="Works",customdata=[[r[2].get("work_uid"),r[2].get("risk_category")] for r in rows],hovertemplate="Work %{customdata[0]}<br>Band: %{customdata[1]}<br>Sanction: ₹%{x:,.0f}<br>Expenditure: ₹%{y:,.0f}<extra></extra>",marker={"size":8,"opacity":0.6}))
    maxv=max(max(r[0] for r in rows),max(r[1] for r in rows))
    fig.add_trace(go.Scatter(x=[0,maxv],y=[0,maxv],mode="lines",name="Sanction = expenditure",line={"dash":"dash"}))
    fig.update_layout(xaxis_title="Sanction amount (₹)",yaxis_title="Recorded expenditure (₹)")
    return fig


def utilization_distribution_figure(data: Any) -> go.Figure:
    frame=to_polars(data)
    if frame.is_empty() or "utilization_pct" not in frame.columns: return empty_figure("Utilisation distribution unavailable")
    vals=_numeric_values(frame,"utilization_pct")
    if not vals: return empty_figure("No valid utilisation values")
    bounds=[0,25,50,75,100,125,150,200,300,float("inf")]
    labels=["0–25%","25–50%","50–75%","75–100%","100–125%","125–150%","150–200%","200–300%","300%+"]
    counts=[0]*len(labels)
    for v in vals:
        for i in range(len(bounds)-1):
            if bounds[i] <= v < bounds[i+1]: counts[i]+=1; break
    fig=base_figure("Utilisation distribution · priority queue",400)
    fig.add_trace(go.Bar(x=labels,y=counts,name="Works",hovertemplate="Utilisation %{x}<br>%{y:,} works<extra></extra>"))
    fig.update_layout(xaxis_title="Recorded expenditure / sanction",yaxis_title="Works")
    return fig


def execution_age_figure(data: Any) -> go.Figure:
    frame=to_polars(data)
    if frame.is_empty() or "days_open_since_sanction" not in frame.columns: return empty_figure("Execution-age distribution unavailable")
    vals=_numeric_values(frame,"days_open_since_sanction")
    if not vals: return empty_figure("No valid execution-age values")
    bins=[0,90,180,365,730,1095,1825,float("inf")]
    labels=["0–90","91–180","181–365","366–730","731–1095","1096–1825","1826+"]
    counts=[0]*len(labels)
    for v in vals:
        for i in range(len(bounds:=bins)-1):
            if bounds[i] <= v < bounds[i+1]: counts[i]+=1; break
    fig=base_figure("Open-work age distribution",400)
    fig.add_trace(go.Bar(x=labels,y=counts,name="Works",hovertemplate="Open age %{x} days<br>%{y:,} works<extra></extra>"))
    fig.update_layout(xaxis_title="Days since sanction",yaxis_title="Open works")
    return fig



# ============================================================
# OVERVIEW-SPECIFIC FIGURES
# ============================================================

def overview_status_figure(scope: Any) -> go.Figure:
    completed = safe_int(scope.get("completed_works")) if isinstance(scope, dict) else 0
    open_works = safe_int(scope.get("open_works")) if isinstance(scope, dict) else 0
    total = completed + open_works
    if total <= 0:
        return empty_figure("Project-status data unavailable", 260)

    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels=["Completed", "Open"],
        values=[completed, open_works],
        hole=.72,
        sort=False,
        textinfo="none",
        hovertemplate="%{label}<br>%{value:,} works<br>%{percent}<extra></extra>",
        marker={"line": {"color": "#0B1117", "width": 3}},
    ))
    fig.add_annotation(
        x=.5, y=.54,
        text=f"<b>{total:,}</b>",
        showarrow=False,
        font={"family": FONT_HEAD, "size": 25, "color": "#EEF4F8"},
    )
    fig.add_annotation(
        x=.5, y=.39,
        text="WORKS",
        showarrow=False,
        font={"family": FONT_MONO, "size": 9, "color": "#7E93A3"},
    )
    fig.update_layout(
        title=None,
        height=380,
        margin={"l":12,"r":12,"t":50,"b":16},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family":FONT_BODY,"color":"#DDE7EE"},
        showlegend=True,
        legend={"orientation":"h","y":-0.02,"x":.5,"xanchor":"center"},
    )
    return fig

def overview_utilization_figure(scope: Any) -> go.Figure:
    """Locked-grid gauge — header, arc, value and status on fixed vertical anchors."""
    value = (
        safe_float(scope.get("portfolio_utilization_pct"), float("nan"))
        if isinstance(scope, dict)
        else float("nan")
    )
    if not math.isfinite(value):
        return empty_figure("Utilisation unavailable", 440)
    value = max(0.0, min(value, 100.0))

    if value < 25:
        primary, status, status_color = "#FF7B7B", "UNDER-UTILISED", "#FF9D9D"
    elif value < 50:
        primary, status, status_color = "#FFC66B", "MODERATE", "#FFD89C"
    elif value < 75:
        primary, status, status_color = "#8FD0FF", "HEALTHY", "#B5E0FF"
    else:
        primary, status, status_color = "#7FE1A5", "STRONG", "#A8F0C0"

    delta = value - 50.0
    delta_color = "#7FE1A5" if delta >= 0 else "#FF7B7B"
    delta_sign  = "▲" if delta >= 0 else "▼"

    fig = go.Figure(
        go.Indicator(
            mode="gauge",
            value=value,
            gauge={
                "shape": "angular",
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 0,
                    "tickvals": [0, 25, 50, 75, 100],
                    "tickfont": {"family": FONT_MONO, "size": 10, "color": "#6E8496"},
                    "ticklen": 6,
                    "tickcolor": "#2A3E50",
                },
                "bar": {
                    "color": primary,
                    "thickness": 0.22,
                    "line": {"color": primary, "width": 0},
                },
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 25],   "color": "rgba(255,123,123,.04)"},
                    {"range": [25, 50],  "color": "rgba(255,198,107,.04)"},
                    {"range": [50, 75],  "color": "rgba(143,208,255,.04)"},
                    {"range": [75, 100], "color": "rgba(127,225,165,.04)"},
                ],
                "threshold": {
                    "line": {"color": "rgba(255,255,255,.85)", "width": 2},
                    "thickness": 0.9,
                    "value": value,
                },
            },
            # ── ARC ANCHOR: apex at 0.80, well clear of the header stack ──
            domain={"x": [0.10, 0.90], "y": [0.30, 0.80]},
        )
    )

    # ── Header stack — anchored top-down so it always sits ABOVE the arc ──
    fig.add_annotation(
        x=0.5, y=0.995, xref="paper", yref="paper",
        text="<b>Portfolio Utilisation</b>",
        showarrow=False,
        font={"family": FONT_HEAD, "size": 16, "color": "#E8F1F8"},
        yanchor="top",
    )
    fig.add_annotation(
        x=0.5, y=0.925, xref="paper", yref="paper",
        text="EXPENDITURE ÷ SANCTIONED",
        showarrow=False,
        font={"family": FONT_MONO, "size": 9, "color": "#6E8496"},
        yanchor="top",
    )

    # ── Value — pinned inside the arc bowl ──
    fig.add_annotation(
        x=0.5, y=0.55, xref="paper", yref="paper",
        text=(
            f"<span style='font-size:52px;font-weight:700;color:#F5FAFE;"
            f"font-family:Bahnschrift,Segoe UI,Arial'>{value:.1f}</span>"
            f"<span style='font-size:22px;color:#6E8496;font-weight:400'>%</span>"
        ),
        showarrow=False, align="center",
    )

    # ── Status + delta — one compact, centred row ──
    fig.add_annotation(
        x=0.5, y=0.14, xref="paper", yref="paper",
        text=(
            f"<span style='color:{status_color};font-weight:700;letter-spacing:.22em'>"
            f"{status}</span>"
            f"<span style='color:#3A4E5E'>     ·     </span>"
            f"<span style='color:{delta_color};font-weight:700'>"
            f"{delta_sign} {abs(delta):.1f} pts</span>"
            f"<span style='color:#5E7484'> vs 50% parity</span>"
        ),
        showarrow=False,
        font={"family": FONT_MONO, "size": 10},
    )

    fig.update_layout(
        height=440,
        margin={"l": 26, "r": 26, "t": 44, "b": 18},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": FONT_BODY, "color": "#DDE7EE"},
    )
    return fig


def overview_risk_spectrum_figure(data: Any) -> go.Figure:
    frame = to_polars(data)
    if frame.is_empty() or not {"risk_category","count"}.issubset(frame.columns):
        return empty_figure("Risk spectrum unavailable", 380)
    lookup = {str(r.get("risk_category")): safe_int(r.get("count")) for r in frame.to_dicts()}
    total = sum(lookup.values())
    if total <= 0:
        return empty_figure("Risk spectrum unavailable", 380)
    fig = go.Figure()
    for band in RISK_ORDER:
        count = lookup.get(band,0)
        fig.add_trace(go.Bar(
            x=[count], y=["Current scope"], orientation="h", name=band,
            customdata=[[count, count/total*100]],
            hovertemplate=f"{band}<br>%{{customdata[0]:,}} works<br>%{{customdata[1]:.1f}}% of scope<extra></extra>",
        ))
    fig.update_layout(
        barmode="stack",
        title={"text":"Risk spectrum · same filtered population", "font":{"family":FONT_HEAD,"size":16}},
        height=285,
        margin={"l":14,"r":18,"t":52,"b":35},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family":FONT_BODY,"color":"#DDE7EE"},
        legend={"orientation":"h","y":1.08,"x":0},
        xaxis={"showgrid":True,"gridcolor":"#202A34","title":"Works"},
        yaxis={"showgrid":False,"title":""},
    )
    return fig


def overview_state_attention_figure(data: Any) -> go.Figure:
    frame = to_polars(data)
    required={"state","completion_rate_pct","high_or_critical_rate_pct","works"}
    if frame.is_empty() or not required.issubset(frame.columns):
        return empty_figure("State comparison unavailable", 390)
    frame=(frame.with_columns([
        pl.col("completion_rate_pct").cast(pl.Float64,strict=False).fill_null(0),
        pl.col("high_or_critical_rate_pct").cast(pl.Float64,strict=False).fill_null(0),
        pl.col("works").cast(pl.Float64,strict=False).fill_null(0),
    ]).filter(pl.col("works")>0).sort("works",descending=True).head(35))
    if frame.is_empty(): return empty_figure("State comparison unavailable",390)
    rows=frame.to_dicts()
    fig=base_figure("State execution × signal landscape",390)
    fig.add_trace(go.Scatter(
        x=[safe_float(r.get("completion_rate_pct")) for r in rows],
        y=[safe_float(r.get("high_or_critical_rate_pct")) for r in rows],
        mode="markers+text",
        text=[str(r.get("state","Unknown"))[:18] for r in rows],
        textposition="top center",
        textfont={"size":8,"color":"#91A5B4"},
        marker={"size":[max(7,min(25,7+math.sqrt(max(0,safe_float(r.get("works")))/10))) for r in rows],"line":{"width":1,"color":"#8FD0FF"}},
        customdata=[[safe_int(r.get("works")),safe_float(r.get("completion_rate_pct")),safe_float(r.get("high_or_critical_rate_pct"))] for r in rows],
        hovertemplate="%{text}<br>Works: %{customdata[0]:,}<br>Completion: %{customdata[1]:.1f}%<br>High/Critical: %{customdata[2]:.1f}%<extra></extra>",
        showlegend=False,
    ))
    fig.add_vline(x=50,line_dash="dot",line_color="#3A4B58")
    fig.add_hline(y=10,line_dash="dot",line_color="#3A4B58")
    fig.update_layout(xaxis_title="Completion rate (%)",yaxis_title="High / Critical signal rate (%)",hovermode="closest")
    return fig


def overview_financial_flow_figure(data: Any) -> go.Figure:
    """Line chart of sanctioned vs expenditure by financial year."""
    frame = to_polars(data)
    if frame.is_empty():
        return empty_figure("Financial flow unavailable", 420)

    # Normalise alternate column names from different API builders.
    rename = {}
    if "financial_year" not in frame.columns and "analysis_year" in frame.columns:
        rename["analysis_year"] = "financial_year"
    if "financial_year" not in frame.columns and "fy" in frame.columns:
        rename["fy"] = "financial_year"
    if "sanctioned_amount" not in frame.columns and "sanction_amount" in frame.columns:
        rename["sanction_amount"] = "sanctioned_amount"
    if "total_expenditure" not in frame.columns and "expenditure" in frame.columns:
        rename["expenditure"] = "total_expenditure"
    if rename:
        frame = frame.rename(rename)

    required = {"financial_year", "sanctioned_amount", "total_expenditure"}
    if not required.issubset(set(frame.columns)):
        return empty_figure("Financial flow unavailable", 420)

    frame = (
        frame.with_columns(
            [
                pl.col("sanctioned_amount").cast(pl.Float64, strict=False).fill_null(0),
                pl.col("total_expenditure").cast(pl.Float64, strict=False).fill_null(0),
                pl.col("financial_year").cast(pl.Utf8, strict=False).fill_null("Unknown"),
            ]
        )
        .group_by("financial_year", maintain_order=True)
        .agg(
            pl.col("sanctioned_amount").sum(),
            pl.col("total_expenditure").sum(),
        )
        .sort("financial_year")
    )
    rows = frame.to_dicts()
    if not rows:
        return empty_figure("Financial flow unavailable", 420)

    x = [str(r.get("financial_year")) for r in rows]
    fig = base_figure("Financial flow by financial year", 420)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=[safe_float(r.get("sanctioned_amount")) for r in rows],
            name="Sanctioned",
            mode="lines+markers",
            line={"width": 3, "color": "#8FD0FF"},
            marker={"size": 8},
            hovertemplate="FY %{x}<br>Sanctioned: ₹%{y:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=[safe_float(r.get("total_expenditure")) for r in rows],
            name="Expenditure",
            mode="lines+markers",
            line={"width": 3, "color": "#F5C56B"},
            marker={"size": 8},
            hovertemplate="FY %{x}<br>Expenditure: ₹%{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(yaxis_title="Amount (₹)", hovermode="x unified", height=420)
    return fig


def overview_signal_figure(data: Any) -> go.Figure:
    """Horizontal bars of rule-signal prevalence (accepts multiple key schemas)."""
    frame = to_polars(data)
    if frame.is_empty():
        return empty_figure("Signal profile unavailable", 420)

    # API may send reason/pct_of_filtered_works OR label/rate_pct
    cols = set(frame.columns)
    if "reason" not in cols and "label" in cols:
        frame = frame.rename({"label": "reason"})
    if "pct_of_filtered_works" not in frame.columns:
        if "rate_pct" in frame.columns:
            frame = frame.rename({"rate_pct": "pct_of_filtered_works"})
        elif "pct" in frame.columns:
            frame = frame.rename({"pct": "pct_of_filtered_works"})

    if not {"reason", "count"}.issubset(set(frame.columns)):
        return empty_figure("Signal profile unavailable", 420)

    if "pct_of_filtered_works" not in frame.columns:
        total = max(sum(safe_int(r.get("count")) for r in frame.to_dicts()), 1)
        frame = frame.with_columns(
            (pl.col("count").cast(pl.Float64, strict=False).fill_null(0) / total * 100).alias(
                "pct_of_filtered_works"
            )
        )

    frame = frame.with_columns(
        pl.col("pct_of_filtered_works").cast(pl.Float64, strict=False).fill_null(0)
    ).sort("pct_of_filtered_works")
    rows = frame.to_dicts()
    fig = base_figure("Signal prevalence · review flags, not findings", 420)
    fig.add_trace(
        go.Bar(
            x=[safe_float(r.get("pct_of_filtered_works")) for r in rows],
            y=[str(r.get("reason")) for r in rows],
            orientation="h",
            marker={"color": "#5B9FD4"},
            customdata=[[safe_int(r.get("count"))] for r in rows],
            hovertemplate="%{y}<br>%{x:.1f}% of filtered works<br>Count: %{customdata[0]:,}<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="Share of filtered works (%)", yaxis_title="", height=420)
    return fig


# ============================================================
# UI COMPONENTS
# ============================================================

def kpi_card(
    label: str,
    value: str,
    hint: str = "",
    tone: str = "",
) -> html.Div:
    return html.Div(
        [
            html.Div(label, className="kpi-label"),
            html.Div(value, className="kpi-value"),
            html.Div(hint, className="kpi-hint") if hint else None,
        ],
        className=f"kpi-card {tone}".strip(),
    )


def section_header(kicker: str, title: str, note: str = "") -> html.Div:
    return html.Div(
        [
            html.Div(kicker.upper(), className="section-kicker"),
            html.H2(title, className="section-title"),
            html.Div(note, className="section-note") if note else None,
        ],
        className="section-header",
    )


def empty_panel(message: str) -> html.Div:
    return html.Div(message, className="empty-panel")


def queue_table(
    data: list[dict[str, Any]],
    selectable: bool = False,
    table_id: str = "queue-table",
) -> dash_table.DataTable:
    if not data:
        return dash_table.DataTable(
            data=[],
            columns=[{"name": "Status", "id": "status"}],
            style_table={"overflowX": "auto"},
            style_header={"backgroundColor": "#151D26", "color": "#A7B5C2"},
            style_cell={
                "backgroundColor": "#0F151C",
                "color": "#DCE5EC",
                "padding": "10px",
            },
        )

    fields = [
        "priority_rank",
        "work_uid",
        "work",
        "state",
        "ida",
        "mp",
        "constituency",
        "work_category",
        "financial_year",
        "sanction_amount",
        "total_expenditure",
        "utilization_pct",
        "days_open_since_sanction",
        "final_risk_score",
        "risk_category",
        "priority_score",
        "confidence_score",
        "primary_risk_reason",
    ]

    frame = to_polars(data)
    available = [field for field in fields if field in frame.columns]
    frame = frame.select(available)

    labels = {
        "priority_rank": "#",
        "work_uid": "Work ID",
        "work": "Work",
        "state": "State",
        "ida": "District / IDA",
        "mp": "MP",
        "constituency": "Constituency",
        "work_category": "Category",
        "financial_year": "FY",
        "sanction_amount": "Sanction",
        "total_expenditure": "Expenditure",
        "utilization_pct": "Utilisation",
        "days_open_since_sanction": "Open age",
        "final_risk_score": "Risk",
        "risk_category": "Band",
        "priority_score": "Priority",
        "confidence_score": "Confidence",
        "primary_risk_reason": "Primary signal",
    }

    return dash_table.DataTable(
        id=table_id,
        data=frame.to_dicts(),
        columns=[
            {"name": labels.get(field, field), "id": field}
            for field in available
        ],
        page_action="none",
        sort_action="native",
        filter_action="native",
        row_selectable="single" if selectable else False,
        style_table={
            "overflowX": "auto",
            "maxHeight": "600px",
            "overflowY": "auto",
        },
        style_header={
            "backgroundColor": "#151D26",
            "color": "#A7B5C2",
            "fontFamily": FONT_BODY,
            "fontWeight": "700",
            "border": "1px solid #27323D",
            "position": "sticky",
            "top": 0,
            "zIndex": 1,
        },
        style_cell={
            "backgroundColor": "#0F151C",
            "color": "#DCE5EC",
            "fontFamily": FONT_BODY,
            "fontSize": "12px",
            "padding": "10px",
            "border": "1px solid #202A34",
            "textAlign": "left",
            "maxWidth": "300px",
            "whiteSpace": "normal",
        },
        style_data_conditional=[
            {
                "if": {"filter_query": '{risk_category} = "CRITICAL"'},
                "backgroundColor": "#24171A",
            },
            {
                "if": {"filter_query": '{risk_category} = "HIGH"'},
                "backgroundColor": "#211D16",
            },
            {
                "if": {"column_id": "final_risk_score"},
                "fontWeight": "700",
            },
            {
                "if": {"column_id": "priority_score"},
                "fontWeight": "700",
            },
        ],
        sort_by=(
            [{"column_id": "priority_score", "direction": "desc"}]
            if "priority_score" in available
            else []
        ),
    )



def collapsible_queue_panel(
    works: list[dict[str, Any]],
    *,
    table_id: str,
    title: str = "Priority investigation queue",
    subtitle: str | None = None,
    selectable: bool = False,
    open_by_default: bool = False,
) -> html.Details:
    """Big interactive drawer — table hidden by default, scrollable when opened."""
    n = len(works) if isinstance(works, list) else 0
    sub = subtitle or f"{n:,} review candidates · capped queue · scroll inside"
    summary = html.Summary(
        [
            html.Div(
                [
                    html.Span("📋", className="queue-drawer-emoji"),
                    html.Div(
                        [
                            html.Div(title, className="queue-drawer-title"),
                            html.Div(sub, className="queue-drawer-sub"),
                        ],
                        className="queue-drawer-text",
                    ),
                ],
                className="queue-drawer-left",
            ),
            html.Div(
                [
                    html.Span(f"{n:,}", className="queue-drawer-count"),
                    html.Span("records", className="queue-drawer-count-label"),
                    html.Span("▸", className="queue-drawer-chevron"),
                ],
                className="queue-drawer-right",
            ),
        ],
        className="queue-drawer-summary",
    )
    body = html.Div(
        [
            html.Div(
                "Select a row for work-level review where enabled. Charts above always use the full filtered population.",
                className="queue-drawer-hint",
            ),
            html.Div(
                queue_table(works, selectable=selectable, table_id=table_id),
                className="queue-drawer-scroll",
            ),
        ],
        className="queue-drawer-body",
    )
    return html.Details(
        [summary, body],
        className="queue-drawer",
        open=open_by_default,
    )



# ============================================================
# APPLICATION
# ============================================================

app = Dash(
    __name__,
    title="MPLADS AI Monitor",
    update_title="MPLADS AI Monitor · updating",
    suppress_callback_exceptions=False,
    serve_locally=True,
)
server = app.server

app.index_string = r"""
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}

<script>
(function(){
  document.addEventListener("keydown", function(event){
    if(event.key !== "Escape") return;

    const sealClose = document.getElementById("seal-viewer-close");
    const sealViewer = document.getElementById("seal-viewer");
    if(sealClose && sealViewer?.classList.contains("is-open")){
      sealClose.click();
      return;
    }

    const teamClose = document.getElementById("team-info-close");
    const teamModal = document.getElementById("team-info-modal");
    if(teamClose && teamModal?.classList.contains("is-open")){
      teamClose.click();
    }
  });
  document.addEventListener("click", function(event){
    var launcher=event.target.closest && event.target.closest("#copilot-launcher");
    var fresh=event.target.closest && event.target.closest("#copilot-new");
    if(launcher||fresh){
      setTimeout(function(){
        var input=document.getElementById("copilot-input");
        if(input) input.focus();
      },220);
    }
  });
})();
</script>

</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>
"""

app.layout = html.Div(
    [
        dcc.Store(id="health-store"),
        dcc.Store(id="options-store"),
        dcc.Store(id="national-store"),
        dcc.Store(id="filtered-store"),
        dcc.Store(id="works-store"),
        dcc.Store(id="sidebar-collapsed", data=False),
        dcc.Store(id="analytics-store"),
        dcc.Store(id="copilot-history", data=[]),
        dcc.Store(id="copilot-chart-context", data={}),
        dcc.Download(id="download-queue"),

        html.Div(
            [
                html.Div(className="boot-noise"),
                html.Div(className="boot-orbit boot-orbit-a"),
                html.Div(className="boot-orbit boot-orbit-b"),
                html.Div(
                    [
                        html.Div("MPLADS AI MONITOR", className="boot-kicker"),
                        html.Div("PUBLIC-WORKS INTELLIGENCE", className="boot-title"),
                        html.Div(
                            [
                                html.Span("●", className="boot-live-dot"),
                                html.Span("Initializing evidence workspace", className="boot-status"),
                            ],
                            className="boot-status-row",
                        ),
                        html.Div(
                            [
                                html.Span("Crafted for public-works transparency", className="boot-caption"),
                                html.Span("TEAM FRESH MINDS", className="boot-team"),
                            ],
                            className="boot-meta-row",
                        ),
                        html.Div(
                            html.Div(className="boot-progress-fill"),
                            className="boot-progress",
                        ),
                    ],
                    className="boot-card",
                ),
            ],
            id="initial-loader",
            className="initial-loader",
        ),

        dcc.Interval(
            id="startup",
            interval=1000,
            n_intervals=0,
            max_intervals=1,
        ),

        html.Div(
            [
                # ---------------- SIDEBAR ----------------
                html.Aside(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.Span(className="brand-seal-aura"),
                                                        html.Span(className="brand-seal-chakra"),
                                                        html.Span(className="brand-seal-ring ring-a"),
                                                        html.Span(className="brand-seal-ring ring-b"),
                                                        html.Span(className="brand-seal-dome"),
                                                        html.Span(className="brand-seal-plinth"),
                                                        html.Span(
                                                            [
                                                                html.Span(className="seal-fig fig-saffron"),
                                                                html.Span(className="seal-fig fig-blue"),
                                                                html.Span(className="seal-fig fig-green"),
                                                            ],
                                                            className="brand-seal-triad",
                                                        ),
                                                    ],
                                                    id="brand-seal",
                                                    className="brand-seal",
                                                    n_clicks=0,
                                                    title="MPLADS AI Monitor · Government of India",
                                                ),
                                                html.Div(
                                                    [
                                                        html.Div("MPLADS", className="brand-mark"),
                                                        html.Div(
                                                            ["AI MONITOR", html.Span(className="brand-live-dot")],
                                                            className="brand-sub",
                                                        ),
                                                    ],
                                                    className="brand-text-wrap",
                                                ),
                                            ],
                                            className="brand-lockup",
                                        ),
                                        html.Button(
                                            "«",
                                            id="sidebar-toggle",
                                            n_clicks=0,
                                            className="sidebar-toggle",
                                            title="Collapse / expand filters",
                                        ),
                                    ],
                                    className="brand-top",
                                ),
                                html.Div(
                                    "MONITOR · DETECT · EXPLAIN · REVIEW",
                                    className="brand-micro",
                                ),
                                html.Div(className="brand-accent-line"),
                            ],
                            className="brand",
                        ),

                        html.Div("VIEW", className="control-label"),
                        dcc.Dropdown(
                            id="view-mode",
                            options=option_values(
                                [
                                    "Overview",
                                    "Risk Intelligence",
                                    "Financial & Execution",
                                    "Geography",
                                    "Work Explorer",
                                    "Methodology",
                                ]
                            ),
                            value="Overview",
                            clearable=False,
                            className="dark-dropdown",
                        ),

                        html.Div("SCOPE", className="control-label"),
                        html.Div("State", className="filter-mini-label"),
                        dcc.Dropdown(
                            id="state-filter",
                            options=[{"label": "All States", "value": "All"}],
                            value="All",
                            clearable=False,
                            className="dark-dropdown",
                        ),
                        html.Div("District / IDA", className="filter-mini-label"),
                        dcc.Dropdown(
                            id="district-filter",
                            options=[{"label": "All Districts", "value": "All"}],
                            value="All",
                            clearable=False,
                            className="dark-dropdown",
                        ),
                        html.Div("MP", className="filter-mini-label"),
                        dcc.Dropdown(
                            id="mp-filter",
                            options=[{"label": "All MPs", "value": "All"}],
                            value="All",
                            clearable=False,
                            className="dark-dropdown",
                        ),
                        html.Div("Constituency", className="filter-mini-label"),
                        dcc.Dropdown(
                            id="constituency-filter",
                            options=[{"label": "All Constituencies", "value": "All"}],
                            value="All",
                            clearable=False,
                            className="dark-dropdown",
                        ),
                        html.Div("Category", className="filter-mini-label"),
                        dcc.Dropdown(
                            id="category-filter",
                            options=[{"label": "All Categories", "value": "All"}],
                            value="All",
                            clearable=False,
                            className="dark-dropdown",
                        ),
                        html.Div("Status", className="filter-mini-label"),
                        dcc.Dropdown(
                            id="status-filter",
                            options=[{"label": "All Status", "value": "All"}],
                            value="All",
                            clearable=False,
                            className="dark-dropdown",
                        ),
                        html.Div("Risk band", className="filter-mini-label"),
                        dcc.Dropdown(
                            id="risk-filter",
                            options=[{"label": "All Risk Levels", "value": "All"}],
                            value="All",
                            clearable=False,
                            className="dark-dropdown",
                        ),
                        html.Div("Completion", className="filter-mini-label"),
                        dcc.Dropdown(
                            id="completion-filter",
                            options=option_values(["All", "Completed", "Open"]),
                            value="All",
                            clearable=False,
                            className="dark-dropdown",
                        ),

                        html.Div("SEARCH", className="control-label"),
                        dcc.Input(
                            id="search-filter",
                            type="search",
                            placeholder="Work, ID, MP, constituency…",
                            className="dark-input",
                        ),

                        html.Div("RISK / VALUE", className="control-label"),
                        dcc.RangeSlider(
                            id="risk-range",
                            min=0,
                            max=100,
                            step=1,
                            value=[0, 100],
                            marks={
                                0: "0",
                                25: "25",
                                50: "50",
                                75: "75",
                                100: "100",
                            },
                            tooltip={"placement": "bottom"},
                        ),

                        dcc.Input(
                            id="min-sanction",
                            type="number",
                            min=0,
                            step=100000,
                            value=0,
                            placeholder="Minimum sanction ₹",
                            className="dark-input",
                        ),
                        dcc.Input(
                            id="max-sanction",
                            type="number",
                            min=0,
                            step=100000,
                            value=0,
                            placeholder="Maximum sanction ₹",
                            className="dark-input",
                        ),

                        html.Button(
                            [html.Span("↻", className="btn-icon"), html.Span("Refresh data", className="btn-label")],
                            id="refresh-button",
                            className="action-button action-primary",
                            title="Refresh data",
                        ),

                        html.Div("WORK LOOKUP", className="control-label"),
                        dcc.Input(
                            id="work-id-input",
                            type="text",
                            placeholder="Enter Work ID…",
                            debounce=True,
                            className="dark-input",
                        ),
                        html.Button(
                            [html.Span("◎", className="btn-icon"), html.Span("Open profile", className="btn-label")],
                            id="work-lookup-button",
                            className="action-button",
                            title="Open work profile",
                        ),
                        html.Button(
                            [html.Span("▶", className="btn-icon"), html.Span("60s Demo", className="btn-label")],
                            id="demo-work-button",
                            className="action-button action-primary",
                            title="Open the current top-priority work and generate its live evidence alert card",
                        ),
                        html.Button(
                            [html.Span("⇩", className="btn-icon"), html.Span("Download queue", className="btn-label")],
                            id="download-button",
                            className="action-button",
                            title="Download current queue",
                        ),

                        html.Div(
                            html.Div(
                                [
                                    html.Span(className="connection-pulse is-checking"),
                                    html.Span("FASTAPI", className="connection-state-label"),
                                    html.Span("CHECKING…", className="connection-state-value"),
                                ],
                                className="connection-state-content",
                            ),
                            id="connection-state",
                            className="connection-state connection-checking",
                            title="Checking FastAPI health on 127.0.0.1:8000",
                        ),

                        html.Div(
                            [
                                html.Div("FASTAPI", className="meta-label"),
                                html.Div(API_BASE, className="meta-value"),
                                html.Div("DASH", className="meta-label"),
                                html.Div(
                                    f"127.0.0.1:{DASH_PORT}",
                                    className="meta-value",
                                ),
                            ],
                            className="sidebar-meta",
                        ),
                    ],
                    id="sidebar-shell",
                    className="sidebar",
                ),

                # ---------------- MAIN ----------------
                html.Main(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            "PUBLIC WORKS · ANALYTICS CONTROL CENTER",
                                            className="eyebrow",
                                        ),
                                        html.H1(
                                            "MPLADS AI Monitor",
                                            className="page-title",
                                        ),
                                        html.P(
                                            "Explainable monitoring of financial, execution, consistency and anomaly signals.",
                                            className="page-subtitle",
                                        ),
                                    ]
                                ),
                                html.Div(
                                    id="runtime-badges",
                                    className="runtime-badges",
                                ),
                            ],
                            className="topbar",
                        ),

                        html.Div(
                            id="scope-banner",
                            className="scope-banner",
                        ),

                        html.Div(
                            id="kpi-grid",
                            className="kpi-grid",
                        ),

                        html.Div(
                            id="secondary-kpi-grid",
                            className="secondary-kpi-grid",
                        ),

                        dcc.Loading(
                            id="analytics-loading",
                            type="circle",
                            color="#8FD0FF",
                            delay_show=150,
                            delay_hide=120,
                            children=html.Div(
                                id="active-view",
                                className="view-container",
                            ),
                        ),

                        html.Div(
                            id="selected-work-detail",
                            className="detail-shell",
                        ),
                        html.Div(
                            [
                                html.Div(className="team-footer-glow"),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                model_logo_interactive(),
                                            ],
                                            className="team-brand-marks",
                                        ),
                                        html.Div(id="_logo-click-sink", style={"display": "none"}),
                                        html.Div(
                                            [
                                                html.Div(
                                                    "MPLADS AI MONITOR · SIGNALS SUPPORT REVIEW & VERIFICATION — NEVER ADJUDICATION.",
                                                    className="footer-note-line",
                                                ),
                                                html.Div(
                                                    "CRAFTED FOR PUBLIC-WORKS TRANSPARENCY",
                                                    className="team-eyebrow",
                                                ),
                                                html.Div(
                                                    [
                                                        html.Span("Engineered with", className="team-credit-text"),
                                                        html.Span("♥", className="team-heart"),
                                                        html.Span("by", className="team-credit-text"),
                                                    ],
                                                    className="team-credit-row",
                                                ),
                                                html.Div(TEAM_NAME, className="team-name-xl"),
                                                html.Div(
                                                    [
                                                        html.Span("TEAM ID", className="team-id-label"),
                                                        html.Span(TEAM_ID, className="team-id-pill"),
                                                        html.Span(f"{len(TEAM_MEMBERS[:6])} MEMBERS", className="team-member-count"),
                                                    ],
                                                    className="team-id-row",
                                                ),
                                                html.Div(
                                                    [
                                                        html.A(
                                                            [
                                                                html.Span("⌘", className="team-link-icon"),
                                                                " GITHUB REPOSITORY",
                                                            ],
                                                            href=GITHUB_REPOSITORY_URL,
                                                            target="_blank",
                                                            rel="noopener noreferrer",
                                                            className="team-link-btn team-repo-btn",
                                                        ),
                                                        html.Button(
                                                            [
                                                                html.Span("✦", className="team-link-icon"),
                                                                " TEAM INFO",
                                                            ],
                                                            id="team-info-button",
                                                            n_clicks=0,
                                                            className="team-link-btn team-info-btn",
                                                        ),
                                                    ],
                                                    className="team-links-row",
                                                ),
                                                html.Div(
                                                    "Explainable public-works intelligence · Demo-ready · Human-in-the-loop",
                                                    className="team-tagline",
                                                ),
                                                html.Div(
                                                    "Click Team Info to meet the six members without leaving the dashboard.",
                                                    className="team-members-note",
                                                ),

                                                # ── LEGAL · LICENCE · SECURITY NOTICES ──
                                                # ── LICENCE · COMPLIANCE · SECURITY ──
                                                 html.Div(
                                                    [
                                                        # ── Divider with centred label ──
                                                        html.Div(
                                                            [
                                                                html.Span(className="trust-line"),
                                                                html.Span(
                                                                    "LICENCE · COMPLIANCE · SECURITY",
                                                                    className="trust-line-label",
                                                                ),
                                                                html.Span(className="trust-line"),
                                                            ],
                                                            className="trust-divider",
                                                        ),

                                                        # ── Licence pills with separators ──
                                                        html.Div(
                                                            [
                                                                html.Span("MIT Licence", className="trust-pill"),
                                                                html.Span("© 2026 Team Fresh Minds", className="trust-pill"),
                                                                html.Span("Jamia Millia Islamia", className="trust-pill"),
                                                                html.Span("All Rights Reserved", className="trust-pill trust-pill-quiet"),
                                                            ],
                                                            className="trust-pill-row",
                                                        ),

                                                        # ── Security banner ──
                                                        html.Div(
                                                            [
                                                                html.Div(
                                                                    [
                                                                        html.Span("🛡", className="trust-shield"),
                                                                        html.Span("PROTECTED BY", className="trust-shield-label"),
                                                                    ],
                                                                    className="trust-shield-wrap",
                                                                ),
                                                                html.Div(
                                                                    [
                                                                        html.Span("WAF", className="trust-pillar"),
                                                                        html.Span("DDoS Mitigation", className="trust-pillar"),
                                                                        html.Span("Rate Limiting", className="trust-pillar"),
                                                                        html.Span("Malware Scan", className="trust-pillar"),
                                                                        html.Span("Spam Filter", className="trust-pillar"),
                                                                    ],
                                                                    className="trust-pillar-row",
                                                                ),
                                                            ],
                                                            className="trust-security-wrap",
                                                        ),

                                                        # ── Advisory notes ──
                                                        html.Div(
                                                            [
                                                                html.Div(
                                                                    [
                                                                        html.Span("◆", className="trust-note-dot"),
                                                                        html.Span(
                                                                            "Advisory signals only — no automated adjudication, accusation, or enforcement.",
                                                                            className="trust-note-text",
                                                                        ),
                                                                    ],
                                                                    className="trust-note-row",
                                                                ),
                                                                html.Div(
                                                                    [
                                                                        html.Span("◆", className="trust-note-dot"),
                                                                        html.Span(
                                                                            "Malicious traffic is detected, throttled, and logged at the edge.",
                                                                            className="trust-note-text",
                                                                        ),
                                                                    ],
                                                                    className="trust-note-row",
                                                                ),
                                                            ],
                                                            className="trust-notes",
                                                        ),

                                                        # ── Jurisdiction line ──
                                                        html.Div(
                                                            "Protected under the Indian IT Act, 2000 · §43 · §66 · §66F",
                                                            className="trust-legal",
                                                        ),
                                                    ],
                                                    className="trust-block",
                                                ),
                                            ],
                                            className="team-footer-copy",
                                        ),
                                    ],
                                    className="team-footer-inner",
                                ),
                            ],
                            id="team-footer-section",
                            className="team-footer team-footer-pending",
                        ),
                        team_info_modal(),
                        seal_viewer(),
                    ],
                    className="main-content",
                ),
            ],
            className="app-shell",
        ),
    ]
)


# ============================================================
# CSS — DARK EDITORIAL / ANALYTICS SYSTEM
# ============================================================

app.layout = html.Div(
    [
        dcc.Location(id="url"),
        app.layout
    ]
) if False else app.layout


app.index_string = app.index_string.replace(
    "</head>",
    """
<style>
/* FINAL TEAM UI — sharp, stable portrait cards; no distortion */
.team-members-grid{display:grid!important;grid-template-columns:repeat(3,minmax(0,1fr))!important;gap:16px!important;align-items:stretch!important}
.team-member-card{display:grid!important;grid-template-columns:180px minmax(0,1fr)!important;gap:16px!important;min-height:210px!important;height:auto!important;padding:14px!important;align-items:stretch!important;overflow:hidden!important}
.team-member-visual{position:relative!important;min-width:0!important;height:100%!important;min-height:180px!important;aspect-ratio:4/5!important}
.team-member-photo-frame{position:relative!important;width:100%!important;height:100%!important;min-height:180px!important;aspect-ratio:4/5!important;border-radius:18px!important;overflow:hidden!important;transform:none!important;filter:none!important;backdrop-filter:none!important;-webkit-backdrop-filter:none!important;isolation:isolate!important}
.team-member-avatar,.team-member-avatar-photo{width:100%!important;height:100%!important;border-radius:16px!important;overflow:hidden!important;transform:none!important;filter:none!important}
.team-member-photo,.team-member-photo-frame img{display:block!important;width:100%!important;height:100%!important;max-width:none!important;max-height:none!important;object-fit:cover!important;object-position:50% 35%!important;transform:none!important;filter:none!important;image-rendering:auto!important;backface-visibility:visible!important;-webkit-backface-visibility:visible!important}
.team-member-photo-frame:after{display:none!important}
.team-member-photo-frame:before{inset:0!important;border-radius:16px!important}
.team-member-photo-corner{left:8px!important;bottom:8px!important}
.team-member-content{min-width:0!important;display:flex!important;flex-direction:column!important;justify-content:center!important;padding:2px 4px 2px 0!important}
.team-member-name-row{gap:8px!important;align-items:flex-start!important}
.team-member-name{font-size:19px!important;line-height:1.1!important}
.team-member-bio{max-width:none!important;line-height:1.48!important}
.team-member-tags{margin-top:auto!important;padding-top:10px!important}
.team-member-jmi-badge{flex:0 0 auto!important}
@media(max-width:980px){.team-members-grid{grid-template-columns:repeat(2,minmax(0,1fr))!important}.team-member-card{grid-template-columns:150px minmax(0,1fr)!important}.team-member-visual,.team-member-photo-frame{min-height:165px!important}}
@media(max-width:680px){.team-members-grid{grid-template-columns:1fr!important}.team-member-card{grid-template-columns:125px minmax(0,1fr)!important;min-height:165px!important}.team-member-visual,.team-member-photo-frame{min-height:145px!important}.team-member-name{font-size:17px!important}}
:root{
  --bg:#070B10;
  --bg2:#0B1118;
  --panel:#0E151D;
  --panel2:#121A23;
  --border:rgba(120,160,200,.12);
  --border2:rgba(120,160,200,.08);
  --text:#E8F0F6;
  --muted:#8FA0AD;
  --muted2:#6A7A88;
  --accent:#7EC8FF;
  --accent2:#C792EA;
  --gold:#F0C675;
  --success:#6BCB77;
  --warn:#F0A35A;
  --danger:#E85A5A;
  --shadow:0 18px 50px rgba(0,0,0,.35);
  --radius:16px;
  --sidebar-w:300px;
}
*{box-sizing:border-box}
html,body,#_dash-app-content{
  margin:0;
  min-height:100%;
  color:var(--text);
  font-family:"Segoe UI", system-ui, -apple-system, sans-serif;
  background:
    /* Moving aurora veil */
    radial-gradient(ellipse 45% 32% at 20% 8%,  rgba(110,255,196,.065), transparent 62%),
    radial-gradient(ellipse 40% 30% at 82% 14%, rgba(168,130,255,.055), transparent 62%),
    radial-gradient(ellipse 50% 34% at 50% 100%, rgba(120,210,255,.045), transparent 68%),
    /* Static base */
    linear-gradient(180deg,#060A0F 0%,#080E15 45%,#060A0F 100%);
  background-size: 260% 260%, 260% 260%, 260% 260%, 100% 100%;
  animation: pageAurora 60s ease-in-out infinite alternate;
}
@keyframes pageAurora{
  0%   { background-position: 0% 0%,   100% 0%,  50% 100%, 0% 0%; }
  50%  { background-position: 40% 30%, 60% 40%,  50% 60%,  0% 0%; }
  100% { background-position: 100% 60%, 0% 100%, 50% 0%,   0% 0%; }
}
@media(prefers-reduced-motion:reduce){
  html,body,#_dash-app-content{ animation:none !important; }
}
.app-shell{display:flex;min-height:100vh;position:relative}
.sidebar{
  position:sticky;top:0;width:var(--sidebar-w);height:100vh;overflow-y:auto;overflow-x:hidden;
  padding:20px 16px 28px;
  background:linear-gradient(180deg, rgba(12,18,26,.98) 0%, rgba(8,12,18,.99) 100%);
  border-right:1px solid var(--border);
  box-shadow:8px 0 40px rgba(0,0,0,.25);
  backdrop-filter:blur(20px);
  z-index:20;
  transition:width .28s cubic-bezier(.4,0,.2,1), padding .28s ease;
}
.sidebar::-webkit-scrollbar{width:6px}
.sidebar::-webkit-scrollbar-thumb{background:rgba(120,160,200,.2);border-radius:99px}
.sidebar-collapsed{width:64px!important;padding:16px 10px!important}
.sidebar-collapsed .brand-text-wrap,
.sidebar-collapsed .brand-micro,
.sidebar-collapsed .brand-accent-line,
.sidebar-collapsed .control-label,
.sidebar-collapsed .filter-mini-label,
.sidebar-collapsed .dark-dropdown,
.sidebar-collapsed .dark-input,
.sidebar-collapsed .sidebar-meta,
.sidebar-collapsed #risk-range,
.sidebar-collapsed .rc-slider,
.sidebar-collapsed #min-sanction,
.sidebar-collapsed #max-sanction,
.sidebar-collapsed #work-id-input,
.sidebar-collapsed #search-filter{display:none!important}
.sidebar-collapsed .brand{border:none;margin:0;padding:4px 0}
.sidebar-collapsed .sidebar-toggle{margin:0 auto}
.brand{padding:4px 6px 18px;margin-bottom:14px;border-bottom:1px solid var(--border2)}
.brand-top{display:flex;align-items:flex-start;justify-content:space-between;gap:8px}
.brand-mark{
  font:800 28px/1 Bahnschrift,"Segoe UI",sans-serif;letter-spacing:-.05em;
  background:linear-gradient(120deg,#F4F8FC 0%,#A8D4F5 55%,#C9B0FF 100%);
  -webkit-background-clip:text;background-clip:text;color:transparent;
}
.brand-sub{
  font:700 11px/1.2 Cascadia Mono,Consolas,monospace;letter-spacing:.22em;
  color:var(--accent);margin-top:6px;
  text-shadow:0 0 24px rgba(126,200,255,.25);
}
.brand-micro{font:500 8px/1.5 Cascadia Mono,Consolas,monospace;letter-spacing:.14em;color:var(--muted2);margin-top:10px}
.brand-accent-line{
  height:2px;margin-top:14px;border-radius:99px;
  background:linear-gradient(90deg, var(--accent), var(--accent2), var(--gold), transparent);
  opacity:.85;
}
.sidebar-toggle{
  width:34px;height:34px;border-radius:10px;cursor:pointer;
  border:1px solid var(--border);background:rgba(20,30,42,.9);
  color:var(--accent);font-size:14px;line-height:1;
  display:flex;align-items:center;justify-content:center;
  transition:all .15s ease;flex-shrink:0;
}
.sidebar-toggle:hover{background:rgba(40,70,100,.55);border-color:var(--accent);box-shadow:0 0 20px rgba(126,200,255,.15)}
.control-label{
  font:700 9px/1 Cascadia Mono,Consolas,monospace;letter-spacing:.16em;
  color:#7A9BB0;margin:16px 0 8px 2px;
}
.filter-mini-label{
  font-size:10px;letter-spacing:.6px;text-transform:uppercase;
  color:#6E8799;margin:10px 0 5px 2px;font-weight:600;
}
.dark-dropdown .Select-control,
.dark-dropdown .dropdown-control,
.Select-control{
  background:#0C141C!important;border:1px solid var(--border)!important;
  border-radius:12px!important;min-height:38px!important;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.03);
}
.dark-input{
  width:100%;box-sizing:border-box;margin:6px 0;
  background:linear-gradient(180deg,#0D151E,#0A1219);
  border:1px solid var(--border);border-radius:12px;
  color:var(--text);padding:10px 12px;font-size:13px;
  outline:none;transition:border-color .15s, box-shadow .15s;
}
.dark-input:focus{border-color:rgba(126,200,255,.45);box-shadow:0 0 0 3px rgba(126,200,255,.1)}
.main-content{
  flex:1;min-width:0;padding:22px 28px 40px;
  background:transparent;
}
.page-kicker{font:700 9px Cascadia Mono,Consolas,monospace;letter-spacing:.18em;color:#7FA8C4}
.page-title{
  font:800 clamp(28px,3.2vw,40px)/1.05 Bahnschrift,"Segoe UI",sans-serif;
  letter-spacing:-.04em;margin:6px 0 8px;
  background:linear-gradient(105deg,#F5F9FC 0%, #B8D9F0 50%, #D4C0FF 100%);
  -webkit-background-clip:text;background-clip:text;color:transparent;
}
.universe-banner{
  margin:16px 0 18px;padding:14px 18px;border-radius:16px;
  border:1px solid var(--border);
  background:linear-gradient(135deg, rgba(20,32,48,.9), rgba(12,18,28,.95));
  box-shadow:var(--shadow);
  position:relative;overflow:hidden;
}
.universe-banner:before{
  content:"";position:absolute;inset:0;
  background:linear-gradient(90deg, rgba(126,200,255,.06), transparent 40%, rgba(199,146,234,.05));
  pointer-events:none;
}
.kpi-grid{
  display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:12px;margin-bottom:12px;
}
.secondary-kpi-grid{
  display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:12px;margin-bottom:8px;
}
.kpi-card{
  min-height:118px;padding:16px 15px 14px;border-radius:16px;
  border:1px solid var(--border);
  background:
    linear-gradient(160deg, rgba(22,32,44,.95) 0%, rgba(12,18,26,.98) 100%);
  box-shadow:0 12px 32px rgba(0,0,0,.22), inset 0 1px 0 rgba(255,255,255,.04);
  position:relative;overflow:hidden;
  transition:transform .15s ease, border-color .15s ease;
}
.kpi-card:hover{transform:translateY(-2px);border-color:rgba(126,200,255,.22)}
.kpi-card:after{
  content:"";position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg, transparent, var(--accent), transparent);
  opacity:.35;
}
.kpi-card.warning:after{background:linear-gradient(90deg, transparent, var(--warn), transparent);opacity:.55}
.kpi-card.danger:after{background:linear-gradient(90deg, transparent, var(--danger), transparent);opacity:.55}
.kpi-card.accent:after{background:linear-gradient(90deg, transparent, var(--accent2), transparent);opacity:.5}
.kpi-label{font:700 9px Cascadia Mono,Consolas,monospace;letter-spacing:.12em;color:#7A93A6;margin-bottom:8px}
.kpi-value{font:750 26px/1.05 Bahnschrift,"Segoe UI",sans-serif;letter-spacing:-.03em;color:#F2F7FB}
.kpi-note{font-size:11px;color:var(--muted2);margin-top:8px;line-height:1.35}
.runtime-badges{display:flex;gap:8px;flex-wrap:wrap}
.runtime-badges .badge,
.badge{
  font:600 10px Cascadia Mono,Consolas,monospace;letter-spacing:.06em;
  padding:6px 11px;border-radius:999px;
  border:1px solid var(--border);background:rgba(14,22,32,.9);color:#A8C0D0;
}

/* ----- Header eyebrow + brand micro ----- */
.eyebrow{
  font:700 10px Cascadia Mono,Consolas,monospace!important;
  letter-spacing:.2em!important;
  color:transparent!important;
  background:linear-gradient(90deg,#8FD0FF,#C9B0FF,#F0C675)!important;
  -webkit-background-clip:text!important;background-clip:text!important;
  filter:drop-shadow(0 0 12px rgba(140,190,255,.25));
  margin-bottom:6px!important;
}
.brand-micro{
  font:600 9px/1.5 Cascadia Mono,Consolas,monospace!important;
  letter-spacing:.16em!important;
  color:#8AA8BC!important;
  margin-top:10px!important;
}
.brand-accent-line{
  height:3px!important;margin-top:12px!important;border-radius:99px!important;
  background:linear-gradient(90deg,#5B9FD4 0%,#8FD0FF 25%,#C792EA 55%,#F5C56B 85%,transparent 100%)!important;
  box-shadow:0 0 16px rgba(140,190,255,.35), 0 0 28px rgba(199,146,234,.2)!important;
  opacity:1!important;
}

/* ----- Runtime status badges (API / SOURCE / DASH) ----- */
.runtime-badges{
  display:flex!important;gap:10px!important;flex-wrap:wrap!important;
  align-items:center!important;justify-content:flex-end!important;
}
.runtime-badge{
  display:inline-flex!important;align-items:center!important;gap:8px!important;
  padding:8px 14px!important;border-radius:999px!important;
  border:1px solid rgba(140,180,220,.16)!important;
  background:linear-gradient(180deg, rgba(22,32,44,.95), rgba(12,18,26,.98))!important;
  box-shadow:0 6px 20px rgba(0,0,0,.22), inset 0 1px 0 rgba(255,255,255,.04)!important;
  font:600 10px Cascadia Mono,Consolas,monospace!important;
  letter-spacing:.06em!important;
  color:#C5D6E4!important;
  transition:transform .15s ease, box-shadow .15s ease, border-color .15s ease!important;
  cursor:default!important;
}
.runtime-badge:hover{transform:translateY(-1px)!important}
.badge-dot{
  width:8px;height:8px;border-radius:50%;flex-shrink:0;
  background:#5a6a78;box-shadow:0 0 0 3px rgba(90,106,120,.2);
}
.badge-dot.online{
  background:#6BCB77;
  box-shadow:0 0 0 3px rgba(107,203,119,.2), 0 0 12px rgba(107,203,119,.55), 0 0 22px rgba(107,203,119,.25);
  animation:badge-pulse 2s ease-in-out infinite;
}
.badge-dot.offline{
  background:#E85A5A;
  box-shadow:0 0 0 3px rgba(232,90,90,.2), 0 0 12px rgba(232,90,90,.4);
}
@keyframes badge-pulse{
  0%,100%{box-shadow:0 0 0 3px rgba(107,203,119,.2), 0 0 10px rgba(107,203,119,.45)}
  50%{box-shadow:0 0 0 5px rgba(107,203,119,.12), 0 0 18px rgba(107,203,119,.65)}
}
.badge-logo{
  font-size:12px;line-height:1;display:inline-flex;align-items:center;justify-content:center;
  width:18px;height:18px;border-radius:6px;
}
.badge-logo.source{
  color:#8FD0FF;
  background:rgba(100,160,220,.15);
  box-shadow:0 0 12px rgba(126,200,255,.25);
}
.badge-logo.dash{
  color:#C792EA;
  background:rgba(180,120,220,.15);
  box-shadow:0 0 12px rgba(199,146,234,.3);
  font-size:11px;
}
.badge-brand{color:#7A93A6;font-weight:700;letter-spacing:.1em}
.badge-status{color:#E8F2FA;font-weight:700}
.runtime-badge.badge-api.is-online{
  border-color:rgba(107,203,119,.35)!important;
  box-shadow:0 0 20px rgba(107,203,119,.12), 0 6px 20px rgba(0,0,0,.2), inset 0 1px 0 rgba(255,255,255,.04)!important;
}
.runtime-badge.badge-api.is-online .badge-status{color:#9EE6A8}
.runtime-badge.badge-api.is-offline{
  border-color:rgba(232,90,90,.4)!important;
}
.runtime-badge.badge-source{
  border-color:rgba(126,200,255,.22)!important;
}
.runtime-badge.badge-source:hover{
  box-shadow:0 0 22px rgba(126,200,255,.18), 0 8px 22px rgba(0,0,0,.25)!important;
}
.runtime-badge.badge-dash{
  border-color:rgba(199,146,234,.25)!important;
}
.runtime-badge.badge-dash:hover{
  box-shadow:0 0 22px rgba(199,146,234,.2), 0 8px 22px rgba(0,0,0,.25)!important;
}

.badge.online{border-color:rgba(107,203,119,.35);color:#9EE6A8;box-shadow:0 0 16px rgba(107,203,119,.12)}
.panel,.overview-chart-panel,.deep-panel{
  border:1px solid var(--border)!important;
  background:linear-gradient(165deg, rgba(16,24,34,.96), rgba(10,15,22,.98))!important;
  box-shadow:0 14px 40px rgba(0,0,0,.2)!important;
  border-radius:16px!important;
}
@media(max-width:1400px){
  .kpi-grid,.secondary-kpi-grid{grid-template-columns:repeat(3,minmax(0,1fr))}
}
@media(max-width:900px){
  .kpi-grid,.secondary-kpi-grid{grid-template-columns:repeat(2,minmax(0,1fr))}
  .sidebar{width:260px}
}

.sidebar{position:sticky;top:0;width:286px;height:100vh;overflow-y:auto;padding:24px 18px;background:rgba(12,17,23,.96);border-right:1px solid var(--border);backdrop-filter:blur(18px);z-index:10}
.brand{padding:6px 8px 22px;border-bottom:1px solid var(--border2);margin-bottom:18px}
.brand-mark{font:700 30px/1 Bahnschrift,Segoe UI,sans-serif;letter-spacing:-.06em}
.brand-sub{font:600 12px/1.2 Cascadia Mono,Consolas,monospace;letter-spacing:.18em;color:var(--accent);margin-top:7px}
.brand-micro{font:500 8px/1.5 Cascadia Mono,Consolas,monospace;letter-spacing:.12em;color:var(--muted2);margin-top:10px}
.filter-mini-label{
  font-size:10px;
  letter-spacing:0.6px;
  text-transform:uppercase;
  color:#7F8F9C;
  margin:8px 0 4px 2px;
}
.control-label{font:700 9px/1 Cascadia Mono,Consolas,monospace;letter-spacing:.15em;color:#6F7F8C;margin:19px 3px 8px}
.dark-dropdown{margin-bottom:8px;position:relative}
.dark-dropdown .Select-control{background:#111821!important;border:1px solid #2A3743!important;border-radius:9px!important;min-height:42px!important;box-shadow:none!important}
.dark-dropdown .Select-control:hover{border-color:#4A6376!important}
.dark-dropdown .Select-value-label{color:#EAF2F7!important;font-weight:500!important}
.dark-dropdown .Select-placeholder{color:#83929E!important}
.dark-dropdown .Select-input>input{color:#EAF2F7!important;background:transparent!important}
.dark-dropdown .Select-arrow{border-top-color:#91A7B6!important}
.dark-dropdown .Select-clear{color:#91A7B6!important}
.dark-dropdown .Select-menu-outer{background:#10171F!important;border:1px solid #2A3743!important;border-radius:9px!important;box-shadow:0 18px 40px rgba(0,0,0,.45)!important;overflow:hidden!important;z-index:9999!important}
.dark-dropdown .Select-menu{background:#10171F!important}
.dark-dropdown .Select-option{background:#10171F!important;color:#DCE6ED!important;padding:10px 12px!important}
.dark-dropdown .Select-option.is-focused{background:#1B2935!important;color:#FFFFFF!important}
.dark-dropdown .Select-option.is-selected{background:#183246!important;color:#FFFFFF!important}
.dark-dropdown .VirtualizedSelectFocusedOption{background:#1B2935!important;color:#FFFFFF!important}
.dark-dropdown .VirtualizedSelectOption{background:#10171F!important;color:#DCE6ED!important}
/* React-Select v5 / current Dash selectors */
.dark-dropdown .Select__control{background:#111821!important;border:1px solid #2A3743!important;border-radius:9px!important;min-height:42px!important;box-shadow:none!important}
.dark-dropdown .Select__control:hover{border-color:#4A6376!important}
.dark-dropdown .Select__single-value{color:#EAF2F7!important}
.dark-dropdown .Select__placeholder{color:#83929E!important}
.dark-dropdown .Select__input-container,.dark-dropdown .Select__input-container input{color:#EAF2F7!important}
.dark-dropdown .Select__dropdown-indicator,.dark-dropdown .Select__clear-indicator{color:#91A7B6!important}
.dark-dropdown .Select__menu{background:#10171F!important;border:1px solid #2A3743!important;border-radius:9px!important;box-shadow:0 18px 40px rgba(0,0,0,.45)!important;overflow:hidden!important;z-index:9999!important}
.dark-dropdown .Select__option{background:#10171F!important;color:#DCE6ED!important;padding:10px 12px!important}
.dark-dropdown .Select__option--is-focused{background:#1B2935!important;color:#FFFFFF!important}
.dark-dropdown .Select__option--is-selected{background:#183246!important;color:#FFFFFF!important}
.dark-input{width:100%;background:#10171F;color:var(--text);border:1px solid var(--border);border-radius:9px;padding:10px 11px;margin:5px 0;outline:none}
.dark-input:focus{border-color:#4C7795;box-shadow:0 0 0 2px rgba(143,208,255,.08)}
.rc-slider{margin:8px 5px 26px}
.rc-slider-track{background:var(--accent)}
.rc-slider-handle{border-color:var(--accent);background:#0E151C}
.rc-slider-mark-text{color:#6F7F8C!important;font-size:9px}
.action-button{width:100%;border:1px solid #31546C;background:#132331;color:#DFF2FF;border-radius:10px;padding:11px 13px;font:700 10px Cascadia Mono,Consolas,monospace;letter-spacing:.08em;cursor:pointer;margin-top:16px}
.action-button:hover{background:#183044;border-color:#4E7897}
.action-button.compact{width:auto;margin:12px 0}
.connection-state{font:500 10px Cascadia Mono,Consolas,monospace;color:var(--green);padding:13px 3px 4px}
.sidebar-meta{border-top:1px solid var(--border2);margin-top:14px;padding-top:14px}
.meta-label{font:700 8px Cascadia Mono,Consolas,monospace;color:#667582;letter-spacing:.12em;margin-top:8px}
.meta-value{font:500 9px Cascadia Mono,Consolas,monospace;color:#9DABB6;word-break:break-all;margin-top:3px}
.main-content{width:calc(100% - 286px);max-width:1700px;margin:0 auto;padding:28px 34px 38px}
.topbar{display:flex;justify-content:space-between;gap:30px;align-items:flex-start}
.eyebrow,.section-kicker{font:700 9px Cascadia Mono,Consolas,monospace;letter-spacing:.16em;color:#71818D}
.page-title{font:700 43px/1 Bahnschrift,Segoe UI,sans-serif;letter-spacing:-.055em;margin:8px 0 8px}
.page-subtitle{color:var(--muted);margin:0;max-width:720px;font-size:14px;line-height:1.6}
.runtime-badges{display:flex;gap:7px;flex-wrap:wrap;justify-content:flex-end}
.runtime-badge{border:1px solid var(--border);background:#10171F;border-radius:999px;padding:7px 10px;font:500 9px Cascadia Mono,Consolas,monospace;color:#AAB8C3}
.scope-banner{margin-top:22px}
.banner{border:1px solid #263F51;background:linear-gradient(135deg,#101C26,#0D151C);border-radius:14px;padding:13px 16px;box-shadow:var(--shadow)}
.banner.error{border-color:#63363A;background:#1C1215;color:#FFB7B7}
.banner-kicker{font:700 9px Cascadia Mono,Consolas,monospace;color:var(--accent);letter-spacing:.15em}
.banner-main{font:600 17px Bahnschrift,Segoe UI,sans-serif;margin-top:3px}
.banner-sub{font-size:11px;color:var(--muted);margin-top:3px}
.kpi-grid{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:10px;margin-top:14px}
.kpi-card{min-height:116px;padding:15px 15px 13px;border:1px solid var(--border);border-radius:14px;background:linear-gradient(145deg,#111821,#0E141B);box-shadow:0 10px 30px rgba(0,0,0,.13)}
.kpi-card.warning{border-color:#4D422C}
.kpi-card.danger{border-color:#553136}
.kpi-label{font:700 8px Cascadia Mono,Consolas,monospace;letter-spacing:.12em;color:#71818D}
.kpi-value{font:700 24px/1.1 Bahnschrift,Segoe UI,sans-serif;letter-spacing:-.04em;margin-top:12px}
.kpi-hint{font-size:9px;color:var(--muted2);margin-top:7px}

/* ============================================================
   OVERVIEW — CITIZEN-FIRST ANALYTICS SURFACE
   ============================================================ */

.overview-two-col{
  display:grid;
  grid-template-columns:repeat(2,minmax(0,1fr));
  gap:18px;
  margin:18px 0 8px;
  align-items:stretch;
}
.overview-chart-cell{display:flex;flex-direction:column;min-height:0}
.overview-section-kicker{
  font:700 10px Cascadia Mono,Consolas,monospace;
  letter-spacing:.14em;
  color:#7FA8C4;
  margin:0 0 10px 2px;
}
/* ── LIVING AMBIENT PANEL ───────────────────────────────── */
/* ============================================================
   NORTH-POLE AMBIENT PANEL
   Aurora curtains · star field · slow drift · ultra-low opacity
   ============================================================ */
.overview-chart-panel{
  position:relative;
  overflow:hidden;
  isolation:isolate;
  flex:1;
  min-height:440px;
  padding:10px 12px 6px;
  border:1px solid rgba(143,208,255,.10);
  border-radius:16px;

  /* Star field — tiny pinpricks, static, sits under everything */
  background-image:
    radial-gradient(1.4px 1.4px at 11% 17%, rgba(255,255,255,.55), transparent 55%),
    radial-gradient(1px   1px   at 27% 71%, rgba(200,228,255,.42), transparent 55%),
    radial-gradient(1.3px 1.3px at 44% 33%, rgba(224,210,255,.50), transparent 55%),
    radial-gradient(1px   1px   at 63% 82%, rgba(255,255,255,.38), transparent 55%),
    radial-gradient(1.2px 1.2px at 79% 22%, rgba(210,235,255,.46), transparent 55%),
    radial-gradient(1px   1px   at 91% 63%, rgba(255,255,255,.34), transparent 55%),
    radial-gradient(1px   1px   at 18% 88%, rgba(180,220,255,.32), transparent 55%),
    radial-gradient(1px   1px   at 56% 09%, rgba(255,255,255,.30), transparent 55%),
    linear-gradient(155deg, rgba(7,11,18,.985) 0%, rgba(9,14,24,.99) 100%);
  background-size: 100% 100%, 100% 100%, 100% 100%, 100% 100%,
                   100% 100%, 100% 100%, 100% 100%, 100% 100%, 100% 100%;

  box-shadow:
    0 14px 40px rgba(0,0,0,.30),
    inset 0 1px 0 rgba(255,255,255,.030),
    inset 0 0 0 1px rgba(143,208,255,.025);
  transition:border-color .4s ease, box-shadow .4s ease;
}

.overview-chart-panel:hover{
  border-color:rgba(143,208,255,.20);
  box-shadow:
    0 18px 46px rgba(0,0,0,.34),
    0 0 40px rgba(143,208,255,.05),
    inset 0 1px 0 rgba(255,255,255,.045),
    inset 0 0 0 1px rgba(143,208,255,.045);
}

/* ── Aurora curtain #1 — greens & soft purples ── */
.overview-chart-panel::before{
  content:"";
  position:absolute;
  inset:-38%;
  pointer-events:none;
  z-index:0;
  background:
    radial-gradient(ellipse 40% 30% at 20% 74%, rgba(110,255,196,.10), transparent 66%),
    radial-gradient(ellipse 36% 26% at 74% 26%, rgba(168,130,255,.085), transparent 66%),
    radial-gradient(ellipse 48% 30% at 46% 54%, rgba(120,210,255,.065), transparent 70%),
    radial-gradient(ellipse 32% 24% at 88% 62%, rgba(190,255,220,.055), transparent 66%);
  filter: blur(34px);
  mix-blend-mode: screen;
  opacity:.85;
  animation: auroraDrift 38s ease-in-out infinite alternate;
  will-change: transform;
}

/* ── Aurora curtain #2 — cyan offset layer, adds depth ── */
.overview-chart-panel::after{
  content:"";
  position:absolute;
  inset:-38%;
  pointer-events:none;
  z-index:0;
  background:
    radial-gradient(ellipse 34% 28% at 82% 22%, rgba(188,132,255,.075), transparent 66%),
    radial-gradient(ellipse 42% 28% at 24% 42%, rgba(80,235,210,.060), transparent 68%),
    radial-gradient(ellipse 30% 22% at 60% 84%, rgba(140,190,255,.055), transparent 66%);
  filter: blur(42px);
  mix-blend-mode: screen;
  opacity:.7;
  animation: auroraDrift 52s ease-in-out infinite alternate-reverse;
  will-change: transform;
}

.overview-chart-panel > *{
  position:relative;
  z-index:1;
}

.overview-chart-panel .js-plotly-plot,
.overview-chart-panel .dash-graph{
  height:100% !important;
  min-height:420px;
}

/* Slow, organic drift — never distracting, always alive */
@keyframes auroraDrift{
  0%   { transform: translate3d(-6%, -3%, 0) scale(1.06); }
  33%  { transform: translate3d( 3%,  5%, 0) scale(1.10); }
  66%  { transform: translate3d( 7%, -2%, 0) scale(1.07); }
  100% { transform: translate3d(-4%,  7%, 0) scale(1.12); }
}

@media(prefers-reduced-motion:reduce){
  .overview-chart-panel::before,
  .overview-chart-panel::after{ animation:none !important; }
}
.overview-guide-panel{
  padding:20px 22px;
  border:1px solid var(--border);
  border-radius:16px;
  background:linear-gradient(160deg,#101820,#0C131A);
  min-height:400px;
}
.overview-clean .band-row{
  display:flex;gap:8px;align-items:baseline;
  padding:7px 0;border-bottom:1px solid rgba(255,255,255,.04);
  font-size:13px;color:#C5D2DB;
}
.overview-clean .band-row b{color:#E8F1F7;font-weight:650;min-width:92px}
.overview-clean .band-row span{color:#8FA0AD}
@media(max-width:1100px){
  .overview-two-col{grid-template-columns:1fr}
}
.overview-hero{
  display:grid;
  grid-template-columns:minmax(0,1.55fr) minmax(290px,.75fr);
  gap:14px;
  margin-bottom:14px;
}
.overview-intro{
  position:relative;
  overflow:hidden;
  min-height:230px;
  padding:24px 25px;
  border:1px solid #29465A;
  border-radius:18px;
  background:
    radial-gradient(circle at 92% 12%,rgba(125,191,232,.12),transparent 34%),
    linear-gradient(135deg,#111F2A 0%,#0E171F 58%,#0B1117 100%);
  box-shadow:0 18px 48px rgba(0,0,0,.20);
}
.overview-intro:after{
  content:"";
  position:absolute;
  right:-80px;
  bottom:-105px;
  width:260px;
  height:260px;
  border:1px solid rgba(143,208,255,.12);
  border-radius:50%;
  box-shadow:0 0 0 34px rgba(143,208,255,.025),0 0 0 68px rgba(143,208,255,.018);
}
.overview-kicker{font:700 9px Cascadia Mono,Consolas,monospace;letter-spacing:.18em;color:#8FD0FF}
.overview-headline{font:700 clamp(30px,3.5vw,48px)/.98 Bahnschrift,Segoe UI,sans-serif;letter-spacing:-.055em;margin:12px 0 10px;max-width:780px}
.overview-copy{font-size:13px;line-height:1.65;color:#AEBCC7;max-width:760px;margin:0}
.overview-meta{display:flex;gap:8px;flex-wrap:wrap;margin-top:20px}
.overview-chip{border:1px solid #2B4455;background:#0C151D;border-radius:999px;padding:7px 10px;color:#9EB1BF;font:600 9px Cascadia Mono,Consolas,monospace}
.overview-hero-panel{
  position:relative;
  overflow:hidden;
  border:1px solid rgba(143,208,255,.10);
  border-radius:18px;
  padding:12px;
  box-shadow:
    0 14px 40px rgba(0,0,0,.28),
    inset 0 1px 0 rgba(255,255,255,.035);
  display:flex;
  flex-direction:column;
  align-items:stretch;
  justify-content:flex-start;
  background:
    radial-gradient(circle at 50% 100%, rgba(143,208,255,.06), transparent 55%),
    radial-gradient(circle at 0% 0%,     rgba(199,146,234,.05), transparent 50%),
    linear-gradient(135deg,
      rgba(15,22,32,.97)  0%,
      rgba(22,30,48,.97) 50%,
      rgba(15,22,32,.97) 100%);
  background-size: 320% 320%, 320% 320%, 420% 420%;
  animation: panelAmbient 26s ease-in-out infinite;
}
.overview-hero-panel:before{
  content:"";
  position:absolute;
  inset:-1px;
  border-radius:inherit;
  pointer-events:none;
  z-index:0;
  background:
    radial-gradient(circle at 50% 50%, rgba(143,208,255,.10), transparent 48%),
    radial-gradient(circle at 80% 20%, rgba(199,146,234,.05), transparent 38%);
  animation: panelPulse 6.2s ease-in-out infinite;
  mix-blend-mode:screen;
}
.overview-hero-panel > *{ position:relative; z-index:1; }
.overview-hero > .overview-hero-copy{padding-top:0}
.overview-hero > .overview-hero-copy .section-kicker,
.overview-hero > .overview-hero-panel .overview-panel-kicker{margin-top:2px;min-height:12px;display:flex;align-items:center}
.overview-panel-kicker{font:700 9px Cascadia Mono,Consolas,monospace;letter-spacing:.14em;color:#8FD0FF;line-height:1.2;margin:2px 4px 0;text-align:left;white-space:nowrap}
.overview-hero-panel .js-plotly-plot{width:100%}
.overview-section-label{display:flex;justify-content:space-between;align-items:end;gap:14px;margin:24px 0 11px}
.overview-section-label h3{font:650 19px Bahnschrift,Segoe UI,sans-serif;letter-spacing:-.025em;margin:3px 0 0}
.overview-section-label p{font-size:10px;color:#748692;margin:0;max-width:700px;line-height:1.5;text-align:right}
.overview-signal-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin-bottom:14px}
.signal-card{min-height:90px;border:1px solid var(--border);border-radius:13px;background:#0E151C;padding:13px 14px}
.signal-card .signal-label{font:700 8px Cascadia Mono,Consolas,monospace;letter-spacing:.12em;color:#70818E}
.signal-card .signal-value{font:700 23px Bahnschrift,Segoe UI,sans-serif;letter-spacing:-.04em;margin-top:9px}
.signal-card .signal-note{font-size:9px;color:#73838F;margin-top:5px}
.signal-card.warning{border-color:#51452D}.signal-card.danger{border-color:#58343A}.signal-card.accent{border-color:#2B4C62}
.secondary-kpi-grid{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:9px;margin-top:10px}
.secondary-kpi-grid:empty{display:none}
.secondary-kpi-grid .kpi-card{min-height:88px;padding:12px 13px;background:#0D141B;border-radius:12px}
.secondary-kpi-grid .kpi-value{font-size:20px;margin-top:8px}
.secondary-kpi-grid .kpi-hint{margin-top:5px}
.overview-queue-head{display:flex;justify-content:space-between;align-items:end;gap:15px;margin-bottom:10px}
.overview-queue-title{font:650 20px Bahnschrift,Segoe UI,sans-serif;letter-spacing:-.025em;margin:3px 0 0}
.overview-queue-note{font-size:10px;color:#748692;line-height:1.5;max-width:620px;text-align:right}
@media(max-width:1050px){.overview-hero{grid-template-columns:1fr}.secondary-kpi-grid{grid-template-columns:repeat(3,1fr)}.overview-signal-grid{grid-template-columns:repeat(2,1fr)}}
@media(max-width:850px){.secondary-kpi-grid{grid-template-columns:repeat(2,1fr)}.overview-signal-grid{grid-template-columns:1fr}.overview-section-label{display:block}.overview-section-label p,.overview-queue-note{text-align:left;margin-top:5px}}
.view-container{margin-top:28px}
.section-header{margin-bottom:15px}
.section-title{font:650 27px/1.1 Bahnschrift,Segoe UI,sans-serif;letter-spacing:-.035em;margin:5px 0 7px}
.section-note{font-size:11px;color:var(--muted);line-height:1.55}
.overview-grid,.chart-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin-bottom:14px}
.panel{border:1px solid var(--border);border-radius:15px;background:rgba(15,21,28,.86);padding:7px 9px;box-shadow:var(--shadow);margin-bottom:14px}
.queue-meta{font:500 10px Cascadia Mono,Consolas,monospace;color:#7E8D99;margin:4px 0 10px}
.detail-shell{margin-top:18px}
.detail-card{border:1px solid #31536A;border-radius:17px;background:linear-gradient(145deg,#101B24,#0E151C);padding:18px;box-shadow:var(--shadow)}
.detail-head{display:flex;justify-content:space-between;gap:20px;align-items:center}
.detail-title{font:650 25px/1.2 Bahnschrift,Segoe UI,sans-serif;letter-spacing:-.03em}
.detail-meta{font-size:11px;color:var(--muted);margin-top:8px}
.mono{font:500 9px Cascadia Mono,Consolas,monospace;color:#7E93A3;margin-top:8px}
.score-orb{width:110px;height:110px;border:1px solid #39566B;border-radius:50%;display:flex;flex-direction:column;justify-content:center;align-items:center;background:radial-gradient(circle,#172631,#0D151C 65%);flex:none}
.score-number{font:700 30px Bahnschrift,Segoe UI,sans-serif}
.score-band{font:700 8px Cascadia Mono,Consolas,monospace;color:var(--accent);letter-spacing:.1em;margin-top:4px}
.detail-kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:9px;margin:16px 0}
.explanation{border-left:3px solid var(--accent);background:#111A22;border-radius:0 10px 10px 0;padding:13px;line-height:1.6;font-size:12px;margin:10px 0 15px}
.evidence{white-space:pre-wrap;max-height:350px;overflow:auto;background:#0A1016;border:1px solid var(--border2);border-radius:10px;padding:12px;color:#9FB0BC;font:10px/1.5 Cascadia Mono,Consolas,monospace}
.method-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.method-flow{font:11px/1.8 Cascadia Mono,Consolas,monospace;color:#AFC1CD;background:#0A1016;border:1px solid var(--border2);padding:15px;border-radius:10px;overflow:auto}
.band-row{display:flex;justify-content:space-between;border-bottom:1px solid var(--border2);padding:10px 4px;font-size:11px}
.band-row span{color:var(--muted)}
.band-low b{color:var(--green)}.band-medium b{color:var(--amber)}.band-high b{color:#F0A36B}.band-critical b{color:var(--red)}
.guardrail{border:1px solid #5B4930;background:#1C1710;color:#D9C39B;border-radius:12px;padding:14px;margin-top:14px;font-size:11px;line-height:1.6}
.empty-panel{border:1px dashed #2B3945;border-radius:13px;padding:30px;color:#74838F;text-align:center;font-size:12px;background:#0D131A}

.method-flow-visual{display:flex;flex-direction:column;gap:6px;margin-top:14px}
.flow-step{display:flex;gap:14px;align-items:flex-start;padding:12px 14px;border-radius:14px;background:linear-gradient(135deg,rgba(20,30,40,.9),rgba(12,18,26,.95));border:1px solid rgba(255,255,255,.05)}
.flow-num{width:42px;height:42px;border-radius:12px;border:1.5px solid;display:flex;align-items:center;justify-content:center;font:700 13px Cascadia Mono,Consolas,monospace;flex-shrink:0;background:rgba(0,0,0,.25)}
.flow-title{font:650 15px Bahnschrift,Segoe UI,sans-serif;color:#E8F1F7;margin-bottom:4px}
.flow-note{font-size:12.5px;line-height:1.55;color:#9AABBA}
.flow-arrow{text-align:center;font-size:16px;opacity:.7;line-height:1;padding:2px 0}
.band-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin-top:12px}
.band-card{padding:12px 14px;border-radius:12px;background:#0E151C;border:1px solid #1E2A34}
.band-name{font:700 13px Cascadia Mono,Consolas,monospace;letter-spacing:.06em}
.band-range{font-size:12px;color:#8FA0AD;margin:4px 0 8px}
.band-bar{height:4px;border-radius:99px;opacity:.85}
.guard-list{margin:8px 0 0 18px;color:#B0C0CC;font-size:13px;line-height:1.7}
.method-panel{min-height:420px}


/* ----- Collapsible work queue drawer ----- */
.queue-drawer{
  margin-top:22px;
  border-radius:18px;
  border:1px solid rgba(140,180,220,.16);
  background:linear-gradient(165deg, rgba(18,28,40,.96), rgba(10,16,24,.98));
  box-shadow:0 16px 40px rgba(0,0,0,.25);
  overflow:hidden;
}
.queue-drawer-summary{
  list-style:none;
  cursor:pointer;
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:16px;
  padding:18px 20px;
  user-select:none;
  transition:background .15s ease;
}
.queue-drawer-summary::-webkit-details-marker{display:none}
.queue-drawer-summary:hover{
  background:linear-gradient(90deg, rgba(80,140,200,.1), rgba(160,100,200,.06));
}
.queue-drawer-left{display:flex;align-items:center;gap:14px;min-width:0}
.queue-drawer-emoji{
  width:48px;height:48px;border-radius:14px;
  display:flex;align-items:center;justify-content:center;
  font-size:24px;
  background:linear-gradient(135deg, rgba(100,160,220,.2), rgba(180,120,220,.15));
  border:1px solid rgba(140,180,220,.2);
  box-shadow:0 0 20px rgba(120,180,255,.12);
  flex-shrink:0;
}
.queue-drawer-title{
  font:700 17px/1.15 Bahnschrift,"Segoe UI",sans-serif;
  letter-spacing:-.02em;color:#EDF4FA;
}
.queue-drawer-sub{font-size:12px;color:#8FA0AD;margin-top:4px;line-height:1.4}
.queue-drawer-right{display:flex;align-items:center;gap:10px;flex-shrink:0}
.queue-drawer-count{
  font:800 20px Bahnschrift,"Segoe UI",sans-serif;
  background:linear-gradient(90deg,#8FD0FF,#C792EA);
  -webkit-background-clip:text;background-clip:text;color:transparent;
}
.queue-drawer-count-label{font:600 11px Cascadia Mono,Consolas,monospace;color:#7A90A0;letter-spacing:.06em}
.queue-drawer-chevron{
  width:32px;height:32px;border-radius:10px;
  display:flex;align-items:center;justify-content:center;
  border:1px solid rgba(140,180,220,.2);
  background:rgba(20,32,44,.8);
  color:#8FD0FF;font-size:16px;
  transition:transform .2s ease;
}
.queue-drawer[open] .queue-drawer-chevron{transform:rotate(90deg)}
.queue-drawer-body{border-top:1px solid rgba(140,180,220,.1);padding:12px 14px 16px}
.queue-drawer-hint{font-size:11px;color:#7A90A0;margin:0 4px 10px;line-height:1.45}
.queue-drawer-scroll{
  max-height:420px;
  overflow:auto;
  border-radius:12px;
  border:1px solid rgba(140,180,220,.08);
}
.queue-drawer-scroll::-webkit-scrollbar{width:8px;height:8px}
.queue-drawer-scroll::-webkit-scrollbar-thumb{background:rgba(120,160,200,.25);border-radius:99px}

.team-footer{position:relative;overflow:hidden;margin-top:40px;padding:28px 18px 18px;border-top:1px solid rgba(143,200,255,.12);text-align:center}
.team-footer-glow{position:absolute;inset:0;background:radial-gradient(ellipse 70% 80% at 50% -10%, rgba(100,160,230,.14), transparent 55%);pointer-events:none}
.team-eyebrow{position:relative;font:700 10px Cascadia Mono,Consolas,monospace;letter-spacing:.2em;color:#6A8FA8;margin:4px 0 10px}

/* ----- Action buttons & collapsed rail ----- */
.action-button{
  width:100%;display:flex;align-items:center;justify-content:center;gap:8px;
  margin:8px 0;padding:11px 12px;border-radius:12px;cursor:pointer;
  border:1px solid rgba(120,160,200,.16);
  background:linear-gradient(180deg, rgba(28,40,54,.95), rgba(16,24,34,.98));
  color:#D5E6F2;font:650 12px "Segoe UI",sans-serif;
  transition:all .15s ease;box-shadow:0 6px 18px rgba(0,0,0,.18);
}
.action-button:hover{border-color:rgba(126,200,255,.4);transform:translateY(-1px);box-shadow:0 8px 22px rgba(0,0,0,.28)}
.action-primary{
  background:linear-gradient(135deg, rgba(50,110,170,.55), rgba(30,70,120,.45));
  border-color:rgba(126,200,255,.35);color:#EAF6FF;
}
.btn-icon{font-size:15px;line-height:1;min-width:18px;text-align:center}
.btn-label{white-space:nowrap}
.sidebar-collapsed .action-button{
  width:44px;height:44px;padding:0;margin:8px auto;border-radius:14px;
}
.sidebar-collapsed .btn-label{display:none}
.sidebar-collapsed .btn-icon{font-size:17px;min-width:auto}
.sidebar-collapsed .connection-state,
.sidebar-collapsed .meta-label,
.sidebar-collapsed .meta-value{display:none!important}
.sidebar-collapsed .sidebar-meta{display:none!important}
.connection-state{font:600 11px Cascadia Mono,Consolas,monospace;color:#8FAF9A;margin-top:10px;padding:8px 10px;border-radius:10px;background:rgba(40,80,50,.15);border:1px solid rgba(120,200,140,.12)}

/* ----- Header / universe / title polish ----- */
.page-header,.main-header{position:relative}
.page-title,h1.page-title,.main-content h1{
  font:800 clamp(30px,3.4vw,42px)/1.02 Bahnschrift,"Segoe UI",sans-serif!important;
  letter-spacing:-.045em!important;
  background:linear-gradient(110deg,#FFFFFF 0%, #B8DCF5 42%, #D2B8FF 78%, #F0D090 100%)!important;
  -webkit-background-clip:text!important;background-clip:text!important;
  color:transparent!important;
  filter:drop-shadow(0 2px 24px rgba(120,180,255,.12));
}
.universe-banner,#scope-banner,.scope-banner{
  position:relative;overflow:hidden;
  border-radius:18px!important;
  border:1px solid rgba(126,180,230,.14)!important;
  background:
    linear-gradient(120deg, rgba(30,50,72,.55), rgba(18,28,40,.85) 40%, rgba(40,30,60,.35)),
    linear-gradient(180deg, #101820, #0C131A)!important;
  box-shadow:0 16px 48px rgba(0,0,0,.28), inset 0 1px 0 rgba(255,255,255,.05)!important;
}
.kpi-card .kpi-value{background:linear-gradient(180deg,#FFFFFF,#C5D8E8);-webkit-background-clip:text;background-clip:text;color:transparent}

/* ----- Interactive team footer ----- */
.team-footer{
  position:relative;overflow:hidden;
  margin-top:48px;padding:32px 20px 22px;
  border-top:1px solid rgba(140,190,240,.1);
  text-align:center;
  background:
    radial-gradient(ellipse 80% 100% at 50% -20%, rgba(100,160,230,.12), transparent 55%),
    radial-gradient(ellipse 50% 60% at 80% 120%, rgba(180,120,220,.08), transparent 50%);
}
.team-footer-glow{display:none}
.team-eyebrow{
  font:700 10px Cascadia Mono,Consolas,monospace;letter-spacing:.22em;
  color:#6E92A8;margin-bottom:12px;
}
.team-credit-row{gap:10px;margin-bottom:6px}
.team-heart{
  display:inline-block;color:#FF6B8A;font-size:18px;
  animation:pulse-heart 1.6s ease-in-out infinite;
  filter:drop-shadow(0 0 8px rgba(255,107,138,.45));
}
@keyframes pulse-heart{0%,100%{transform:scale(1)}50%{transform:scale(1.2)}}
.team-name-xl{
  font:800 clamp(28px,4vw,40px)/1.05 Bahnschrift,"Segoe UI",sans-serif;
  letter-spacing:-.03em;margin:2px 0 12px;
  background:linear-gradient(105deg,#8FD0FF 0%, #C792EA 40%, #F5C56B 85%, #FF8B9A 100%);
  -webkit-background-clip:text;background-clip:text;color:transparent;
  filter:drop-shadow(0 4px 30px rgba(160,140,255,.15));
  transition:filter .2s ease;
}
.team-footer:hover .team-name-xl{filter:drop-shadow(0 4px 40px rgba(160,140,255,.3))}
.team-id-row{margin-bottom:16px}
.team-id-pill{
  font:700 13px Cascadia Mono,Consolas,monospace;color:#0A1220;
  background:linear-gradient(90deg,#8FD0FF,#C4A8FF,#F0C675);
  padding:5px 14px;border-radius:999px;
  box-shadow:0 4px 20px rgba(140,180,255,.25);
}
.team-links-row{display:flex;justify-content:center;gap:12px;flex-wrap:wrap;margin:4px 0 14px}
.team-link-btn{
  display:inline-flex;align-items:center;gap:8px;
  padding:10px 18px;border-radius:999px;
  border:1px solid rgba(140,200,255,.22);
  background:linear-gradient(180deg, rgba(30,48,66,.9), rgba(16,26,38,.95));
  color:#D0E6F6;font:650 13px "Segoe UI",sans-serif;
  text-decoration:none;
  transition:all .18s ease;
  box-shadow:0 6px 20px rgba(0,0,0,.2);
}
.team-link-btn:hover{
  transform:translateY(-2px) scale(1.02);
  border-color:rgba(160,210,255,.5);
  background:linear-gradient(135deg, rgba(50,100,150,.55), rgba(80,60,120,.4));
  color:#fff;
  box-shadow:0 10px 28px rgba(80,140,220,.25);
}
.team-tagline{
  color:#7A90A0;font:13px/1.5 "Segoe UI",sans-serif;font-style:italic;
  max-width:520px;margin:0 auto 8px;
}
.team-members-note{
  margin-top:6px;color:#556878;
  font:10px Cascadia Mono,Consolas,monospace;letter-spacing:.04em;
}
.footer-note-line{color:#5A6A78;font:10px Cascadia Mono,Consolas,monospace;margin-bottom:14px;opacity:.9}

.team-name-xl{position:relative;font:700 clamp(26px,3.5vw,36px)/1.05 Bahnschrift,Segoe UI,sans-serif;letter-spacing:-.03em;margin:4px 0 10px;background:linear-gradient(105deg,#8FD0FF 0%,#C792EA 45%,#F5C56B 100%);-webkit-background-clip:text;background-clip:text;color:transparent}
.team-id-row{display:flex;justify-content:center;align-items:center;gap:8px;margin-bottom:14px}
.team-id-label{font:700 9px Cascadia Mono,Consolas,monospace;letter-spacing:.16em;color:#6A7A88}
.team-id-pill{font:700 13px Cascadia Mono,Consolas,monospace;color:#0B1220;background:linear-gradient(90deg,#8FD0FF,#B8A0FF);padding:4px 12px;border-radius:999px}
.team-link-btn{display:inline-flex;align-items:center;gap:6px;padding:8px 14px;margin:0 4px;border-radius:999px;border:1px solid rgba(143,200,255,.25);background:rgba(20,32,44,.8);color:#C8DDEE;font:600 12px Segoe UI,sans-serif;text-decoration:none;transition:all .15s ease}
.team-link-btn:hover{border-color:#8FD0FF;color:#fff;background:rgba(40,70,100,.55);transform:translateY(-1px)}
.team-link-icon{opacity:.85}
.team-credit-row{position:relative;display:flex;justify-content:center;align-items:center;gap:8px}

.team-footer{
  margin-top:34px;padding:22px 18px 10px;border-top:1px solid rgba(255,255,255,.06);
  text-align:center;background:radial-gradient(ellipse at 50% 0%, rgba(80,140,200,.08), transparent 60%);
}
.footer-note-line{color:#5E6B76;font:10px Cascadia Mono,Consolas,monospace;margin-bottom:12px}
.team-credit-row{display:flex;justify-content:center;align-items:center;gap:6px;flex-wrap:wrap;margin-bottom:8px}
.team-credit-text{color:#8FA0AD;font:500 14px Segoe UI,sans-serif}
.team-heart{color:#FF6B8A;font-size:16px;animation:pulse-heart 1.8s ease-in-out infinite}
@keyframes pulse-heart{0%,100%{transform:scale(1);opacity:1}50%{transform:scale(1.15);opacity:.85}}
.team-name{font:700 16px Bahnschrift,Segoe UI,sans-serif;background:linear-gradient(90deg,#8FD0FF,#C792EA,#F5C56B);-webkit-background-clip:text;background-clip:text;color:transparent;letter-spacing:.02em}
.team-id{font:700 14px Cascadia Mono,Consolas,monospace;color:#8FD0FF}
.team-links-row{display:flex;justify-content:center;align-items:center;gap:6px;flex-wrap:wrap;margin-top:4px}
.team-link{color:#7EB6E0;font:600 12px Segoe UI,sans-serif;text-decoration:none;border-bottom:1px solid rgba(126,182,224,.35)}
.team-link:hover{color:#B8E0FF;border-bottom-color:#B8E0FF}
.team-sep{color:#4A5864}
.team-tagline{color:#6A7A88;font:12px Segoe UI,sans-serif;font-style:italic}
.team-members-note{margin-top:8px;color:#556370;font:10px Cascadia Mono,Consolas,monospace}

.footer-note{text-align:center;color:#5E6B76;font:9px Cascadia Mono,Consolas,monospace;margin-top:30px;padding-top:18px;border-top:1px solid var(--border2)}
.js-plotly-plot .plotly .modebar{background:transparent!important}
@media(max-width:1200px){.kpi-grid{grid-template-columns:repeat(3,1fr)}}
@media(max-width:850px){.sidebar{position:relative;width:100%;height:auto}.app-shell{display:block}.main-content{width:100%;padding:20px}.overview-grid,.chart-grid,.method-grid{grid-template-columns:1fr}.kpi-grid{grid-template-columns:repeat(2,1fr)}.topbar{display:block}.runtime-badges{justify-content:flex-start;margin-top:15px}}

/* HARD OVERRIDE: Dash React-Select / DCC dropdown surfaces */
.sidebar .dark-dropdown,
.sidebar .dark-dropdown > div,
.sidebar .dark-dropdown .Select,
.sidebar .dark-dropdown .Select-control,
.sidebar .dark-dropdown [class*="control"],
.sidebar .dark-dropdown [class*="Control"]{
  background-color:#111821 !important;
  background:#111821 !important;
  color:#EEF4F8 !important;
  border-color:#2A3743 !important;
  box-shadow:none !important;
}
.sidebar .dark-dropdown .Select-value,
.sidebar .dark-dropdown [class*="singleValue"],
.sidebar .dark-dropdown [class*="SingleValue"],
.sidebar .dark-dropdown [class*="placeholder"],
.sidebar .dark-dropdown [class*="Placeholder"]{
  background:transparent !important;
  color:#EAF2F7 !important;
  opacity:1 !important;
}
.sidebar .dark-dropdown [class*="placeholder"]{color:#8796A3 !important}
.sidebar .dark-dropdown .Select-menu-outer,
.sidebar .dark-dropdown [class*="menu"]{
  background:#0F161E !important;
  background-color:#0F161E !important;
  color:#EAF2F7 !important;
  border-color:#2A3743 !important;
  box-shadow:0 18px 45px rgba(0,0,0,.55) !important;
}
.sidebar .dark-dropdown .Select-option,
.sidebar .dark-dropdown [class*="option"]{
  background:#0F161E !important;
  color:#DDE7EE !important;
}
.sidebar .dark-dropdown .Select-option.is-focused,
.sidebar .dark-dropdown [class*="option--is-focused"]{
  background:#1B2D3C !important;
  color:#FFFFFF !important;
}
.sidebar .dark-dropdown .Select-option.is-selected,
.sidebar .dark-dropdown [class*="option--is-selected"]{
  background:#17384D !important;
  color:#FFFFFF !important;
}
.sidebar .dark-dropdown input,
.sidebar .dark-dropdown textarea{
  color:#EEF4F8 !important;
  background:transparent !important;
}
.sidebar .dark-dropdown svg{color:#9FB2C1 !important;fill:#9FB2C1 !important}
</style>
</head>"""
)


# DEEP STYLE INJECTION

app.index_string = app.index_string.replace("</head>", r"""
<style>
/* ============================================================
   DEEP ANALYTICS SYSTEM — executive dark / high-density / lucid
   ============================================================ */
.deep-overview,.deep-view{width:100%;}
.deep-hero{
  display:grid;
  grid-template-columns:minmax(0,1.62fr) minmax(290px,.68fr);
  gap:14px;
  padding:16px;
  border:1px solid #29485D;
  border-radius:20px;
  background:
    radial-gradient(circle at 8% 15%,rgba(143,208,255,.10),transparent 28%),
    radial-gradient(circle at 86% 15%,rgba(166,108,255,.08),transparent 24%),
    linear-gradient(135deg,#101C26 0%,#0D171F 52%,#090E14 100%);
  box-shadow:0 24px 65px rgba(0,0,0,.26);
  overflow:hidden;
  position:relative;
}
.deep-hero:before{
  content:"";
  position:absolute;
  inset:-1px;
  pointer-events:none;
  background:linear-gradient(90deg,rgba(143,208,255,.05),transparent 45%,rgba(166,108,255,.03));
}
.deep-hero-copy-wrap{position:relative;z-index:1;padding:12px 13px 10px 12px;}
.deep-hero-kicker,.deep-kicker,.deep-section-kicker,.deep-kicker{
  font:700 8px/1.2 Cascadia Mono,Consolas,monospace;
  letter-spacing:.18em;
  color:#84C8F8;
}
.deep-hero-title{
  font:750 clamp(36px,4.5vw,60px)/.96 Bahnschrift,Segoe UI,sans-serif;
  letter-spacing:-.065em;
  margin:14px 0 12px;
  max-width:900px;
}
.deep-hero-copy{font-size:13px;line-height:1.7;color:#B2C0CA;max-width:930px;margin:0;}
.deep-chip-row{display:flex;gap:7px;flex-wrap:wrap;margin-top:18px;}
.deep-chip{border:1px solid #2A4A5F;background:rgba(9,17,24,.76);padding:7px 10px;border-radius:999px;color:#AFC0CC;font:700 8px Cascadia Mono,Consolas,monospace;letter-spacing:.05em;}
.deep-scope-card{
  position:relative;
  z-index:1;
  display:flex;
  flex-direction:column;
  justify-content:center;
  min-height:225px;
  padding:18px;
  border:1px solid #2E4B5E;
  border-radius:17px;
  background:linear-gradient(145deg,#111C25,#0C141B);
  box-shadow:0 16px 42px rgba(0,0,0,.24);
}
.deep-hero-number{font:750 42px Bahnschrift,Segoe UI,sans-serif;letter-spacing:-.055em;margin-top:10px;}
.deep-hero-note{font-size:10px;color:#7E92A0;line-height:1.5;}
.deep-stat-strip{display:grid;grid-template-columns:1fr;gap:5px;margin-top:18px;padding-top:13px;border-top:1px solid #22303C;}
.deep-stat-strip span{font:700 9px Cascadia Mono,Consolas,monospace;color:#95AAB8;}
.deep-insight-grid{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:9px;margin:12px 0 4px;}
.deep-insight-grid.four{grid-template-columns:repeat(4,minmax(0,1fr));}
.deep-section{margin-top:23px;}
.deep-section-head{display:flex;justify-content:space-between;align-items:end;gap:20px;margin-bottom:11px;padding:0 1px;}
.deep-section-title{font:650 23px/1.05 Bahnschrift,Segoe UI,sans-serif;letter-spacing:-.03em;margin:6px 0 4px;}
.deep-section-note{font-size:10px;line-height:1.55;color:#778A98;max-width:760px;margin:0;text-align:right;}
.deep-chart-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;align-items:stretch;}.deep-chart-grid .panel,.deep-chart-grid .deep-graph-panel{min-height:380px}.deep-graph-panel .js-plotly-plot{min-height:360px}
.deep-panel{
  min-width:0;
  min-height:100px;
  padding:7px 8px 10px;
  border-radius:15px;
  background:
    radial-gradient(circle at 90% 0%,rgba(143,208,255,.025),transparent 28%),
    rgba(14,21,28,.94);
  border:1px solid #263542;
  box-shadow:0 12px 34px rgba(0,0,0,.15);
  overflow:hidden;
}
.deep-panel:hover{border-color:#335268;box-shadow:0 16px 40px rgba(0,0,0,.22);}
.deep-panel .js-plotly-plot{width:100%!important;}
.deep-panel .plot-container{border-radius:11px;overflow:hidden;}
.deep-panel .modebar-group{background:rgba(11,16,22,.72);border-radius:8px;}
.deep-panel .modebar-btn{color:#9AAEBC!important;}
.deep-panel .modebar-btn:hover{color:#EEF4F8!important;}
.deep-note{padding:5px 7px 0;color:#627582;font-size:9px;line-height:1.45;}
.deep-map-panel{margin-top:14px;padding-bottom:5px;}
.deep-queue-panel{margin-top:3px;}
.deep-queue-header{margin-top:26px;}
.deep-overview .kpi-card,.deep-view .kpi-card{min-height:86px;}
.deep-overview .kpi-card .kpi-value,.deep-view .kpi-card .kpi-value{font-size:20px;}
.deep-overview .section-kicker,.deep-view .section-kicker{color:#8CCCF6;}
.deep-overview .js-plotly-plot .plotly .legendtext,
.deep-view .js-plotly-plot .plotly .legendtext{font-size:9px!important;}
.deep-overview .js-plotly-plot .plotly .xtick text,
.deep-overview .js-plotly-plot .plotly .ytick text,
.deep-view .js-plotly-plot .plotly .xtick text,
.deep-view .js-plotly-plot .plotly .ytick text{font-family:Segoe UI,Arial,sans-serif!important;}
.deep-overview .dash-table-container,.deep-view .dash-table-container{font-size:11px;}
@media(max-width:1180px){
  .deep-insight-grid{grid-template-columns:repeat(3,minmax(0,1fr));}
  .deep-insight-grid.four{grid-template-columns:repeat(2,minmax(0,1fr));}
}
@media(max-width:900px){
  .deep-hero{grid-template-columns:1fr;}
  .deep-section-head{display:block;}
  .deep-section-note{text-align:left;margin-top:4px;}
  .deep-chart-grid{grid-template-columns:1fr;}
}
@media(max-width:650px){
  .deep-insight-grid,.deep-insight-grid.four{grid-template-columns:repeat(2,minmax(0,1fr));}
  .deep-hero-title{font-size:38px;}
}

.analytics-state-panel{min-height:220px;display:flex;flex-direction:column;justify-content:center;align-items:flex-start;padding:34px;margin-top:24px;}
.analytics-status-row{display:flex;flex-wrap:wrap;gap:8px;margin:16px 0 4px;}
.analytics-status-chip{
  display:inline-flex;align-items:center;padding:7px 10px;border-radius:999px;
  border:1px solid rgba(143,208,255,.15);background:rgba(11,22,31,.72);
  color:#A9BDCA;font:700 9px/1 var(--mono,Consolas,monospace);letter-spacing:.08em;
}
.analytics-status-chip.warning{border-color:rgba(245,197,107,.22);color:#E5C98E;}
.deep-ask-ai-btn:focus-visible{outline:2px solid rgba(143,208,255,.8);outline-offset:2px;}
.analytics-state-panel.error{border-color:#6E3B3F;}
.analytics-error-text{color:#E6A5AC!important;white-space:pre-wrap;word-break:break-word;}
#analytics-loading{width:100%;}
#analytics-loading .dash-spinner{margin-top:32px;margin-bottom:32px;}

/* ============================================================
   PREMIUM UI PASS — VISUAL ONLY
   Apple-inspired / modern analytics control center
   ============================================================ */

html, body{
  -webkit-font-smoothing:antialiased!important;
  -moz-osx-font-smoothing:grayscale!important;
  text-rendering:optimizeLegibility!important;
}

.main-content{
  background:
    radial-gradient(circle at 78% -8%, rgba(126,200,255,.045), transparent 30%),
    radial-gradient(circle at 15% 2%, rgba(199,146,234,.025), transparent 24%),
    linear-gradient(180deg,#080D13 0%,#070B10 100%)!important;
}

.scope-banner{
  margin-top:20px!important;
  position:relative!important;
  isolation:isolate!important;
}

.scope-banner:before{
  content:"";
  position:absolute;
  inset:-1px;
  border-radius:20px;
  padding:1px;
  background:linear-gradient(
    90deg,
    rgba(126,200,255,.55),
    rgba(199,146,234,.22) 48%,
    rgba(240,198,117,.32)
  );
  -webkit-mask:
    linear-gradient(#000 0 0) content-box,
    linear-gradient(#000 0 0);
  -webkit-mask-composite:xor;
  mask-composite:exclude;
  pointer-events:none;
  opacity:.75;
}

.banner{
  position:relative!important;
  overflow:hidden!important;
  min-height:98px!important;
  padding:17px 22px 16px 24px!important;
  border:1px solid rgba(126,200,255,.16)!important;
  border-radius:19px!important;
  background:
    radial-gradient(circle at 100% 0%,rgba(126,200,255,.12),transparent 32%),
    radial-gradient(circle at 72% 100%,rgba(199,146,234,.05),transparent 35%),
    linear-gradient(135deg,rgba(17,29,40,.97),rgba(10,17,24,.985))!important;
  box-shadow:
    0 18px 55px rgba(0,0,0,.34),
    0 1px 0 rgba(255,255,255,.045) inset,
    0 -12px 28px rgba(126,200,255,.025) inset!important;
  backdrop-filter:blur(14px) saturate(125%)!important;
  -webkit-backdrop-filter:blur(14px) saturate(125%)!important;
}

.banner:before{
  content:"";
  position:absolute;
  left:0;
  top:14px;
  bottom:14px;
  width:3px;
  border-radius:0 5px 5px 0;
  background:linear-gradient(180deg,#8FD0FF,#C792EA 58%,#F0C675);
  box-shadow:0 0 18px rgba(126,200,255,.28);
}

.banner:after{
  content:"";
  position:absolute;
  right:-20px;
  top:-28px;
  width:260px;
  height:120px;
  background:
    linear-gradient(rgba(143,208,255,.055) 1px, transparent 1px),
    linear-gradient(90deg, rgba(143,208,255,.055) 1px, transparent 1px);
  background-size:22px 22px;
  transform:perspective(420px) rotateY(-18deg) skewY(-4deg);
  opacity:.35;
  mask-image:linear-gradient(90deg,transparent,#000 35%,#000);
  -webkit-mask-image:linear-gradient(90deg,transparent,#000 35%,#000);
  pointer-events:none;
}

.banner-kicker{
  position:relative;
  z-index:1;
  font-size:9px!important;
  letter-spacing:.22em!important;
  color:#8FD0FF!important;
  text-shadow:0 0 14px rgba(143,208,255,.22);
}

.banner-main{
  position:relative;
  z-index:1;
  font-size:20px!important;
  line-height:1.1!important;
  font-weight:650!important;
  letter-spacing:-.025em!important;
  margin-top:4px!important;
  color:#F2F7FB!important;
  text-shadow:0 1px 0 rgba(255,255,255,.04);
}

.banner-sub{
  position:relative;
  z-index:1;
  font-size:10.5px!important;
  line-height:1.45!important;
  color:#9DB0BE!important;
  margin-top:6px!important;
}

/* KPI matrix */
.kpi-grid{
  display:grid!important;
  grid-template-columns:repeat(6,minmax(0,1fr))!important;
  gap:11px!important;
  margin-top:16px!important;
}

.kpi-card{
  min-height:128px!important;
  padding:17px 16px 15px!important;
  border:1px solid rgba(132,166,195,.14)!important;
  border-radius:17px!important;
  position:relative!important;
  overflow:hidden!important;
  isolation:isolate!important;
  background:
    radial-gradient(circle at 90% 10%,rgba(126,200,255,.055),transparent 34%),
    linear-gradient(145deg,rgba(18,27,37,.98),rgba(10,16,23,.99))!important;
  box-shadow:
    0 14px 38px rgba(0,0,0,.25),
    0 1px 0 rgba(255,255,255,.035) inset,
    0 -10px 28px rgba(0,0,0,.12) inset!important;
  backdrop-filter:blur(10px) saturate(115%)!important;
  -webkit-backdrop-filter:blur(10px) saturate(115%)!important;
  transition:
    transform .22s cubic-bezier(.2,.8,.2,1),
    border-color .22s ease,
    box-shadow .22s ease,
    background .22s ease!important;
}

.kpi-card:before{
  content:"";
  position:absolute;
  left:14px;
  right:14px;
  top:0;
  height:1px;
  background:linear-gradient(90deg,transparent,rgba(143,208,255,.65),transparent);
  opacity:.75;
  z-index:2;
}

.kpi-card:after{
  content:"";
  position:absolute;
  right:-42px;
  bottom:-58px;
  width:145px;
  height:145px;
  border-radius:50%;
  border:1px solid rgba(126,200,255,.08);
  box-shadow:0 0 0 18px rgba(126,200,255,.018),0 0 40px rgba(126,200,255,.035);
  pointer-events:none;
}

.kpi-card:hover{
  transform:translateY(-4px)!important;
  border-color:rgba(143,208,255,.27)!important;
  box-shadow:
    0 20px 46px rgba(0,0,0,.32),
    0 0 0 1px rgba(143,208,255,.04) inset,
    0 1px 0 rgba(255,255,255,.05) inset!important;
}

.kpi-card.warning{
  border-color:rgba(240,198,117,.24)!important;
  background:
    radial-gradient(circle at 92% 8%,rgba(240,198,117,.065),transparent 35%),
    linear-gradient(145deg,rgba(27,25,21,.98),rgba(12,16,21,.99))!important;
}

.kpi-card.warning:hover{
  border-color:rgba(240,198,117,.43)!important;
}

.kpi-card.danger{
  border-color:rgba(232,90,90,.25)!important;
  background:
    radial-gradient(circle at 92% 8%,rgba(232,90,90,.07),transparent 35%),
    linear-gradient(145deg,rgba(29,22,25,.98),rgba(12,16,21,.99))!important;
}

.kpi-card.danger:hover{
  border-color:rgba(232,90,90,.44)!important;
}

.kpi-card:nth-child(2) .kpi-value,
.kpi-card:nth-child(6) .kpi-value{
  text-shadow:0 0 24px rgba(143,208,255,.08);
}

.kpi-label{
  font-size:8px!important;
  letter-spacing:.18em!important;
  color:#7F98AA!important;
  margin-bottom:0!important;
  position:relative;
  z-index:2;
}

.kpi-value{
  font-size:27px!important;
  line-height:1!important;
  letter-spacing:-.045em!important;
  font-weight:700!important;
  margin-top:12px!important;
  color:#F2F7FB!important;
  position:relative;
  z-index:2;
}

.kpi-hint{
  font-size:9.5px!important;
  line-height:1.35!important;
  color:#6F8494!important;
  margin-top:9px!important;
  position:relative;
  z-index:2;
}

#kpi-grid .kpi-card:nth-child(1),
#kpi-grid .kpi-card:nth-child(2),
#kpi-grid .kpi-card:nth-child(3),
#kpi-grid .kpi-card:nth-child(4),
#kpi-grid .kpi-card:nth-child(5),
#kpi-grid .kpi-card:nth-child(6){
  box-shadow:
    0 14px 38px rgba(0,0,0,.25),
    0 1px 0 rgba(255,255,255,.035) inset;
}

/* Secondary KPI row — visually quieter but still premium */
.secondary-kpi-grid{
  display:grid!important;
  grid-template-columns:repeat(6,minmax(0,1fr))!important;
  gap:11px!important;
  margin-top:11px!important;
}

.secondary-kpi-grid .kpi-card{
  min-height:94px!important;
  padding:13px 14px 12px!important;
  border-radius:15px!important;
  background:
    linear-gradient(145deg,rgba(15,23,31,.98),rgba(9,15,21,.99))!important;
}

.secondary-kpi-grid .kpi-label{
  font-size:7.5px!important;
  letter-spacing:.16em!important;
}

.secondary-kpi-grid .kpi-value{
  font-size:22px!important;
  margin-top:8px!important;
}

.secondary-kpi-grid .kpi-hint{
  font-size:8.5px!important;
  margin-top:6px!important;
}

/* Consistent subtle hierarchy for the first-screen content */
#scope-banner + #kpi-grid{
  position:relative;
}

#scope-banner + #kpi-grid:before{
  content:"LIVE ANALYTICS";
  position:absolute;
  right:4px;
  top:-13px;
  padding:0 5px;
  font:700 7px Cascadia Mono,Consolas,monospace;
  letter-spacing:.18em;
  color:#526A7A;
  pointer-events:none;
}

@media(max-width:1250px){
  .kpi-grid,
  .secondary-kpi-grid{
    grid-template-columns:repeat(3,minmax(0,1fr))!important;
  }
}

@media(max-width:760px){
  .kpi-grid,
  .secondary-kpi-grid{
    grid-template-columns:repeat(2,minmax(0,1fr))!important;
  }
  .banner{
    min-height:auto!important;
    padding:15px 17px 14px 20px!important;
  }
  .banner-main{
    font-size:18px!important;
  }
}

@media(prefers-reduced-motion:reduce){
  .kpi-card{
    transition:none!important;
  }
  .kpi-card:hover{
    transform:none!important;
  }
}


/* ============================================================
   LOADING EXPERIENCE + PREMIUM TEAM FOOTER
   Visual-only: no application logic / callbacks / API behavior.
   ============================================================ */

body{
  background:#05090E!important;
  overflow-x:hidden;
}

/* ---------- Initial boot screen ---------- */
.initial-loader{
  position:fixed;
  inset:0;
  z-index:99999;
  display:flex;
  align-items:center;
  justify-content:center;
  overflow:hidden;
  background:
    radial-gradient(circle at 50% 45%,rgba(89,156,215,.12),transparent 25%),
    radial-gradient(circle at 75% 75%,rgba(199,146,234,.075),transparent 28%),
    radial-gradient(circle at 20% 20%,rgba(143,208,255,.045),transparent 24%),
    #05090E;
 animation:boot-exit 900ms cubic-bezier(.65,0,.25,1) 2.55s forwards;
}
.initial-loader:before{
  content:"";
  position:absolute;
  inset:0;
  background:
    linear-gradient(rgba(143,208,255,.035) 1px,transparent 1px),
    linear-gradient(90deg,rgba(143,208,255,.035) 1px,transparent 1px);
  background-size:38px 38px;
  mask-image:radial-gradient(circle at 50% 50%,#000 0%,rgba(0,0,0,.7) 42%,transparent 78%);
  -webkit-mask-image:radial-gradient(circle at 50% 50%,#000 0%,rgba(0,0,0,.7) 42%,transparent 78%);
  opacity:.5;
}
.initial-loader:after{
  content:"";
  position:absolute;
  inset:0;
  background:linear-gradient(transparent 0%,rgba(255,255,255,.025) 50%,transparent 100%);
  transform:translateY(-100%);
  animation:scanline 1.65s linear infinite;
  pointer-events:none;
}
.boot-card{
  position:relative;
  z-index:3;
  width:min(660px,calc(100vw - 44px));
  padding:34px 38px 30px;
  border:1px solid rgba(143,208,255,.20);
  border-radius:28px;
  background:
    radial-gradient(circle at 15% 0%,rgba(143,208,255,.08),transparent 32%),
    radial-gradient(circle at 100% 100%,rgba(199,146,234,.08),transparent 34%),
    linear-gradient(145deg,rgba(16,27,37,.93),rgba(7,13,19,.97));
  box-shadow:
    0 35px 100px rgba(0,0,0,.52),
    0 1px 0 rgba(255,255,255,.06) inset,
    0 0 90px rgba(89,156,215,.08);
  backdrop-filter:blur(22px) saturate(130%);
  -webkit-backdrop-filter:blur(22px) saturate(130%);
  overflow:hidden;
  animation:boot-card-in 800ms cubic-bezier(.16,1,.3,1) both;
}
.boot-card:before{
  content:"";
  position:absolute;
  left:10%;right:10%;top:0;height:1px;
  background:linear-gradient(90deg,transparent,#8FD0FF 28%,#C792EA 52%,#F0C675 74%,transparent);
  box-shadow:0 0 24px rgba(143,208,255,.48);
}
.boot-card:after{
  content:"";
  position:absolute;
  left:-25%;bottom:-60%;
  width:150%;height:100%;
  background:radial-gradient(ellipse at center,rgba(143,208,255,.05),transparent 62%);
  pointer-events:none;
}
.boot-kicker{
  position:relative;z-index:2;
  font:700 10px/1 Cascadia Mono,Consolas,monospace;
  letter-spacing:.28em;
  color:#8FD0FF;
  text-align:center;
  opacity:0;
  animation:boot-fade-up .7s .15s ease forwards;
}
.boot-title{
  position:relative;z-index:2;
  margin-top:10px;
  font:800 clamp(27px,5vw,43px)/1.05 Bahnschrift,"Segoe UI",sans-serif;
  letter-spacing:-.045em;
  text-align:center;
  background:linear-gradient(110deg,#FFFFFF 0%,#BFE4FF 36%,#C792EA 68%,#F2D090 100%);
  -webkit-background-clip:text;background-clip:text;color:transparent;
  filter:drop-shadow(0 7px 32px rgba(143,208,255,.12));
  opacity:0;
  animation:boot-fade-up .8s .25s ease forwards;
}
.boot-status-row{
  position:relative;z-index:2;
  display:flex;align-items:center;justify-content:center;gap:9px;
  margin-top:17px;
  opacity:0;
  animation:boot-fade-up .7s .4s ease forwards;
}
.boot-live-dot{
  color:#7FE1A5;
  font-size:10px;
  filter:drop-shadow(0 0 7px rgba(127,225,165,.7));
  animation:boot-pulse 1.2s ease-in-out infinite;
}
.boot-status{font:600 11px Cascadia Mono,Consolas,monospace;color:#91A8B8;letter-spacing:.05em}
.boot-meta-row{
  position:relative;z-index:2;
  display:flex;align-items:center;justify-content:space-between;gap:16px;
  margin-top:26px;
  opacity:0;
  animation:boot-fade-up .7s .55s ease forwards;
}
.boot-caption{font:10px "Segoe UI",Arial,sans-serif;color:#607789;letter-spacing:.02em}
.boot-team{font:700 9px Cascadia Mono,Consolas,monospace;letter-spacing:.18em;color:#6E8FA4;white-space:nowrap}
.boot-progress{
  position:relative;z-index:2;
  height:3px;margin-top:13px;border-radius:99px;
  background:rgba(145,180,205,.10);overflow:hidden;
  opacity:0;
  animation:boot-fade-up .7s .65s ease forwards;
}
.boot-progress-fill{
  width:28%;height:100%;border-radius:99px;
  background:linear-gradient(90deg,#8FD0FF,#C792EA,#F0C675);
  box-shadow:0 0 18px rgba(143,208,255,.5);
  animation:boot-progress 2.05s cubic-bezier(.18,.73,.26,1) forwards;
}
.boot-orbit{
  position:absolute;
  z-index:1;
  border:1px solid rgba(143,208,255,.08);
  border-radius:50%;
  box-shadow:0 0 45px rgba(143,208,255,.035);
  pointer-events:none;
}
.boot-orbit-a{width:780px;height:780px;animation:orbit-spin 16s linear infinite}
.boot-orbit-b{width:490px;height:490px;border-color:rgba(199,146,234,.09);animation:orbit-spin-rev 11s linear infinite}
.boot-noise{
  position:absolute;inset:-50%;z-index:0;opacity:.035;pointer-events:none;
  background-image:radial-gradient(rgba(255,255,255,.6) .6px,transparent .6px);
  background-size:5px 5px;transform:rotate(12deg);
}
@keyframes boot-card-in{from{opacity:0;transform:translateY(18px) scale(.985);filter:blur(4px)}to{opacity:1;transform:none;filter:none}}
@keyframes boot-fade-up{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
@keyframes boot-progress{from{transform:translateX(-115%) scaleX(.8)}to{transform:translateX(350%) scaleX(1.35)}}
@keyframes boot-pulse{0%,100%{opacity:.55;transform:scale(.9)}50%{opacity:1;transform:scale(1.15)}}
@keyframes orbit-spin{to{transform:rotate(360deg) translateZ(0)}}
@keyframes orbit-spin-rev{to{transform:rotate(-360deg) translateZ(0)}}
@keyframes scanline{to{transform:translateY(100%)}}
@keyframes boot-exit{to{opacity:0;visibility:hidden;transform:scale(1.015)}}

/* ---------- Real bottom-of-page team section ---------- */
/* ============================================================
   TEAM FOOTER — SEAMLESS AMBIENT SECTION
   No box. No border. No rectangle. The aurora is continuous.
   ============================================================ */

/* ── Wrapper: bleeds into the page, no visual container ── */
.team-footer{
  position:relative;
  isolation:isolate;
  overflow:visible;               /* allow aurora to extend outward */
  margin:96px -34px 0;            /* cancel .main-content padding, extend full-bleed */
  padding:88px 34px 64px;
  border:none;
  border-radius:0;
  background:transparent;         /* page background shows through */
  box-shadow:none;
  text-align:center;
  transition:opacity .9s ease, transform .9s ease;
}

/* ── Seamless top dissolve — hides where footer "begins" ── */
.team-footer:before{
  content:"";
  position:absolute;
  top:-1px; left:0; right:0;
  height:180px;
  pointer-events:none;
  z-index:0;
  background:linear-gradient(180deg,
    rgba(7,11,16,1) 0%,
    rgba(7,11,16,.85) 25%,
    rgba(7,11,16,.45) 55%,
    rgba(7,11,16,.12) 82%,
    transparent 100%);
}

/* ── Ambient aurora field behind the whole section ── */
.team-footer:after{
  content:"";
  position:absolute;
  inset:-140px -60px -100px;
  pointer-events:none;
  z-index:0;
  background:
    radial-gradient(ellipse 40% 34% at 22% 62%, rgba(110,255,196,.115), transparent 68%),
    radial-gradient(ellipse 36% 30% at 78% 42%, rgba(168,130,255,.10),  transparent 68%),
    radial-gradient(ellipse 46% 30% at 50% 76%, rgba(120,210,255,.075), transparent 70%),
    radial-gradient(ellipse 30% 24% at 12% 38%, rgba(190,255,220,.06),  transparent 68%),
    radial-gradient(ellipse 32% 26% at 88% 74%, rgba(140,180,255,.07),  transparent 68%);
  filter: blur(46px);
  mix-blend-mode: screen;
  opacity:.85;
  animation: footerAurora 44s ease-in-out infinite alternate;
  will-change:transform;
}

/* ── Star pinprick layer (HD detail, sits above the aurora) ── */
.team-footer .team-footer-inner{
  position:relative;
  z-index:2;
  width:min(860px,100%);
  margin:0 auto;
  display:flex;
  flex-direction:column;
  align-items:center;
  justify-content:center;
  text-align:center;
}

/* ── Kill the old glow div (aurora handles it now) ── */
.team-footer-glow{ display:none !important; }

/* ── Dissolve the star-pattern from the old design ── */
.team-footer > *{ position:relative; z-index:2; }

/* ── Signature mark — becomes the section's "north star" ── */
.team-brand-marks{
  display:flex;
  justify-content:center;
  align-items:center;
  margin-bottom:22px;
  position:relative;
}
.team-brand-marks:before{
  content:"";
  position:absolute;
  inset:-38px;
  border-radius:50%;
  background:radial-gradient(circle, rgba(143,208,255,.14), transparent 62%);
  filter:blur(28px);
  animation: sigPulse 6.5s ease-in-out infinite;
  pointer-events:none;
  z-index:-1;
}

/* ── Copy block ── */
.team-footer-copy{
  width:min(760px,100%);
  text-align:center;
}
.team-footer-copy .footer-note-line,
.team-footer-copy .team-eyebrow,
.team-footer-copy .team-credit-row,
.team-footer-copy .team-id-row,
.team-footer-copy .team-links-row{
  justify-content:center !important;
}
.team-footer-copy .team-tagline,
.team-footer-copy .team-members-note{
  margin-left:auto !important;
  margin-right:auto !important;
  text-align:center !important;
}

/* ── Ambient keyframes ── */
@keyframes footerAurora{
  0%   { transform:translate3d(-5%,  -2%, 0) scale(1.05); }
  33%  { transform:translate3d( 4%,   5%, 0) scale(1.11); }
  66%  { transform:translate3d( 6%,  -3%, 0) scale(1.07); }
  100% { transform:translate3d(-3%,   6%, 0) scale(1.13); }
}
@keyframes sigPulse{
  0%,100% { opacity:.55; transform:scale(1);    }
  50%     { opacity:1;   transform:scale(1.18); }
}

/* ── Reveal states — still respect the loading gate ── */
.team-footer-pending{
  visibility:hidden !important;
  opacity:0 !important;
  pointer-events:none !important;
  transform:translateY(22px) !important;
  transition:none !important;
  max-height:0 !important;
  overflow:hidden !important;
  padding:0 !important;
  margin:0 !important;
}
.team-footer-ready{
  visibility:visible !important;
  opacity:1 !important;
  pointer-events:auto !important;
  transform:none !important;
  animation: footerReveal 1.05s cubic-bezier(.16,1,.3,1) both;
}
@keyframes footerReveal{
  from { opacity:0; transform:translateY(24px); }
  to   { opacity:1; transform:none; }
}

/* ── Full-bleed override: kill any lingering container look ── */
.team-footer.team-footer-ready{
  border:none !important;
  border-radius:0 !important;
  box-shadow:none !important;
  background:transparent !important;
}

@media(prefers-reduced-motion:reduce){
  .team-footer:after,
  .team-brand-marks:before{ animation:none !important; }
}

/* ============================================================
   TEAM INFO MODAL + SIGNATURE BRAND SYSTEM
   Visual / interaction layer only
   ============================================================ */

/* Signature model logo: CSS/vector construction, crisp at any DPI. */
.model-logo{
  position:relative!important;
  width:52px!important;
  height:52px!important;
  flex:0 0 52px!important;
  border-radius:16px!important;
  display:grid!important;
  place-items:center!important;
  overflow:hidden!important;
  border:1px solid rgba(143,208,255,.30)!important;
  background:
    radial-gradient(circle at 30% 20%,rgba(143,208,255,.20),transparent 38%),
    linear-gradient(145deg,#142536,#0A111A)!important;
  box-shadow:
    0 10px 28px rgba(0,0,0,.30),
    0 0 28px rgba(126,200,255,.10),
    inset 0 1px 0 rgba(255,255,255,.08)!important;
  isolation:isolate!important;
  transition:box-shadow .3s ease,border-color .3s ease,transform .3s cubic-bezier(.34,1.56,.64,1)!important;
}
.model-logo:before{
  content:"";
  position:absolute;
  inset:7px;
  border:1px solid rgba(143,208,255,.20);
  border-radius:13px;
  transform:rotate(45deg);
  box-shadow:0 0 22px rgba(143,208,255,.08);
}
.model-logo:after{
  content:"";
  position:absolute;
  width:26px;
  height:26px;
  border-radius:50%;
  border:1px solid rgba(199,146,234,.28);
  box-shadow:0 0 22px rgba(199,146,234,.12);
}
.model-logo-orbit{
  position:absolute!important;
  width:38px!important;
  height:18px!important;
  border:1px solid rgba(240,198,117,.30)!important;
  border-left-color:transparent!important;
  border-right-color:transparent!important;
  border-radius:50%!important;
  transform:rotate(-28deg)!important;
  animation:model-orbit 6s linear infinite!important;
}
.model-logo-core{
  width:11px!important;
  height:11px!important;
  border-radius:50%!important;
  background:radial-gradient(circle at 35% 30%,#FFFFFF 0%,#8FD0FF 44%,#7F5CE1 100%)!important;
  box-shadow:0 0 18px rgba(143,208,255,.55)!important;
  z-index:3!important;
}
.model-logo-bar{
  position:absolute!important;
  bottom:10px!important;
  width:3px!important;
  border-radius:99px!important;
  background:linear-gradient(180deg,#8FD0FF,#C792EA)!important;
  box-shadow:0 0 7px rgba(143,208,255,.22)!important;
  transform-origin:bottom!important;
  z-index:3!important;
}
.model-logo .bar-a{height:7px!important;left:13px!important;transform:rotate(-13deg)}
.model-logo .bar-b{height:12px!important;left:19px!important;transform:rotate(0)}
.model-logo .bar-c{height:9px!important;left:25px!important;transform:rotate(13deg)}
.model-logo-ai{
  position:absolute!important;
  right:6px!important;
  top:5px!important;
  font:800 7px/1 Cascadia Mono,Consolas,monospace!important;
  letter-spacing:.08em!important;
  color:#F0C675!important;
  text-shadow:0 0 9px rgba(240,198,117,.35)!important;
  z-index:4!important;
}
@keyframes model-orbit{to{transform:rotate(332deg)}}

/* ============================================================
   INTERACTIVE LOGO STAGE  —  footer cosmic animation
   ============================================================ */
.model-logo-interactive-wrap{
  display:flex;align-items:center;justify-content:center;
}
.model-logo-stage{
  position:relative;
  width:96px;height:96px;
  display:flex;align-items:center;justify-content:center;
  cursor:pointer;
  border-radius:50%;
  user-select:none;
  -webkit-tap-highlight-color:transparent;
}

/* ─── Aurora spinning cones ─────────────────────────────── */
.mlg-aurora{
  position:absolute;
  inset:0;
  border-radius:50%;
  pointer-events:none;
}
.mlg-a1{
  background:conic-gradient(
    from 0deg,
    transparent 0%,rgba(143,208,255,.22) 20%,
    transparent 42%,rgba(199,146,234,.16) 68%,
    transparent 100%
  );
  animation:mlg-spin 10s linear infinite;
}
.mlg-a2{
  inset:8px;
  background:conic-gradient(
    from 120deg,
    transparent 0%,rgba(240,198,117,.15) 25%,
    transparent 55%,rgba(143,208,255,.12) 80%,
    transparent 100%
  );
  animation:mlg-spin 16s linear infinite reverse;
}
.mlg-a3{
  inset:18px;
  background:conic-gradient(
    from 240deg,
    transparent 0%,rgba(199,146,234,.25) 18%,
    rgba(143,208,255,.12) 36%,
    transparent 65%,transparent 100%
  );
  animation:mlg-spin 7s linear infinite;
  filter:blur(2px);
}

/* ─── Pulsing orbital rings ──────────────────────────────── */
.mlg-ring{
  position:absolute;
  border-radius:50%;
  border:1px solid transparent;
  pointer-events:none;
}
.mlg-r1{inset:2px;border-color:rgba(143,208,255,.22);animation:mlg-pulse 4.5s ease-in-out infinite;}
.mlg-r2{inset:10px;border-color:rgba(199,146,234,.18);animation:mlg-pulse 4.5s ease-in-out infinite 1.5s;}
.mlg-r3{inset:18px;border-color:rgba(240,198,117,.14);animation:mlg-pulse 4.5s ease-in-out infinite 3s;}

/* ─── Floating star-dots ─────────────────────────────────── */
.mlg-star{
  position:absolute;
  border-radius:50%;
  background:#fff;
  pointer-events:none;
}
.s1{width:3px;height:3px;top:7px;left:22px;animation:mlg-twinkle 2.9s ease-in-out infinite;}
.s2{width:2px;height:2px;top:12px;right:18px;animation:mlg-twinkle 3.7s ease-in-out infinite .6s;}
.s3{width:3px;height:3px;bottom:10px;left:17px;animation:mlg-twinkle 2.3s ease-in-out infinite 1.1s;}
.s4{width:2px;height:2px;bottom:7px;right:24px;animation:mlg-twinkle 3.1s ease-in-out infinite .3s;}
.s5{width:2px;height:2px;top:32px;left:4px;animation:mlg-twinkle 4.1s ease-in-out infinite 1.9s;}
.s6{width:3px;height:3px;top:28px;right:5px;animation:mlg-twinkle 3.4s ease-in-out infinite 2.4s;
    background:rgba(240,198,117,.9);}

/* ─── Neural arc fragments ───────────────────────────────── */
.mlg-arc{
  position:absolute;
  border-radius:50%;
  border:1px dashed transparent;
  pointer-events:none;
}
.arc1{width:74px;height:40px;top:16px;left:11px;
  border-top-color:rgba(143,208,255,.14);
  animation:mlg-spin 22s linear infinite;}
.arc2{width:50px;height:28px;top:22px;left:23px;
  border-bottom-color:rgba(199,146,234,.12);
  animation:mlg-spin 18s linear infinite reverse;}

/* ─── The logo mark — centered & interactive ──────────────── */
.mlg-mark{
  position:absolute!important;
  top:50%!important;left:50%!important;
  transform:translate(-50%,-50%)!important;
  z-index:6!important;
  transition:
    transform .35s cubic-bezier(.34,1.56,.64,1),
    box-shadow .35s ease,
    border-color .35s ease!important;
}

/* ─── Click-ripple ───────────────────────────────────────── */
.mlg-ripple{
  position:absolute;
  inset:22px;
  border-radius:50%;
  background:radial-gradient(circle,rgba(143,208,255,.7),rgba(199,146,234,.4) 55%,transparent 75%);
  transform:scale(0);
  opacity:0;
  pointer-events:none;
  z-index:10;
}

/* ─── Hover text hint ────────────────────────────────────── */
.mlg-hint{
  position:absolute;
  bottom:-22px;
  left:50%;
  transform:translateX(-50%);
  font:700 7px/1 'Cascadia Mono',Consolas,monospace;
  letter-spacing:.12em;
  color:rgba(143,208,255,.75);
  white-space:nowrap;
  opacity:0;
  transition:opacity .25s;
  pointer-events:none;
  text-shadow:0 0 10px rgba(143,208,255,.5);
}

/* ─── HOVER ──────────────────────────────────────────────── */
.model-logo-stage:hover .mlg-mark{
  transform:translate(-50%,-50%) scale(1.14)!important;
  box-shadow:
    0 0 32px rgba(143,208,255,.5),
    0 0 64px rgba(199,146,234,.28),
    inset 0 1px 0 rgba(255,255,255,.18)!important;
  border-color:rgba(143,208,255,.6)!important;
}
.model-logo-stage:hover .mlg-a1{animation-duration:5.5s;}
.model-logo-stage:hover .mlg-a2{animation-duration:9s;}
.model-logo-stage:hover .mlg-r1{border-color:rgba(143,208,255,.48);}
.model-logo-stage:hover .mlg-r2{border-color:rgba(199,146,234,.38);}
.model-logo-stage:hover .mlg-r3{border-color:rgba(240,198,117,.30);}
.model-logo-stage:hover .mlg-hint{opacity:1;}
.model-logo-stage:hover .s1,.model-logo-stage:hover .s3,
.model-logo-stage:hover .s6{background:rgba(240,198,117,.95);}

/* ─── CLICK (via JS .mlg-clicked class) ─────────────────── */
.model-logo-stage:active .mlg-mark{
  transform:translate(-50%,-50%) scale(.93)!important;
}
.model-logo-stage.mlg-clicked .mlg-ripple{
  animation:mlg-ripple-burst .7s ease-out forwards;
}
.model-logo-stage.mlg-clicked .mlg-mark{
  animation:mlg-burst .45s ease-out!important;
}
.model-logo-stage.mlg-clicked .mlg-a1{animation-duration:2.5s;}
.model-logo-stage.mlg-clicked .mlg-r1,.model-logo-stage.mlg-clicked .mlg-r2{
  animation:mlg-ring-flash .5s ease-out;
}

/* ─── Keyframes ──────────────────────────────────────────── */
@keyframes mlg-spin{to{transform:rotate(360deg);}}
@keyframes mlg-pulse{
  0%,100%{opacity:.35;transform:scale(1);}
  50%{opacity:.9;transform:scale(1.07);}
}
@keyframes mlg-twinkle{
  0%,100%{opacity:.12;transform:scale(.65);}
  50%{opacity:1;transform:scale(1.4);}
}
@keyframes mlg-ripple-burst{
  0%{transform:scale(0);opacity:.9;}
  60%{opacity:.5;}
  100%{transform:scale(3.5);opacity:0;}
}
@keyframes mlg-burst{
  0%{transform:translate(-50%,-50%) scale(1);}
  35%{transform:translate(-50%,-50%) scale(1.28);filter:brightness(1.7) drop-shadow(0 0 18px #8FD0FF);}
  100%{transform:translate(-50%,-50%) scale(1);filter:brightness(1);}
}
@keyframes mlg-ring-flash{
  0%{opacity:.3;transform:scale(1);}
  40%{opacity:1;transform:scale(1.22);}
  100%{opacity:.35;transform:scale(1);}
}

/* Footer brand area */
.team-footer-inner{
  width:min(980px,100%)!important;
  margin:0 auto!important;
  display:grid!important;
  grid-template-columns:72px minmax(0,1fr)!important;
  align-items:center!important;
  gap:22px!important;
  text-align:left!important;
}
.team-brand-marks{
  display:flex!important;
  flex-direction:column!important;
  align-items:center!important;
  gap:11px!important;
}
.mplads-emblem{
  width:46px!important;
  height:46px!important;
  object-fit:cover!important;
  border-radius:50%!important;
  background:#fff!important;
  border:1px solid rgba(255,255,255,.25)!important;
  box-shadow:0 7px 22px rgba(0,0,0,.28),0 0 22px rgba(143,208,255,.08)!important;
  opacity:.96!important;
}
.team-footer-copy{text-align:left!important}
.team-footer-copy .team-credit-row{justify-content:flex-start!important}
.team-footer-copy .team-name-xl{
  margin-left:0!important;
  margin-right:0!important;
}
.team-footer-copy .team-id-row{justify-content:flex-start!important}
.team-footer-copy .team-links-row{justify-content:flex-start!important}
.team-footer-copy .team-tagline{margin-left:0!important;margin-right:0!important}
.team-footer-copy .team-members-note{margin-left:0!important;margin-right:0!important}
.team-member-count{
  display:inline-flex!important;
  align-items:center!important;
  padding:5px 10px!important;
  border-radius:999px!important;
  border:1px solid rgba(143,208,255,.15)!important;
  color:#91B6CA!important;
  background:rgba(143,208,255,.045)!important;
  font:700 8px/1 Cascadia Mono,Consolas,monospace!important;
  letter-spacing:.13em!important;
}

/* New interaction buttons */
.team-repo-btn,
.team-info-btn{
  cursor:pointer!important;
  min-width:190px!important;
  letter-spacing:.02em!important;
  font-weight:700!important;
}
.team-info-btn{
  border-color:rgba(199,146,234,.34)!important;
  background:
    linear-gradient(135deg,rgba(65,47,91,.62),rgba(18,28,41,.95))!important;
  color:#E8D9FA!important;
}
.team-info-btn:hover{
  border-color:rgba(199,146,234,.70)!important;
  box-shadow:
    0 13px 34px rgba(105,71,150,.23),
    0 0 26px rgba(199,146,234,.08)!important;
}
.team-repo-btn:hover{
  box-shadow:
    0 13px 34px rgba(70,130,190,.22),
    0 0 26px rgba(143,208,255,.07)!important;
}

/* ---------- fullscreen magnified seal viewer ---------- */
.seal-viewer{
  position:fixed!important;
  inset:0!important;
  z-index:120000!important;
  display:flex!important;
  align-items:center!important;
  justify-content:center!important;
  padding:18px!important;
  isolation:isolate!important;
}
.seal-viewer.is-hidden{display:none!important}
.seal-viewer-backdrop{
  position:absolute!important;
  inset:0!important;
  cursor:pointer!important;
  background:
    radial-gradient(circle at 50% 40%,rgba(143,208,255,.14),transparent 19%),
    radial-gradient(circle at 30% 70%,rgba(19,136,8,.08),transparent 26%),
    radial-gradient(circle at 72% 63%,rgba(255,153,51,.08),transparent 27%),
    rgba(2,6,10,.74)!important;
  backdrop-filter:blur(22px) saturate(128%)!important;
  -webkit-backdrop-filter:blur(22px) saturate(128%)!important;
  animation:seal-viewer-backdrop-in .28s ease both!important;
}
.seal-viewer-card{
  position:relative!important;
  z-index:2!important;
  width:min(720px,94vw)!important;
  max-height:calc(100vh - 32px)!important;
  min-height:0!important;
  display:flex!important;
  flex-direction:column!important;
  align-items:center!important;
  justify-content:flex-start!important;
  gap:0!important;
  padding:30px 34px 20px!important;
  border-radius:36px!important;
  border:1px solid rgba(143,208,255,.24)!important;
  background:
    radial-gradient(circle at 50% 4%,rgba(143,208,255,.12),transparent 28%),
    radial-gradient(circle at 6% 92%,rgba(19,136,8,.055),transparent 22%),
    radial-gradient(circle at 94% 92%,rgba(255,153,51,.055),transparent 22%),
    linear-gradient(155deg,rgba(15,27,39,.95),rgba(5,11,17,.98))!important;
  box-shadow:
    0 48px 140px rgba(0,0,0,.64),
    0 0 0 1px rgba(255,255,255,.025) inset,
    0 0 110px rgba(126,200,255,.12)!important;
  overflow:hidden!important;
  animation:seal-viewer-card-in .42s cubic-bezier(.16,1,.3,1) both!important;
}
.seal-viewer-card:before{
  content:"";
  position:absolute!important;
  inset:0!important;
  pointer-events:none!important;
  background:
    linear-gradient(rgba(143,208,255,.028) 1px,transparent 1px),
    linear-gradient(90deg,rgba(143,208,255,.028) 1px,transparent 1px);
  background-size:32px 32px!important;
  mask-image:linear-gradient(180deg,#000 0%,transparent 84%)!important;
  -webkit-mask-image:linear-gradient(180deg,#000 0%,transparent 84%)!important;
}
.seal-viewer-card:after{
  content:"";
  position:absolute!important;
  width:420px!important;
  height:2px!important;
  left:50%!important;
  top:116px!important;
  transform:translateX(-50%)!important;
  background:linear-gradient(90deg,transparent,rgba(143,208,255,.33),rgba(255,255,255,.72),rgba(143,208,255,.33),transparent)!important;
  filter:blur(.2px)!important;
  opacity:.62!important;
  pointer-events:none!important;
}
.seal-viewer-heading{
  position:relative!important;
  z-index:3!important;
  width:100%!important;
  text-align:center!important;
  padding:0 62px!important;
}
.seal-viewer-eyebrow{
  color:#6FA7C7!important;
  font:800 8px/1.1 Cascadia Mono,Consolas,monospace!important;
  letter-spacing:.34em!important;
  margin-bottom:9px!important;
  text-transform:uppercase!important;
}
.seal-viewer-title{
  display:flex!important;
  justify-content:center!important;
  align-items:baseline!important;
  gap:10px!important;
  flex-wrap:wrap!important;
  line-height:.96!important;
}
.seal-viewer-title-main{
  color:#F5F8FB!important;
  font:800 clamp(24px,4vw,38px)/.96 Bahnschrift,"Segoe UI",Arial,sans-serif!important;
  letter-spacing:-.045em!important;
  text-shadow:0 0 34px rgba(143,208,255,.13)!important;
}
.seal-viewer-title-accent{
  color:#8FD0FF!important;
  font:800 clamp(11px,1.8vw,15px)/1 Cascadia Mono,Consolas,monospace!important;
  letter-spacing:.17em!important;
  text-shadow:0 0 18px rgba(143,208,255,.25)!important;
}
.seal-viewer-subtitle{
  margin:12px auto 0!important;
  max-width:560px!important;
  color:#77909E!important;
  font:700 8px/1.55 Cascadia Mono,Consolas,monospace!important;
  letter-spacing:.11em!important;
}
.seal-viewer-status{
  display:inline-flex!important;
  align-items:center!important;
  justify-content:center!important;
  gap:7px!important;
  margin-top:13px!important;
  padding:6px 11px!important;
  border-radius:999px!important;
  border:1px solid rgba(98,211,155,.15)!important;
  background:rgba(98,211,155,.035)!important;
  box-shadow:0 0 22px rgba(98,211,155,.035)!important;
}
.seal-viewer-status-dot{
  width:5px!important;
  height:5px!important;
  flex:0 0 5px!important;
  border-radius:50%!important;
  background:#62D39B!important;
  box-shadow:0 0 0 3px rgba(98,211,155,.09),0 0 9px rgba(98,211,155,.60)!important;
  animation:brand-live-pulse 2.1s ease-in-out infinite!important;
}
.seal-viewer-status-text{
  color:#8BC9A9!important;
  font:800 7px/1 Cascadia Mono,Consolas,monospace!important;
  letter-spacing:.18em!important;
}
.seal-viewer-status-sep{
  color:#536976!important;
  font:700 8px/1 Cascadia Mono,Consolas,monospace!important;
}
.seal-viewer-status-muted{
  color:#5D7583!important;
  font:700 7px/1 Cascadia Mono,Consolas,monospace!important;
  letter-spacing:.14em!important;
}
.seal-viewer-seal-wrap{
  position:relative!important;
  z-index:2!important;
  width:300px!important;
  height:300px!important;
  flex:0 0 300px!important;
  margin:12px 0 2px!important;
  display:block!important;
}
.seal-viewer-seal-wrap:before,
.seal-viewer-seal-wrap:after{
  content:"";
  position:absolute!important;
  left:50%!important;
  top:50%!important;
  border-radius:50%!important;
  transform:translate(-50%,-50%)!important;
  pointer-events:none!important;
}
.seal-viewer-seal-wrap:before{
  width:282px!important;
  height:282px!important;
  border:1px solid rgba(143,208,255,.10)!important;
  box-shadow:
    0 0 34px rgba(143,208,255,.07),
    inset 0 0 34px rgba(143,208,255,.035)!important;
}
.seal-viewer-seal-wrap:after{
  width:246px!important;
  height:246px!important;
  border:1px dashed rgba(255,153,51,.10)!important;
  animation:seal-orbit-guide 18s linear infinite!important;
}
.seal-viewer-seal{
  position:absolute!important;
  left:50%!important;
  top:50%!important;
  width:58px!important;
  height:58px!important;
  margin:0!important;
  transform:translate(-50%,-50%) scale(4.62)!important;
  transform-origin:center!important;
  cursor:default!important;
  pointer-events:none!important;
  animation:brand-seal-float 6.5s ease-in-out infinite!important;
  filter:drop-shadow(0 24px 34px rgba(0,0,0,.30))!important;
}
.seal-viewer-chips{
  position:relative!important;
  z-index:3!important;
  display:flex!important;
  flex-wrap:wrap!important;
  justify-content:center!important;
  gap:8px!important;
  margin:0!important;
}
.seal-viewer-chip{
  display:inline-flex!important;
  align-items:center!important;
  justify-content:center!important;
  min-width:74px!important;
  padding:7px 11px!important;
  border-radius:999px!important;
  color:#9BB7C5!important;
  border:1px solid rgba(143,208,255,.15)!important;
  background:linear-gradient(180deg,rgba(143,208,255,.055),rgba(143,208,255,.025))!important;
  font:800 7px/1 Cascadia Mono,Consolas,monospace!important;
  letter-spacing:.15em!important;
  box-shadow:0 8px 20px rgba(0,0,0,.13),0 0 20px rgba(143,208,255,.025)!important;
  transition:all .2s ease!important;
}
.seal-viewer-chip:hover{
  color:#D9EDF7!important;
  border-color:rgba(143,208,255,.34)!important;
  transform:translateY(-1px)!important;
  box-shadow:0 10px 24px rgba(0,0,0,.20),0 0 24px rgba(143,208,255,.06)!important;
}
.seal-viewer-bottom{
  position:relative!important;
  z-index:3!important;
  display:flex!important;
  flex-direction:column!important;
  align-items:center!important;
  gap:9px!important;
  width:100%!important;
}
.seal-viewer-caption{
  color:#647D8B!important;
  font:700 8px/1.45 Cascadia Mono,Consolas,monospace!important;
  letter-spacing:.04em!important;
  text-align:center!important;
}
.seal-viewer-hint{
  position:relative!important;
  z-index:3!important;
  display:flex!important;
  justify-content:center!important;
  align-items:center!important;
  gap:6px!important;
  flex-wrap:wrap!important;
  margin-top:15px!important;
  padding-top:12px!important;
  width:min(460px,82vw)!important;
  border-top:1px solid rgba(143,208,255,.08)!important;
  color:#596F7C!important;
  font:700 7px/1.4 Cascadia Mono,Consolas,monospace!important;
  text-align:center!important;
  letter-spacing:.06em!important;
}
.seal-viewer-hint-strong{
  color:#8CAEBE!important;
  font-weight:800!important;
  letter-spacing:.12em!important;
}
.seal-viewer-hint-or{
  color:#485D69!important;
  font-weight:700!important;
}
.seal-viewer-hint-tail{
  color:#566E7B!important;
}
.seal-viewer-close{
  position:absolute!important;
  top:17px!important;
  right:17px!important;
  z-index:6!important;
  width:44px!important;
  height:44px!important;
  border-radius:14px!important;
  border:1px solid rgba(143,208,255,.16)!important;
  background:rgba(8,16,24,.72)!important;
  color:#A9BFCD!important;
  font:400 29px/1 "Segoe UI",Arial,sans-serif!important;
  cursor:pointer!important;
  transition:transform .2s ease,border-color .2s ease,color .2s ease,background .2s ease,box-shadow .2s ease!important;
}
.seal-viewer-close:hover{
  transform:rotate(90deg) scale(1.04)!important;
  color:#FFFFFF!important;
  border-color:rgba(143,208,255,.48)!important;
  background:rgba(21,39,53,.94)!important;
  box-shadow:0 12px 32px rgba(0,0,0,.36),0 0 20px rgba(143,208,255,.08)!important;
}
@keyframes seal-viewer-backdrop-in{
  from{opacity:0}
  to{opacity:1}
}
@keyframes seal-viewer-card-in{
  from{opacity:0;transform:translateY(16px) scale(.965)}
  to{opacity:1;transform:none}
}
@keyframes seal-orbit-guide{
  to{transform:translate(-50%,-50%) rotate(360deg)}
}

@media (max-height:720px){
  .seal-viewer-card{
    padding:22px 26px 16px!important;
    border-radius:30px!important;
  }
  .seal-viewer-eyebrow{margin-bottom:6px!important}
  .seal-viewer-subtitle{margin-top:8px!important}
  .seal-viewer-status{margin-top:9px!important;padding:5px 10px!important}
  .seal-viewer-seal-wrap{
    width:230px!important;
    height:230px!important;
    flex-basis:230px!important;
    margin:6px 0 0!important;
  }
  .seal-viewer-seal{
    transform:translate(-50%,-50%) scale(3.55)!important;
  }
  .seal-viewer-seal-wrap:before{width:214px!important;height:214px!important}
  .seal-viewer-seal-wrap:after{width:188px!important;height:188px!important}
  .seal-viewer-chip{padding:6px 9px!important;min-width:68px!important}
  .seal-viewer-caption{font-size:7px!important}
  .seal-viewer-hint{margin-top:10px!important;padding-top:9px!important}
}

@media (max-width:520px){
  .seal-viewer{padding:10px!important}
  .seal-viewer-card{
    width:min(96vw,620px)!important;
    padding:22px 14px 15px!important;
  }
  .seal-viewer-heading{padding:0 34px!important}
  .seal-viewer-title-main{font-size:25px!important}
  .seal-viewer-title-accent{font-size:10px!important}
  .seal-viewer-subtitle{
    max-width:330px!important;
    font-size:7px!important;
    letter-spacing:.07em!important;
  }
  .seal-viewer-seal-wrap{
    width:230px!important;
    height:230px!important;
    flex-basis:230px!important;
  }
  .seal-viewer-seal{
    transform:translate(-50%,-50%) scale(3.55)!important;
  }
  .seal-viewer-seal-wrap:before{width:214px!important;height:214px!important}
  .seal-viewer-seal-wrap:after{width:188px!important;height:188px!important}
  .seal-viewer-hint{
    width:88vw!important;
  }
}
@media (prefers-reduced-motion:reduce){
  .seal-viewer-status-dot,
  .seal-viewer-seal,
  .seal-viewer-seal-wrap:after{
    animation:none!important;
  }
}

  position:fixed!important;
  inset:0!important;
  z-index:120000!important;
  display:flex!important;
  align-items:center!important;
  justify-content:center!important;
  padding:24px!important;
  isolation:isolate!important;
}
.seal-viewer.is-hidden{display:none!important}
.seal-viewer-backdrop{
  position:absolute!important;
  inset:0!important;
  cursor:pointer!important;
  background:
    radial-gradient(circle at 50% 43%,rgba(143,208,255,.13),transparent 22%),
    radial-gradient(circle at 50% 56%,rgba(255,153,51,.08),transparent 31%),
    rgba(2,6,10,.68)!important;
  backdrop-filter:blur(18px) saturate(125%)!important;
  -webkit-backdrop-filter:blur(18px) saturate(125%)!important;
  animation:seal-viewer-backdrop-in .28s ease both!important;
}
.seal-viewer-card{
  position:relative!important;
  z-index:2!important;
  width:min(560px,92vw)!important;
  min-height:min(680px,86vh)!important;
  display:flex!important;
  flex-direction:column!important;
  align-items:center!important;
  justify-content:center!important;
  gap:18px!important;
  padding:42px 34px 34px!important;
  border-radius:34px!important;
  border:1px solid rgba(143,208,255,.22)!important;
  background:
    radial-gradient(circle at 50% 0%,rgba(143,208,255,.11),transparent 34%),
    radial-gradient(circle at 15% 90%,rgba(255,153,51,.055),transparent 28%),
    linear-gradient(155deg,rgba(16,27,38,.93),rgba(6,12,18,.96))!important;
  box-shadow:
    0 50px 140px rgba(0,0,0,.62),
    0 0 0 1px rgba(255,255,255,.025) inset,
    0 0 90px rgba(126,200,255,.10)!important;
  overflow:hidden!important;
  animation:seal-viewer-card-in .42s cubic-bezier(.16,1,.3,1) both!important;
}
.seal-viewer-card:before{
  content:"";
  position:absolute!important;
  inset:0!important;
  pointer-events:none!important;
  background:
    linear-gradient(rgba(143,208,255,.026) 1px,transparent 1px),
    linear-gradient(90deg,rgba(143,208,255,.026) 1px,transparent 1px);
  background-size:30px 30px!important;
  mask-image:linear-gradient(180deg,#000 0%,transparent 88%)!important;
  -webkit-mask-image:linear-gradient(180deg,#000 0%,transparent 88%)!important;
}
.seal-viewer-heading{
  position:relative!important;
  z-index:2!important;
  text-align:center!important;
}
.seal-viewer-kicker{
  color:#F2F7FB!important;
  font:800 clamp(22px,4vw,31px)/1 Bahnschrift,"Segoe UI",sans-serif!important;
  letter-spacing:-.035em!important;
  text-shadow:0 0 28px rgba(143,208,255,.15)!important;
}
.seal-viewer-label{
  margin-top:8px!important;
  color:#8FD0FF!important;
  font:700 9px/1.2 Cascadia Mono,Consolas,monospace!important;
  letter-spacing:.24em!important;
}
.seal-viewer-seal-wrap{
  position:relative!important;
  z-index:2!important;
  width:58px!important;
  height:58px!important;
  flex:0 0 58px!important;
  margin:42px 0 36px!important;
  transform:scale(4.65)!important;
  transform-origin:center!important;
  filter:drop-shadow(0 22px 30px rgba(0,0,0,.28))!important;
}
.seal-viewer-seal{
  cursor:default!important;
  animation:brand-seal-float 6.5s ease-in-out infinite!important;
}
.seal-viewer-seal:hover{
  transform:translateY(-3px) scale(1.05)!important;
}
.seal-viewer-chips{
  position:relative!important;
  z-index:2!important;
  display:flex!important;
  flex-wrap:wrap!important;
  justify-content:center!important;
  gap:8px!important;
  margin-top:12px!important;
}
.seal-viewer-chip{
  display:inline-flex!important;
  align-items:center!important;
  justify-content:center!important;
  min-width:72px!important;
  padding:7px 10px!important;
  border-radius:999px!important;
  color:#94BCD1!important;
  border:1px solid rgba(143,208,255,.13)!important;
  background:rgba(143,208,255,.04)!important;
  font:700 8px/1 Cascadia Mono,Consolas,monospace!important;
  letter-spacing:.13em!important;
}
.seal-viewer-hint{
  position:relative!important;
  z-index:2!important;
  color:#617A89!important;
  font:9px/1.5 Cascadia Mono,Consolas,monospace!important;
  text-align:center!important;
  letter-spacing:.03em!important;
}
.seal-viewer-close{
  position:absolute!important;
  top:18px!important;
  right:18px!important;
  z-index:4!important;
  width:44px!important;
  height:44px!important;
  border-radius:14px!important;
  border:1px solid rgba(143,208,255,.16)!important;
  background:rgba(10,18,27,.78)!important;
  color:#A9BFCD!important;
  font:400 29px/1 "Segoe UI",Arial,sans-serif!important;
  cursor:pointer!important;
  transition:all .18s ease!important;
}
.seal-viewer-close:hover{
  transform:rotate(90deg)!important;
  color:#FFFFFF!important;
  border-color:rgba(143,208,255,.46)!important;
  background:rgba(26,45,61,.94)!important;
  box-shadow:0 10px 28px rgba(0,0,0,.34)!important;
}
@keyframes seal-viewer-backdrop-in{
  from{opacity:0}
  to{opacity:1}
}
@keyframes seal-viewer-card-in{
  from{opacity:0;transform:translateY(14px) scale(.965);filter:blur(4px)}
  to{opacity:1;transform:none;filter:none}
}

/* ---------- modal shell ---------- */
.team-modal{
  position:fixed!important;
  inset:0!important;
  z-index:100000!important;
  display:flex!important;
  align-items:center!important;
  justify-content:center!important;
  padding:28px!important;
}
.team-modal.is-hidden{display:none!important}
.team-modal-backdrop{
  position:absolute!important;
  inset:0!important;
  background:
    radial-gradient(circle at 50% 35%,rgba(143,208,255,.10),transparent 26%),
    rgba(2,5,8,.78)!important;
  backdrop-filter:blur(12px) saturate(115%)!important;
  -webkit-backdrop-filter:blur(12px) saturate(115%)!important;
}
.team-modal-dialog{
  position:relative!important;
  z-index:2!important;
  width:min(1120px,96vw)!important;
  max-height:min(850px,92vh)!important;
  overflow:hidden!important;
  border:1px solid rgba(143,208,255,.24)!important;
  border-radius:28px!important;
  background:
    radial-gradient(circle at 15% 5%,rgba(143,208,255,.08),transparent 28%),
    radial-gradient(circle at 90% 15%,rgba(199,146,234,.08),transparent 27%),
    linear-gradient(150deg,#0F1822 0%,#080F16 56%,#070B10 100%)!important;
  box-shadow:
    0 45px 120px rgba(0,0,0,.58),
    0 0 0 1px rgba(255,255,255,.025) inset,
    0 0 70px rgba(126,200,255,.08)!important;
  animation:team-modal-in .44s cubic-bezier(.16,1,.3,1) both!important;
}
.team-modal-dialog:before{
  content:"";
  position:absolute!important;
  inset:0!important;
  pointer-events:none!important;
  background:
    linear-gradient(rgba(143,208,255,.025) 1px,transparent 1px),
    linear-gradient(90deg,rgba(143,208,255,.025) 1px,transparent 1px);
  background-size:28px 28px!important;
  mask-image:linear-gradient(180deg,#000,transparent 95%)!important;
  -webkit-mask-image:linear-gradient(180deg,#000,transparent 95%)!important;
}
.team-modal-top{
  position:relative!important;
  display:flex!important;
  align-items:flex-start!important;
  justify-content:space-between!important;
  gap:18px!important;
  padding:25px 27px 19px!important;
  border-bottom:1px solid rgba(143,208,255,.11)!important;
  background:linear-gradient(180deg,rgba(255,255,255,.015),transparent)!important;
}
.team-modal-heading{
  display:flex!important;
  align-items:center!important;
  gap:16px!important;
  min-width:0!important;
}
.model-logo-modal{width:60px!important;height:60px!important;flex-basis:60px!important;border-radius:18px!important}
.team-modal-kicker{
  color:#8FD0FF!important;
  font:700 9px/1.2 Cascadia Mono,Consolas,monospace!important;
  letter-spacing:.22em!important;
  margin-bottom:7px!important;
}
.team-modal-title{
  margin:0!important;
  color:#F1F6FA!important;
  font:800 clamp(24px,3vw,34px)/1.05 Bahnschrift,"Segoe UI",sans-serif!important;
  letter-spacing:-.035em!important;
}
.team-modal-subtitle{
  color:#8298A8!important;
  font:11px/1.45 "Segoe UI",Arial,sans-serif!important;
  margin-top:7px!important;
}
.team-modal-close{
  width:42px!important;
  height:42px!important;
  flex:0 0 42px!important;
  border-radius:13px!important;
  border:1px solid rgba(143,208,255,.16)!important;
  background:rgba(14,24,34,.86)!important;
  color:#A9BECC!important;
  font:400 27px/1 "Segoe UI",Arial,sans-serif!important;
  cursor:pointer!important;
  transition:all .18s ease!important;
}
.team-modal-close:hover{
  transform:rotate(90deg)!important;
  color:#FFFFFF!important;
  border-color:rgba(143,208,255,.45)!important;
  background:rgba(27,46,63,.95)!important;
  box-shadow:0 10px 28px rgba(0,0,0,.30)!important;
}
.team-modal-scroll{
  position:relative!important;
  overflow-y:auto!important;
  max-height:calc(min(850px,92vh) - 106px)!important;
  padding:20px 22px 24px!important;
}
.team-modal-pills{
  display:flex!important;
  flex-wrap:wrap!important;
  gap:7px!important;
  margin:0 2px 14px!important;
}
.team-modal-pill{
  display:inline-flex!important;
  align-items:center!important;
  padding:6px 9px!important;
  border-radius:999px!important;
  color:#8FBED7!important;
  border:1px solid rgba(143,208,255,.13)!important;
  background:rgba(143,208,255,.035)!important;
  font:700 7px/1 Cascadia Mono,Consolas,monospace!important;
  letter-spacing:.15em!important;
}
.team-modal-pill-alt{
  color:#C5A8DE!important;
  border-color:rgba(199,146,234,.17)!important;
  background:rgba(199,146,234,.035)!important;
}

/* ---------- six member cards ---------- */
.team-members-grid{
  display:grid!important;
  grid-template-columns:repeat(3,minmax(0,1fr))!important;
  gap:12px!important;
}
.team-member-card{
  position:relative!important;
  min-height:215px!important;
  padding:17px!important;
  border-radius:19px!important;
  border:1px solid rgba(142,172,196,.13)!important;
  background:
    linear-gradient(145deg,rgba(18,29,40,.94),rgba(9,15,22,.98))!important;
  box-shadow:
    0 13px 30px rgba(0,0,0,.20),
    0 1px 0 rgba(255,255,255,.04) inset!important;
  overflow:hidden!important;
  transition:transform .20s ease,border-color .20s ease,box-shadow .20s ease!important;
}
.team-member-card:before{
  content:"";
  position:absolute!important;
  top:0!important;
  left:15px!important;
  right:15px!important;
  height:1px!important;
  background:linear-gradient(90deg,transparent,rgba(143,208,255,.52),transparent)!important;
}
.team-member-card:after{
  content:"";
  position:absolute!important;
  width:120px!important;
  height:120px!important;
  right:-70px!important;
  bottom:-70px!important;
  border-radius:50%!important;
  border:1px solid rgba(143,208,255,.05)!important;
  box-shadow:0 0 0 14px rgba(143,208,255,.012)!important;
  pointer-events:none!important;
}
.team-member-card:hover{
  transform:translateY(-5px)!important;
  border-color:rgba(143,208,255,.32)!important;
  box-shadow:
    0 20px 42px rgba(0,0,0,.30),
    0 0 30px rgba(126,200,255,.06)!important;
}
.team-member-avatar-wrap{
  position:relative!important;
  width:70px!important;
  height:70px!important;
  margin-bottom:13px!important;
}
.team-member-avatar{
  width:70px!important;
  height:70px!important;
  border-radius:21px!important;
  display:grid!important;
  place-items:center!important;
  overflow:hidden!important;
  border:1px solid rgba(143,208,255,.25)!important;
  box-shadow:0 9px 24px rgba(0,0,0,.28)!important;
}
.team-member-avatar-initials{
  color:#F0F6FA!important;
  font:800 20px/1 Bahnschrift,"Segoe UI",sans-serif!important;
  letter-spacing:-.03em!important;
  background:
    radial-gradient(circle at 30% 25%,rgba(143,208,255,.25),transparent 38%),
    linear-gradient(145deg,#1C354B,#111A25)!important;
}
.team-member-photo{
  width:100%!important;
  height:100%!important;
  object-fit:cover!important;
  display:block!important;
}
.team-member-jmi-badge-logo{
  padding:2px!important;
  width:30px!important;
  min-width:30px!important;
  height:30px!important;
}
.team-member-jmi-logo{
  width:100%!important;
  height:100%!important;
  object-fit:contain!important;
  border-radius:50%!important;
  background:#fff!important;
  display:block!important;
}

.team-member-jmi-badge{
  position:absolute!important;
  right:-6px!important;
  bottom:-6px!important;
  min-width:31px!important;
  height:22px!important;
  padding:0 7px!important;
  display:flex!important;
  align-items:center!important;
  justify-content:center!important;
  border-radius:999px!important;
  color:#09121B!important;
  background:linear-gradient(90deg,#8FD0FF,#C792EA)!important;
  border:2px solid #0A1118!important;
  box-shadow:0 5px 15px rgba(0,0,0,.30)!important;
  font:800 7px/1 Cascadia Mono,Consolas,monospace!important;
  letter-spacing:.08em!important;
}
.team-member-name{
  color:#F1F6FA!important;
  font:800 18px/1.1 Bahnschrift,"Segoe UI",sans-serif!important;
  letter-spacing:-.025em!important;
  margin-bottom:4px!important;
}
.team-member-role{
  color:#9BD2F0!important;
  font:700 9px/1.2 Cascadia Mono,Consolas,monospace!important;
  letter-spacing:.12em!important;
  text-transform:uppercase!important;
}
.team-member-degree{
  color:#7C94A5!important;
  font:600 10px/1.3 "Segoe UI",Arial,sans-serif!important;
  margin-top:6px!important;
}
.team-member-bio{
  color:#758A99!important;
  font:10px/1.45 "Segoe UI",Arial,sans-serif!important;
  margin-top:9px!important;
  max-width:310px!important;
}
.team-modal-edit-box{
  display:flex!important;
  flex-wrap:wrap!important;
  align-items:center!important;
  gap:8px!important;
  margin:14px 2px 0!important;
  padding:10px 12px!important;
  border-radius:13px!important;
  border:1px dashed rgba(143,208,255,.13)!important;
  background:rgba(143,208,255,.025)!important;
}
.team-modal-edit-hint{
  color:#657D8D!important;
  font:9px/1.4 Cascadia Mono,Consolas,monospace!important;
}
.team-modal-edit-hint.strong{color:#86A1B2!important}

@keyframes team-modal-in{
  from{opacity:0;transform:translateY(18px) scale(.985);filter:blur(3px)}
  to{opacity:1;transform:none;filter:none}
}

@media(max-width:900px){
  .team-footer-inner{grid-template-columns:1fr!important;gap:12px!important}
  .team-brand-marks{flex-direction:row!important;justify-content:center!important}
  .team-footer-copy{text-align:center!important}
  .team-footer-copy .team-credit-row,
  .team-footer-copy .team-id-row,
  .team-footer-copy .team-links-row{justify-content:center!important}
  .team-footer-copy .team-tagline,
  .team-footer-copy .team-members-note{margin-left:auto!important;margin-right:auto!important}
  .team-members-grid{grid-template-columns:repeat(2,minmax(0,1fr))!important}
}
@media(max-width:620px){
  .seal-viewer{padding:12px!important}
  .seal-viewer-card{width:min(94vw,480px)!important;min-height:min(610px,88vh)!important;padding:36px 18px 26px!important;border-radius:27px!important}
  .seal-viewer-seal-wrap{transform:scale(3.65)!important;margin:33px 0 26px!important}
  .seal-viewer-close{top:12px!important;right:12px!important;width:40px!important;height:40px!important}
  .seal-viewer-chip{min-width:64px!important;font-size:7px!important}
  .team-modal{padding:10px!important}
  .team-modal-dialog{border-radius:22px!important}
  .team-modal-top{padding:18px 17px 15px!important}
  .team-modal-scroll{padding:15px 13px 18px!important}
  .team-modal-heading{gap:11px!important}
  .model-logo-modal{width:50px!important;height:50px!important;flex-basis:50px!important}
  .team-members-grid{grid-template-columns:1fr!important}
  .team-member-card{min-height:190px!important}
  .team-link-btn,.team-repo-btn,.team-info-btn{min-width:150px!important}
}
@media(prefers-reduced-motion:reduce){
  .seal-viewer-backdrop,.seal-viewer-card,.seal-viewer-seal,.model-logo-orbit,.team-modal-dialog,.team-heart{animation:none!important}
  .team-member-card,.team-modal-close,.team-link-btn{transition:none!important}
}


/* ============================================================
   FINAL FOOTER / LOADING FIX — VISUAL + PRESENTATION LAYER ONLY
   ============================================================ */

/* Footer stays at the actual end of the page; never auto-pushed above content. */
.main-content{
  display:flex!important;
  flex-direction:column!important;
  min-height:100vh!important;
}
.team-footer{
  margin-top:64px!important;
  margin-bottom:8px!important;
  flex:0 0 auto!important;
  visibility:visible!important;
}


/* Prevent the team section from appearing before the dashboard has populated. */
.team-footer-pending{
  visibility:hidden!important;
  opacity:0!important;
  pointer-events:none!important;
  transform:translateY(18px)!important;
}
.team-footer-ready{
  visibility:visible!important;
  opacity:1!important;
  pointer-events:auto!important;
  transform:none!important;
  animation:footer-ready-in .72s cubic-bezier(.16,1,.3,1) both!important;
}
/* ============================================================
   LEGAL · LICENCE · SECURITY CHROME
   ============================================================ */
 /* ============================================================
   TRUST BAR — minimal · classic · elegant
   ============================================================ */
.trust-block{
  margin-top:30px;
  padding-top:24px;
  width:100%;
  max-width:860px;
  margin-left:auto;
  margin-right:auto;
  display:flex;
  flex-direction:column;
  align-items:center;
  gap:16px;
}

/* Divider with centred label */
.trust-divider{
  display:flex;
  align-items:center;
  gap:16px;
  width:100%;
  max-width:620px;
}
.trust-line{
  flex:1;
  height:1px;
  background:linear-gradient(90deg,transparent,rgba(143,208,255,.22),transparent);
}
.trust-line-label{
  font:700 8px/1 Cascadia Mono,Consolas,monospace;
  letter-spacing:.30em;
  color:#6E8496;
  text-transform:uppercase;
  white-space:nowrap;
}

/* Licence pills — separated by hairlines, not boxed */
.trust-pill-row{
  display:flex;
  flex-wrap:wrap;
  justify-content:center;
  align-items:center;
  gap:0;
  width:100%;
}
.trust-pill{
  font:700 9px/1 Cascadia Mono,Consolas,monospace;
  letter-spacing:.14em;
  color:#9DB4C4;
  padding:2px 16px;
  border-right:1px solid rgba(143,208,255,.12);
  text-transform:uppercase;
  white-space:nowrap;
}
.trust-pill:last-child{ border-right:none; }
.trust-pill-quiet{ color:#5E7484; }

/* Security banner */
.trust-security-wrap{
  display:flex;
  flex-wrap:wrap;
  align-items:center;
  justify-content:center;
  gap:18px;
  padding:12px 22px;
  border-radius:14px;
  border:1px solid rgba(127,225,165,.16);
  background:linear-gradient(135deg,rgba(127,225,165,.045),rgba(15,22,30,.55));
  box-shadow:inset 0 1px 0 rgba(255,255,255,.03);
  max-width:820px;
}
.trust-shield-wrap{
  display:flex;
  align-items:center;
  gap:9px;
  padding-right:16px;
  border-right:1px solid rgba(127,225,165,.14);
}
.trust-shield{
  font-size:15px;
  line-height:1;
  filter:drop-shadow(0 0 8px rgba(127,225,165,.5));
}
.trust-shield-label{
  font:700 8.5px/1 Cascadia Mono,Consolas,monospace;
  letter-spacing:.20em;
  color:#9FE1B8;
}
.trust-pillar-row{
  display:flex;
  flex-wrap:wrap;
  gap:16px;
  align-items:center;
}
.trust-pillar{
  position:relative;
  padding-left:11px;
  font:700 8.5px/1 Cascadia Mono,Consolas,monospace;
  letter-spacing:.13em;
  color:#A8C8BC;
  text-transform:uppercase;
}
.trust-pillar:before{
  content:"";
  position:absolute;
  left:0;
  top:50%;
  transform:translateY(-50%);
  width:3px;
  height:3px;
  border-radius:50%;
  background:#7FE1A5;
  box-shadow:0 0 6px rgba(127,225,165,.6);
}

/* Advisory notes */
.trust-notes{
  display:flex;
  flex-direction:column;
  gap:6px;
  align-items:flex-start;
  max-width:700px;
  width:100%;
  padding:0 8px;
}
.trust-note-row{
  display:flex;
  align-items:flex-start;
  gap:10px;
}
.trust-note-dot{
  color:#5E7484;
  font-size:7px;
  line-height:1.8;
  flex-shrink:0;
}
.trust-note-text{
  font:11.5px/1.55 "Segoe UI",Arial,sans-serif;
  color:#8DA3B3;
}

/* Legal footer */
.trust-legal{
  width:100%;
  max-width:620px;
  padding-top:12px;
  border-top:1px dashed rgba(143,208,255,.08);
  font:9px/1.5 Cascadia Mono,Consolas,monospace;
  letter-spacing:.06em;
  color:#5E7484;
  text-align:center;
}

@media(max-width:640px){
  .trust-pill{ padding:2px 10px; font-size:8px; }
  .trust-security-wrap{ gap:12px; padding:10px 14px; }
  .trust-shield-wrap{ padding-right:12px; }
  .trust-pillar-row{ gap:12px; }
  .trust-pillar{ font-size:8px; padding-left:9px; }
  .trust-note-text{ font-size:11px; }
}

@media(max-width:640px){
  .legal-badge-row{ gap:5px; }
  .legal-badge{ font-size:7.5px; padding:4px 9px; }
  .security-banner{ padding:9px 12px; }
  .security-text{ font-size:7.5px; }
  .security-note{ font-size:10.5px; }
}
/* Calm the star field: fewer, finer stars = cleaner + more premium. */
.team-footer:before{
  opacity:.48!important;
  background-image:
    radial-gradient(circle at 8% 18%,rgba(255,255,255,.48) 0 1px,transparent 1.25px),
    radial-gradient(circle at 17% 74%,rgba(143,208,255,.35) 0 .9px,transparent 1.2px),
    radial-gradient(circle at 28% 31%,rgba(255,255,255,.24) 0 .8px,transparent 1.1px),
    radial-gradient(circle at 41% 80%,rgba(199,146,234,.30) 0 .9px,transparent 1.2px),
    radial-gradient(circle at 52% 24%,rgba(255,255,255,.34) 0 .9px,transparent 1.15px),
    radial-gradient(circle at 63% 72%,rgba(143,208,255,.28) 0 .8px,transparent 1.1px),
    radial-gradient(circle at 74% 16%,rgba(255,255,255,.28) 0 .8px,transparent 1.1px),
    radial-gradient(circle at 85% 56%,rgba(143,208,255,.32) 0 .9px,transparent 1.2px),
    radial-gradient(circle at 93% 27%,rgba(199,146,234,.26) 0 .8px,transparent 1.1px),
    radial-gradient(circle at 91% 82%,rgba(255,255,255,.20) 0 .8px,transparent 1.1px)!important;
}

/* Keep the footer structure elegant and centered. */
.team-footer-inner{
  width:min(860px,100%)!important;
  margin:0 auto!important;
  display:flex!important;
  flex-direction:column!important;
  align-items:center!important;
  justify-content:center!important;
  text-align:center!important;
}
.team-footer-copy{
  width:min(760px,100%)!important;
  text-align:center!important;
}
.team-footer-copy .footer-note-line,
.team-footer-copy .team-eyebrow,
.team-footer-copy .team-credit-row,
.team-footer-copy .team-id-row,
.team-footer-copy .team-links-row{
  justify-content:center!important;
}
.team-footer-copy .team-tagline,
.team-footer-copy .team-members-note{
  margin-left:auto!important;
  margin-right:auto!important;
  text-align:center!important;
}

/* Compact, premium signature mark. */
.team-signature-mark{
  width:100%!important;
  display:flex!important;
  justify-content:center!important;
  align-items:center!important;
  margin:0 0 13px!important;
}
.team-signature-logo{
  width:54px!important;
  height:54px!important;
  flex-basis:54px!important;
  border-radius:17px!important;
  box-shadow:
    0 12px 34px rgba(0,0,0,.36),
    0 0 30px rgba(143,208,255,.09),
    inset 0 1px 0 rgba(255,255,255,.08)!important;
}

/* The previous gradient needed explicit text clipping; otherwise it renders
   as a large rectangular gradient behind the heading. */
.team-name-xl{
  display:inline-block!important;
  margin:5px 0 13px!important;
  font:800 clamp(30px,4vw,44px)/1 Bahnschrift,"Segoe UI",sans-serif!important;
  letter-spacing:-.045em!important;
  background:linear-gradient(105deg,#F8FBFE 0%,#A9D9F7 30%,#CA9AE8 62%,#F0C87B 95%)!important;
  -webkit-background-clip:text!important;
  background-clip:text!important;
  -webkit-text-fill-color:transparent!important;
  color:transparent!important;
  filter:drop-shadow(0 8px 34px rgba(142,180,230,.12))!important;
}

.footer-note-line{
  margin-bottom:11px!important;
  color:#8094A2!important;
  font:10px/1.4 Cascadia Mono,Consolas,monospace!important;
  letter-spacing:.035em!important;
}
.team-eyebrow{
  color:#9BC9E4!important;
  margin-bottom:12px!important;
  font:700 9px/1 Cascadia Mono,Consolas,monospace!important;
  letter-spacing:.24em!important;
}
.team-credit-row{margin-bottom:6px!important}
.team-credit-text{color:#BBC8D1!important}
.team-heart{
  color:#FF6F91!important;
  filter:drop-shadow(0 0 11px rgba(255,111,145,.45))!important;
}
.team-id-row{margin-bottom:17px!important}
.team-links-row{
  gap:10px!important;
  margin:2px 0 17px!important;
}
.team-link-btn{
  min-width:205px!important;
  min-height:44px!important;
  border-radius:999px!important;
}
.team-tagline{
  color:#9BAEBB!important;
  font-size:11.5px!important;
}
.team-members-note{
  color:#647A89!important;
  font-size:8.5px!important;
}

@keyframes footer-ready-in{
  from{opacity:0;transform:translateY(16px) scale(.992);filter:blur(2px)}
  to{opacity:1;transform:none;filter:none}
}

@media(max-width:700px){
  .team-footer{margin-top:48px!important}
  .team-links-row{width:100%!important}
  .team-link-btn{min-width:150px!important;width:100%!important}
}




/* ============================================================
   FASTAPI STATUS — CLEAR, COMPACT, AND HONEST
   ============================================================ */
.connection-state{
  margin:14px 0 0!important;
  padding:9px 10px!important;
  min-height:42px!important;
  display:flex!important;
  align-items:center!important;
  border:1px solid rgba(122,160,188,.16)!important;
  border-radius:12px!important;
  background:linear-gradient(145deg,rgba(14,24,33,.96),rgba(8,14,20,.98))!important;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.035),0 8px 22px rgba(0,0,0,.16)!important;
}
.connection-state-content{
  width:100%!important;
  display:flex!important;
  align-items:center!important;
  gap:8px!important;
}
.connection-pulse{
  width:7px!important;
  height:7px!important;
  flex:0 0 7px!important;
  border-radius:50%!important;
  background:#8A9AA6!important;
  box-shadow:0 0 0 3px rgba(138,154,166,.08)!important;
}
.connection-pulse.is-online{
  background:#62D39B!important;
  box-shadow:0 0 0 3px rgba(98,211,155,.10),0 0 12px rgba(98,211,155,.35)!important;
}
.connection-pulse.is-offline{
  background:#F06A72!important;
  box-shadow:0 0 0 3px rgba(240,106,114,.10),0 0 12px rgba(240,106,114,.28)!important;
}
.connection-pulse.is-degraded{
  background:#F0C675!important;
  box-shadow:0 0 0 3px rgba(240,198,117,.10),0 0 12px rgba(240,198,117,.24)!important;
}
.connection-pulse.is-checking{
  animation:connection-pulse 1.4s ease-in-out infinite!important;
}
.connection-state-label{
  font:800 8px/1 Cascadia Mono,Consolas,monospace!important;
  letter-spacing:.16em!important;
  color:#8EA4B3!important;
}
.connection-state-value{
  margin-left:auto!important;
  font:800 8px/1 Cascadia Mono,Consolas,monospace!important;
  letter-spacing:.12em!important;
  color:#A9BBC7!important;
}
.connection-online .connection-state-value{color:#83E3AF!important}
.connection-degraded .connection-state-value{color:#F0C675!important}
.connection-offline .connection-state-value{color:#FF8990!important}
@keyframes connection-pulse{0%,100%{opacity:.35;transform:scale(.85)}50%{opacity:1;transform:scale(1.15)}}
@media(prefers-reduced-motion:reduce){.connection-pulse.is-checking{animation:none!important;opacity:.8!important}}

/* ============================================================
   KPI READABILITY + PREMIUM INTERACTION PASS
   Visual-only override: preserves all values, callbacks & data.
   ============================================================ */

/* Clean card surface: subtle depth, no oversized background disks. */
.kpi-card{
  position:relative!important;
  isolation:isolate!important;
  overflow:hidden!important;
  border-color:rgba(137,165,187,.20)!important;
  background:
    radial-gradient(120px 90px at 100% 0%,rgba(143,208,255,.055),transparent 72%),
    linear-gradient(145deg,rgba(18,27,37,.985) 0%,rgba(10,16,23,.995) 100%)!important;
  box-shadow:
    0 12px 30px rgba(0,0,0,.24),
    inset 0 1px 0 rgba(255,255,255,.045),
    inset 0 -1px 0 rgba(0,0,0,.35)!important;
  transition:
    transform .22s cubic-bezier(.2,.8,.2,1),
    border-color .22s ease,
    box-shadow .22s ease,
    background .22s ease!important;
}

/* Fine technical texture stays in the background and never crosses text. */
.kpi-card:after{
  content:"";
  position:absolute;
  right:-54px;
  top:-54px;
  width:150px;
  height:150px;
  border-radius:50%;
  background:
    radial-gradient(circle at 42% 58%,rgba(143,208,255,.16) 0,rgba(143,208,255,.055) 20%,transparent 48%),
    repeating-radial-gradient(circle at 42% 58%,rgba(143,208,255,.045) 0 1px,transparent 1px 13px);
  opacity:.42;
  filter:blur(.1px);
  pointer-events:none;
  z-index:0!important;
}

/* Moving corner light: elegant interaction rather than a heavy glow. */
.kpi-card:before{
  content:"";
  position:absolute;
  top:0;
  left:-24%;
  width:42%;
  height:1px;
  background:linear-gradient(90deg,transparent,rgba(182,223,255,.98),transparent);
  box-shadow:0 0 12px rgba(143,208,255,.24);
  opacity:.72;
  transform:translateX(0);
  animation:kpi-sheen 5.8s ease-in-out infinite;
  z-index:3!important;
  pointer-events:none;
}

@keyframes kpi-sheen{
  0%,58%{transform:translateX(0);opacity:.08}
  68%{opacity:.78}
  82%{transform:translateX(305%);opacity:.42}
  100%{transform:translateX(305%);opacity:.06}
}

/* Keep all actual content above every decorative layer. */
.kpi-card > *{
  position:relative!important;
  z-index:4!important;
}

/* Labels: brighter and crisper for instant scanning. */
.kpi-label{
  font-size:9px!important;
  line-height:1.2!important;
  letter-spacing:.16em!important;
  font-weight:750!important;
  color:#AFC2CF!important;
  margin-bottom:0!important;
  text-shadow:0 1px 8px rgba(0,0,0,.50)!important;
  opacity:1!important;
}

/* Add a tiny signal marker without modifying the underlying text. */
.kpi-label:before{
  content:"";
  display:inline-block;
  width:4px;
  height:4px;
  margin:0 7px 1px 0;
  border-radius:50%;
  background:#8FD0FF;
  box-shadow:0 0 8px rgba(143,208,255,.48);
  vertical-align:middle;
}

.kpi-card.warning .kpi-label:before{
  background:#F0C675;
  box-shadow:0 0 8px rgba(240,198,117,.45);
}

.kpi-card.danger .kpi-label:before{
  background:#E85A5A;
  box-shadow:0 0 8px rgba(232,90,90,.45);
}

/* Numerals intentionally remain the strongest visual hierarchy. */
.kpi-value{
  position:relative!important;
  z-index:4!important;
  text-shadow:0 1px 14px rgba(0,0,0,.34)!important;
}

/* Supporting copy: distinctly brighter, but still secondary to the number. */
.kpi-hint{
  font-size:10px!important;
  line-height:1.35!important;
  font-weight:500!important;
  color:#91A8B8!important;
  margin-top:9px!important;
  text-shadow:0 1px 7px rgba(0,0,0,.42)!important;
  opacity:1!important;
}

/* Hover state gives a tactile, premium control-center feel. */
.kpi-card:hover{
  transform:translateY(-4px)!important;
  border-color:rgba(143,208,255,.34)!important;
  background:
    radial-gradient(150px 110px at 100% 0%,rgba(143,208,255,.08),transparent 72%),
    linear-gradient(145deg,rgba(21,31,42,.99) 0%,rgba(9,15,21,.995) 100%)!important;
  box-shadow:
    0 18px 40px rgba(0,0,0,.31),
    inset 0 1px 0 rgba(255,255,255,.055),
    inset 0 0 0 1px rgba(143,208,255,.025)!important;
}

.kpi-card:hover .kpi-label{
  color:#C5D4DE!important;
}

.kpi-card:hover .kpi-hint{
  color:#A4B7C4!important;
}

.kpi-card.warning:hover{
  border-color:rgba(240,198,117,.46)!important;
}

.kpi-card.warning .kpi-label,
.kpi-card.warning .kpi-hint{
  color:#C7B99F!important;
}

.kpi-card.danger:hover{
  border-color:rgba(232,90,90,.46)!important;
}

.kpi-card.danger .kpi-label,
.kpi-card.danger .kpi-hint{
  color:#CDA6AC!important;
}

/* Secondary KPI cards: same system, quieter scale. */
.secondary-kpi-grid .kpi-card{
  min-height:88px!important;
  background:
    radial-gradient(100px 75px at 100% 0%,rgba(143,208,255,.045),transparent 72%),
    linear-gradient(145deg,rgba(14,22,30,.99),rgba(8,14,20,.995))!important;
}

.secondary-kpi-grid .kpi-label{
  font-size:8.5px!important;
  line-height:1.2!important;
  color:#A7BBC8!important;
}

.secondary-kpi-grid .kpi-label:before{
  width:3px;
  height:3px;
  margin-right:6px;
}

.secondary-kpi-grid .kpi-hint{
  font-size:9.5px!important;
  color:#8EA4B4!important;
  margin-top:6px!important;
}

/* Reduced motion: retain appearance and interaction without animation. */
@media(prefers-reduced-motion:reduce){
  .kpi-card:before{animation:none!important;opacity:.28!important}
  .kpi-card{transition:none!important}
  .kpi-card:hover{transform:none!important}
}



/* ============================================================
   THE LIVING SEAL — pure-CSS MPLADS sidebar brand
   Inspired by the official MPLADS emblem:
     • Parliament dome
     • three-figure triad (saffron / blue / green)
     • chakra spoke ring
     • tri-colour flag palette
   No images. No external assets.
   ============================================================ */

/* ---------- Brand lockup ---------- */
.brand-top{
  display:flex!important;
  align-items:flex-start!important;
  justify-content:space-between!important;
  gap:10px!important;
}
.brand-lockup{
  display:flex!important;
  align-items:center!important;
  gap:14px!important;
  min-width:0!important;
  flex:1!important;
}

/* ---------- Seal container ---------- */
.brand-seal{
  position:relative!important;
  width:58px!important;
  height:58px!important;
  flex:0 0 58px!important;
  border-radius:18px!important;
  display:grid!important;
  place-items:center!important;
  overflow:visible!important;
  cursor:pointer!important;
  isolation:isolate!important;
  background:
    radial-gradient(circle at 30% 20%, rgba(255,255,255,.09), transparent 42%),
    linear-gradient(145deg,#182A3C 0%,#0B131C 62%,#070C12 100%)!important;
  border:1px solid rgba(143,208,255,.22)!important;
  box-shadow:
    0 10px 26px rgba(0,0,0,.34),
    0 0 0 1px rgba(255,255,255,.025) inset,
    0 1px 0 rgba(255,255,255,.06) inset,
    0 0 26px rgba(126,200,255,.08)!important;
  transition:
    transform .28s cubic-bezier(.34,1.56,.64,1),
    box-shadow .28s ease,
    border-color .28s ease!important;
  animation:brand-seal-float 6.5s ease-in-out infinite!important;
  -webkit-tap-highlight-color:transparent!important;
}

/* ---------- 1. Rotating tri-colour aura ---------- */
.brand-seal-aura{
  position:absolute!important;
  inset:-9px!important;
  border-radius:50%!important;
  background:conic-gradient(
    from 0deg,
    rgba(255,153,51,.60) 0deg,
    rgba(255,153,51,.06) 75deg,
    rgba(255,255,255,.30) 150deg,
    rgba(91,159,212,.14) 195deg,
    rgba(19,136,8,.55) 280deg,
    rgba(19,136,8,.06) 340deg,
    rgba(255,153,51,.60) 360deg
  )!important;
  filter:blur(11px) saturate(125%)!important;
  opacity:.55!important;
  z-index:0!important;
  animation:brand-aura-spin 16s linear infinite!important;
  pointer-events:none!important;
}

/* ---------- 2. 24-spoke chakra ring ---------- */
.brand-seal-chakra{
  position:absolute!important;
  inset:5px!important;
  border-radius:50%!important;
  background:repeating-conic-gradient(
    from 0deg,
    rgba(143,208,255,.75) 0deg 1.5deg,
    transparent 1.5deg 15deg
  )!important;
  -webkit-mask:radial-gradient(
    circle at center,
    transparent 60%,
    #000 63%,
    #000 76%,
    transparent 79%
  )!important;
  mask:radial-gradient(
    circle at center,
    transparent 60%,
    #000 63%,
    #000 76%,
    transparent 79%
  )!important;
  z-index:1!important;
  animation:brand-chakra-spin 60s linear infinite!important;
  pointer-events:none!important;
  opacity:.85!important;
}

/* ---------- 3. Pulsing orbital rings ---------- */
.brand-seal-ring{
  position:absolute!important;
  border-radius:50%!important;
  pointer-events:none!important;
  z-index:2!important;
  border:1px solid transparent!important;
}
.brand-seal-ring.ring-a{
  inset:2px!important;
  border-color:rgba(143,208,255,.34)!important;
  animation:brand-ring-pulse 3.6s ease-in-out infinite!important;
}
.brand-seal-ring.ring-b{
  inset:11px!important;
  border-color:rgba(255,153,51,.28)!important;
  animation:brand-ring-pulse 5.2s ease-in-out infinite .8s!important;
}

/* ---------- 4. Parliament dome ---------- */
.brand-seal-dome{
  position:absolute!important;
  width:20px!important;
  height:9px!important;
  top:14px!important;
  left:50%!important;
  transform:translateX(-50%)!important;
  background:linear-gradient(180deg,#F4F8FC 0%,#A8D4F5 100%)!important;
  border-radius:10px 10px 0 0!important;
  box-shadow:
    0 0 6px rgba(143,208,255,.45),
    0 0 14px rgba(143,208,255,.18)!important;
  z-index:3!important;
  animation:brand-dome-glow 4.4s ease-in-out infinite!important;
}
/* Tiny spire on top of the dome */
.brand-seal-dome:after{
  content:""!important;
  position:absolute!important;
  top:-3px!important;
  left:50%!important;
  transform:translateX(-50%)!important;
  width:2px!important;
  height:3px!important;
  background:#F4F8FC!important;
  border-radius:1px 1px 0 0!important;
  box-shadow:0 0 5px rgba(143,208,255,.75)!important;
}
/* Central pinnacle dot inside the dome */
.brand-seal-dome:before{
  content:""!important;
  position:absolute!important;
  top:2px!important;
  left:50%!important;
  transform:translateX(-50%)!important;
  width:3px!important;
  height:3px!important;
  border-radius:50%!important;
  background:rgba(11,19,28,.9)!important;
  box-shadow:0 0 3px rgba(0,0,0,.55)!important;
}

/* ---------- 5. Plinth bar ---------- */
.brand-seal-plinth{
  position:absolute!important;
  width:26px!important;
  height:1.5px!important;
  top:24px!important;
  left:50%!important;
  transform:translateX(-50%)!important;
  background:linear-gradient(
    90deg,
    transparent 0%,
    rgba(143,208,255,.35) 20%,
    rgba(143,208,255,.9) 50%,
    rgba(143,208,255,.35) 80%,
    transparent 100%
  )!important;
  border-radius:1px!important;
  z-index:3!important;
  box-shadow:0 0 4px rgba(143,208,255,.35)!important;
}

/* ---------- 6. Three-figure triad ---------- */
.brand-seal-triad{
  position:absolute!important;
  top:31px!important;
  left:50%!important;
  transform:translateX(-50%)!important;
  display:flex!important;
  gap:3px!important;
  align-items:flex-end!important;
  z-index:3!important;
}
.seal-fig{
  display:block!important;
  width:4px!important;
  height:9px!important;
  border-radius:2px!important;
  position:relative!important;
  transition:transform .3s ease, filter .3s ease!important;
}
/* Head above each figure */
.seal-fig:before{
  content:""!important;
  position:absolute!important;
  top:-4px!important;
  left:50%!important;
  transform:translateX(-50%)!important;
  width:4.5px!important;
  height:4.5px!important;
  border-radius:50%!important;
  background:inherit!important;
}
.fig-saffron{
  background:linear-gradient(180deg,#FF9933,#E68A00)!important;
  box-shadow:0 0 6px rgba(255,153,51,.65)!important;
  animation:brand-fig-pulse 3.2s ease-in-out infinite 0s!important;
}
.fig-blue{
  background:linear-gradient(180deg,#8FD0FF,#5B9FD4)!important;
  box-shadow:0 0 6px rgba(143,208,255,.6)!important;
  animation:brand-fig-pulse 3.2s ease-in-out infinite .35s!important;
}
.fig-green{
  background:linear-gradient(180deg,#1DA30B,#0E6606)!important;
  box-shadow:0 0 6px rgba(19,136,8,.6)!important;
  animation:brand-fig-pulse 3.2s ease-in-out infinite .7s!important;
}

/* ---------- 7. Wordmark beside the seal ---------- */
.brand-lockup .brand-text-wrap{
  display:flex!important;
  flex-direction:column!important;
  justify-content:center!important;
  min-width:0!important;
  gap:3px!important;
}
.brand-lockup .brand-mark{
  font:800 24px/1 Bahnschrift,"Segoe UI",sans-serif!important;
  letter-spacing:-.04em!important;
  background:linear-gradient(120deg,#F4F8FC 0%,#A8D4F5 55%,#C9B0FF 100%)!important;
  -webkit-background-clip:text!important;
  background-clip:text!important;
  -webkit-text-fill-color:transparent!important;
  color:transparent!important;
  transition:filter .25s ease!important;
}
.brand-lockup .brand-sub{
  display:flex!important;
  align-items:center!important;
  gap:7px!important;
  font:700 10px/1.2 Cascadia Mono,Consolas,monospace!important;
  letter-spacing:.22em!important;
  color:#8FD0FF!important;
  text-shadow:0 0 18px rgba(143,208,255,.28)!important;
  margin-top:2px!important;
}
.brand-live-dot{
  display:inline-block!important;
  width:5px!important;
  height:5px!important;
  border-radius:50%!important;
  background:#62D39B!important;
  box-shadow:
    0 0 0 2px rgba(98,211,155,.14),
    0 0 8px rgba(98,211,155,.72)!important;
  animation:brand-live-pulse 2.1s ease-in-out infinite!important;
}

/* ---------- 8. Interaction — hover ---------- */
.brand-seal:hover{
  transform:translateY(-3px) scale(1.05)!important;
  border-color:rgba(143,208,255,.55)!important;
  box-shadow:
    0 16px 38px rgba(0,0,0,.42),
    0 0 0 1px rgba(143,208,255,.10) inset,
    0 0 42px rgba(126,200,255,.24),
    0 0 74px rgba(255,153,51,.10)!important;
}
.brand-seal:hover .brand-seal-aura{
  opacity:.95!important;
  filter:blur(13px) saturate(145%)!important;
}
.brand-seal:hover .brand-seal-chakra{
  opacity:1!important;
}
.brand-seal:hover .brand-seal-dome{
  filter:brightness(1.15)!important;
}
.brand-seal:hover .seal-fig{
  transform:translateY(-1px) scaleY(1.08)!important;
  filter:brightness(1.12)!important;
}
.brand-seal:hover .brand-seal-ring.ring-a{
  border-color:rgba(143,208,255,.6)!important;
}
.brand-seal:hover .brand-seal-ring.ring-b{
  border-color:rgba(255,153,51,.55)!important;
}
.brand-lockup:hover .brand-mark{
  filter:drop-shadow(0 0 14px rgba(143,208,255,.38))!important;
}

/* Click feedback */
.brand-seal:active{
  transform:translateY(-1px) scale(.94)!important;
}

/* ---------- 9. Collapsed sidebar — the seal becomes the whole brand ---------- */
.sidebar-collapsed .brand-top{
  flex-direction:column!important;
  align-items:center!important;
  justify-content:center!important;
  gap:14px!important;
}
.sidebar-collapsed .brand-lockup{
  justify-content:center!important;
  gap:0!important;
}
.sidebar-collapsed .brand-text-wrap{
  display:none!important;
}
.sidebar-collapsed .brand-seal{
  width:48px!important;
  height:48px!important;
  flex:0 0 48px!important;
  border-radius:15px!important;
}
.sidebar-collapsed .brand-seal-dome{width:17px!important;height:8px!important;top:11px!important}
.sidebar-collapsed .brand-seal-plinth{width:22px!important;top:21px!important}
.sidebar-collapsed .brand-seal-triad{top:27px!important;gap:2.5px!important}
.sidebar-collapsed .seal-fig{width:3.5px!important;height:8px!important}
.sidebar-collapsed .brand-seal-aura{inset:-7px!important}

/* ---------- 10. Keyframes ---------- */
@keyframes brand-aura-spin{
  to{transform:rotate(360deg)}
}
@keyframes brand-chakra-spin{
  to{transform:rotate(360deg)}
}
@keyframes brand-ring-pulse{
  0%,100%{opacity:.32;transform:scale(1)}
  50%{opacity:.95;transform:scale(1.10)}
}
@keyframes brand-seal-float{
  0%,100%{transform:translateY(0)}
  50%{transform:translateY(-2px)}
}
@keyframes brand-dome-glow{
  0%,100%{box-shadow:0 0 6px rgba(143,208,255,.45), 0 0 14px rgba(143,208,255,.18)}
  50%{box-shadow:0 0 10px rgba(143,208,255,.7), 0 0 22px rgba(143,208,255,.32)}
}
@keyframes brand-fig-pulse{
  0%,100%{filter:brightness(1)}
  50%{filter:brightness(1.25) drop-shadow(0 0 5px currentColor)}
}
@keyframes brand-live-pulse{
  0%,100%{opacity:.55;transform:scale(.9)}
  50%{opacity:1;transform:scale(1.15)}
}

/* ---------- 11. Reduced-motion safety ---------- */
@media (prefers-reduced-motion:reduce){
  .brand-seal,
  .brand-seal-aura,
  .brand-seal-chakra,
  .brand-seal-ring,
  .brand-seal-dome,
  .seal-fig,
  .brand-live-dot{
    animation:none!important;
  }
  .brand-seal-aura{opacity:.35!important}
}

/* ============================================================
   FINAL SEAL VIEWER UX OVERRIDES
   Must remain last so legacy viewer rules cannot overlap the mark.
   ============================================================ */
.seal-viewer{
  align-items:center!important;
  justify-content:center!important;
  padding:18px!important;
}
.seal-viewer-card{
  width:min(720px,94vw)!important;
  max-height:calc(100vh - 32px)!important;
  min-height:0!important;
  padding:30px 34px 20px!important;
  border-radius:36px!important;
  overflow:hidden!important;
  display:flex!important;
  flex-direction:column!important;
  align-items:center!important;
  justify-content:flex-start!important;
  gap:0!important;
}
.seal-viewer-heading{
  width:100%!important;
  padding:0 62px!important;
  text-align:center!important;
  position:relative!important;
  z-index:3!important;
}
.seal-viewer-eyebrow{
  display:block!important;
  color:#6FA7C7!important;
  font:800 8px/1.1 Cascadia Mono,Consolas,monospace!important;
  letter-spacing:.34em!important;
  margin-bottom:9px!important;
}
.seal-viewer-title{
  display:flex!important;
  justify-content:center!important;
  align-items:baseline!important;
  flex-wrap:wrap!important;
  gap:10px!important;
  line-height:.96!important;
}
.seal-viewer-title-main{
  color:#F5F8FB!important;
  font:800 clamp(24px,4vw,38px)/.96 Bahnschrift,"Segoe UI",Arial,sans-serif!important;
  letter-spacing:-.045em!important;
}
.seal-viewer-title-accent{
  color:#8FD0FF!important;
  font:800 clamp(11px,1.8vw,15px)/1 Cascadia Mono,Consolas,monospace!important;
  letter-spacing:.17em!important;
}
.seal-viewer-subtitle{
  margin:12px auto 0!important;
  max-width:560px!important;
  color:#77909E!important;
  font:700 8px/1.55 Cascadia Mono,Consolas,monospace!important;
  letter-spacing:.11em!important;
}
.seal-viewer-status{
  display:inline-flex!important;
  align-items:center!important;
  justify-content:center!important;
  gap:7px!important;
  margin-top:13px!important;
  padding:6px 11px!important;
  border-radius:999px!important;
  border:1px solid rgba(98,211,155,.15)!important;
  background:rgba(98,211,155,.035)!important;
}
.seal-viewer-status-dot{
  width:5px!important;
  height:5px!important;
  flex:0 0 5px!important;
  border-radius:50%!important;
  background:#62D39B!important;
  box-shadow:0 0 0 3px rgba(98,211,155,.09),0 0 9px rgba(98,211,155,.60)!important;
}
.seal-viewer-status-text{
  color:#8BC9A9!important;
  font:800 7px/1 Cascadia Mono,Consolas,monospace!important;
  letter-spacing:.18em!important;
}
.seal-viewer-status-sep{
  color:#536976!important;
  font:700 8px/1 Cascadia Mono,Consolas,monospace!important;
}
.seal-viewer-status-muted{
  color:#5D7583!important;
  font:700 7px/1 Cascadia Mono,Consolas,monospace!important;
  letter-spacing:.14em!important;
}
.seal-viewer-seal-wrap{
  position:relative!important;
  z-index:2!important;
  width:300px!important;
  height:300px!important;
  flex:0 0 300px!important;
  margin:12px 0 2px!important;
  transform:none!important;
  filter:none!important;
}
.seal-viewer-seal-wrap:before,
.seal-viewer-seal-wrap:after{
  content:""!important;
  position:absolute!important;
  left:50%!important;
  top:50%!important;
  border-radius:50%!important;
  transform:translate(-50%,-50%)!important;
  pointer-events:none!important;
}
.seal-viewer-seal-wrap:before{
  width:282px!important;
  height:282px!important;
  border:1px solid rgba(143,208,255,.10)!important;
  box-shadow:0 0 34px rgba(143,208,255,.07),inset 0 0 34px rgba(143,208,255,.035)!important;
}
.seal-viewer-seal-wrap:after{
  width:246px!important;
  height:246px!important;
  border:1px dashed rgba(255,153,51,.10)!important;
  animation:seal-orbit-guide 18s linear infinite!important;
}
.seal-viewer-seal{
  position:absolute!important;
  left:50%!important;
  top:50%!important;
  width:58px!important;
  height:58px!important;
  margin:0!important;
  transform:translate(-50%,-50%) scale(4.62)!important;
  transform-origin:center!important;
  cursor:default!important;
  pointer-events:none!important;
  filter:drop-shadow(0 24px 34px rgba(0,0,0,.30))!important;
}
.seal-viewer-seal:hover{
  transform:translate(-50%,-50%) scale(4.62)!important;
}
.seal-viewer-chips{
  position:relative!important;
  z-index:3!important;
  display:flex!important;
  flex-wrap:wrap!important;
  justify-content:center!important;
  gap:8px!important;
  margin:0!important;
  transform:none!important;
}
.seal-viewer-chip{
  min-width:74px!important;
  padding:7px 11px!important;
  border-radius:999px!important;
  color:#9BB7C5!important;
  border:1px solid rgba(143,208,255,.15)!important;
  background:linear-gradient(180deg,rgba(143,208,255,.055),rgba(143,208,255,.025))!important;
  font:800 7px/1 Cascadia Mono,Consolas,monospace!important;
  letter-spacing:.15em!important;
}
.seal-viewer-bottom{
  position:relative!important;
  z-index:3!important;
  display:flex!important;
  flex-direction:column!important;
  align-items:center!important;
  gap:9px!important;
  width:100%!important;
}
.seal-viewer-caption{
  color:#647D8B!important;
  font:700 8px/1.45 Cascadia Mono,Consolas,monospace!important;
  letter-spacing:.04em!important;
  text-align:center!important;
}
.seal-viewer-hint{
  position:relative!important;
  z-index:3!important;
  display:flex!important;
  justify-content:center!important;
  align-items:center!important;
  gap:6px!important;
  flex-wrap:wrap!important;
  width:min(460px,82vw)!important;
  margin-top:15px!important;
  padding-top:12px!important;
  border-top:1px solid rgba(143,208,255,.08)!important;
  color:#596F7C!important;
  font:700 7px/1.4 Cascadia Mono,Consolas,monospace!important;
  text-align:center!important;
  letter-spacing:.06em!important;
}
.seal-viewer-hint-strong{
  color:#8CAEBE!important;
  font-weight:800!important;
  letter-spacing:.12em!important;
}
.seal-viewer-hint-or{
  color:#485D69!important;
}
.seal-viewer-hint-tail{
  color:#566E7B!important;
}
.seal-viewer-close{
  position:absolute!important;
  top:17px!important;
  right:17px!important;
  z-index:6!important;
  width:44px!important;
  height:44px!important;
  border-radius:14px!important;
  border:1px solid rgba(143,208,255,.16)!important;
  background:rgba(8,16,24,.72)!important;
  color:#A9BFCD!important;
}
@keyframes seal-orbit-guide{
  to{transform:translate(-50%,-50%) rotate(360deg)}
}
@media (max-height:720px){
  .seal-viewer-card{
    padding:22px 26px 16px!important;
    border-radius:30px!important;
  }
  .seal-viewer-heading{padding:0 40px!important}
  .seal-viewer-eyebrow{margin-bottom:6px!important}
  .seal-viewer-subtitle{margin-top:8px!important}
  .seal-viewer-status{margin-top:9px!important;padding:5px 10px!important}
  .seal-viewer-seal-wrap{
    width:230px!important;
    height:230px!important;
    flex-basis:230px!important;
    margin:6px 0 0!important;
  }
  .seal-viewer-seal{transform:translate(-50%,-50%) scale(3.55)!important}
  .seal-viewer-seal:hover{transform:translate(-50%,-50%) scale(3.55)!important}
  .seal-viewer-seal-wrap:before{width:214px!important;height:214px!important}
  .seal-viewer-seal-wrap:after{width:188px!important;height:188px!important}
  .seal-viewer-hint{margin-top:10px!important;padding-top:9px!important}
}
@media (max-width:520px){
  .seal-viewer{padding:10px!important}
  .seal-viewer-card{
    width:min(96vw,620px)!important;
    padding:22px 14px 15px!important;
  }
  .seal-viewer-heading{padding:0 34px!important}
  .seal-viewer-title-main{font-size:25px!important}
  .seal-viewer-title-accent{font-size:10px!important}
  .seal-viewer-subtitle{
    max-width:330px!important;
    font-size:7px!important;
    letter-spacing:.07em!important;
  }
  .seal-viewer-seal-wrap{
    width:230px!important;
    height:230px!important;
    flex-basis:230px!important;
  }
  .seal-viewer-seal{transform:translate(-50%,-50%) scale(3.55)!important}
  .seal-viewer-seal:hover{transform:translate(-50%,-50%) scale(3.55)!important}
  .seal-viewer-seal-wrap:before{width:214px!important;height:214px!important}
  .seal-viewer-seal-wrap:after{width:188px!important;height:188px!important}
  .seal-viewer-close{top:12px!important;right:12px!important}
}
@media (prefers-reduced-motion:reduce){
  .seal-viewer-status-dot,
  .seal-viewer-seal,
  .seal-viewer-seal-wrap:after{
    animation:none!important;
  }
}

</style>
""" + "</head>")

# ============================================================
# RUNTIME / FILTER CALLBACKS
# ============================================================

# ============================================================
# AI COPILOT STYLE + KEYBOARD INTERACTION
# ============================================================

app.index_string = app.index_string.replace("</head>", r"""
<style>
.copilot-root{position:relative;z-index:1200}
.copilot-launcher{position:fixed;right:24px;bottom:24px;z-index:1210;display:inline-flex;align-items:center;gap:9px;min-width:150px;height:46px;padding:0 16px;border-radius:999px;border:1px solid rgba(143,208,255,.28);background:linear-gradient(135deg,rgba(17,34,47,.96),rgba(10,17,24,.98));color:#EAF5FC;cursor:pointer;box-shadow:0 18px 48px rgba(0,0,0,.42),0 0 32px rgba(126,200,255,.10);backdrop-filter:blur(16px);font:800 11px/1 "Cascadia Mono",Consolas,monospace;letter-spacing:.12em;transition:all .2s ease}
.copilot-launcher:hover{transform:translateY(-3px);border-color:rgba(143,208,255,.62);box-shadow:0 22px 56px rgba(0,0,0,.48),0 0 46px rgba(126,200,255,.18)}
.copilot-launch-icon{font-size:16px;color:#9DD7FF}.copilot-launch-live{color:#6BCB77;font-size:9px;animation:copilotLive 1.8s ease-in-out infinite}.copilot-launch-label{white-space:nowrap}
@keyframes copilotLive{0%,100%{opacity:.55;transform:scale(.9)}50%{opacity:1;transform:scale(1.1)}}
.copilot-panel{position:fixed;right:24px;bottom:82px;z-index:1220;width:min(440px,calc(100vw - 28px));height:min(740px,calc(100vh - 110px));display:flex;flex-direction:column;overflow:hidden;border:1px solid rgba(143,208,255,.20);border-radius:24px;background:radial-gradient(circle at 100% 0%,rgba(143,208,255,.10),transparent 30%),radial-gradient(circle at 0% 100%,rgba(199,146,234,.08),transparent 28%),linear-gradient(180deg,#0D151D 0%,#091017 100%);box-shadow:0 32px 90px rgba(0,0,0,.58),0 0 70px rgba(112,178,228,.12);backdrop-filter:blur(22px);transform-origin:bottom right;transition:opacity .18s ease,transform .22s ease,visibility .18s ease}
.copilot-panel.is-hidden{opacity:0;visibility:hidden;pointer-events:none;transform:translateY(16px) scale(.97)}.copilot-panel.is-open{opacity:1;visibility:visible;pointer-events:auto;transform:none}
.copilot-header{display:flex;justify-content:space-between;gap:12px;padding:16px 17px 12px;border-bottom:1px solid rgba(255,255,255,.055)}
.copilot-brand{display:flex;gap:6px;color:#9FD8FF;font:800 9px "Cascadia Mono",Consolas,monospace;letter-spacing:.18em}.copilot-brand-star{color:#F0C675}.copilot-title{margin-top:4px;color:#F0F6FA;font:800 19px/1 Bahnschrift,"Segoe UI",sans-serif}.copilot-status{display:flex;gap:6px;align-items:center;margin-top:6px;color:#7190A4;font:700 8px "Cascadia Mono",Consolas,monospace;letter-spacing:.15em}.copilot-status-dot{color:#6BCB77}.copilot-heading-actions{display:flex;gap:6px}.copilot-icon-btn{width:34px;height:34px;border-radius:11px;border:1px solid rgba(143,208,255,.12);background:rgba(255,255,255,.03);color:#8FA8BA;cursor:pointer;font-size:18px}.copilot-icon-btn:hover{border-color:rgba(143,208,255,.38);color:#EEF6FB}
.copilot-context-ribbon{margin:12px 14px 0;padding:10px 11px;border:1px solid rgba(143,208,255,.12);border-radius:14px;background:rgba(12,24,34,.82)}.copilot-context-title{color:#7291A4;font:800 8px "Cascadia Mono",Consolas,monospace;letter-spacing:.18em}.copilot-context-row{display:flex;flex-wrap:wrap;gap:6px;margin-top:7px}.copilot-context-chip{padding:5px 8px;border-radius:999px;border:1px solid rgba(143,208,255,.11);background:rgba(143,208,255,.035);color:#BFD3E1;font:700 8px "Cascadia Mono",Consolas,monospace}.copilot-context-chip strong{color:#EDF6FB}.copilot-context-scope{margin-top:7px;color:#6F8999;font:700 8px "Cascadia Mono",Consolas,monospace}
.copilot-messages{flex:1;min-height:0;overflow-y:auto;padding:14px;scrollbar-width:thin}.copilot-welcome{padding:15px;border:1px solid rgba(143,208,255,.11);border-radius:17px;background:rgba(143,208,255,.04)}.copilot-welcome-kicker{color:#88CFF9;font:800 8px "Cascadia Mono",Consolas,monospace;letter-spacing:.17em}.copilot-welcome-title{margin-top:6px;color:#EFF7FB;font:750 21px/1.08 Bahnschrift,"Segoe UI",sans-serif}.copilot-welcome-copy{margin-top:7px;color:#8499A8;font:12px/1.55 "Segoe UI",sans-serif}
.copilot-message{display:flex;margin:8px 0}.copilot-message-user{justify-content:flex-end}.copilot-bubble{max-width:88%;padding:10px 12px;border-radius:15px;font:12px/1.5 "Segoe UI",sans-serif}.copilot-message-user .copilot-bubble{background:linear-gradient(135deg,#183145,#112230);border:1px solid rgba(143,208,255,.20);color:#EDF7FC;border-bottom-right-radius:5px}.copilot-message-assistant .copilot-bubble{background:rgba(255,255,255,.026);border:1px solid rgba(255,255,255,.07);color:#D5E2EA;border-bottom-left-radius:5px}.copilot-role{margin-bottom:5px;color:#6F8A9D;font:800 7px "Cascadia Mono",Consolas,monospace;letter-spacing:.17em}.copilot-sources{display:flex;flex-wrap:wrap;gap:5px;margin-top:7px}.copilot-source-chip{padding:4px 7px;border-radius:999px;border:1px solid rgba(143,208,255,.11);background:rgba(143,208,255,.035);color:#7EA7BF;font:700 7px "Cascadia Mono",Consolas,monospace}
.copilot-quick-wrap{padding:10px 14px 8px;border-top:1px solid rgba(255,255,255,.045)}.copilot-section-label{color:#667E8D;font:800 7px "Cascadia Mono",Consolas,monospace;letter-spacing:.17em;margin-bottom:7px}.copilot-quick-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:6px}.copilot-quick-btn{min-height:32px;padding:7px 8px;border-radius:10px;border:1px solid rgba(143,208,255,.10);background:rgba(255,255,255,.025);color:#9EB6C5;cursor:pointer;font:700 8px "Cascadia Mono",Consolas,monospace;text-align:left}.copilot-quick-btn:hover{border-color:rgba(143,208,255,.30);color:#ECF5FA}
.copilot-composer{display:flex;gap:8px;padding:10px 14px 11px}.copilot-input{flex:1;min-height:58px;resize:none;padding:11px 12px;border-radius:14px;border:1px solid rgba(143,208,255,.12);background:#0B131B;color:#EEF6FA;outline:none;font:12px/1.45 "Segoe UI",sans-serif}.copilot-input:focus{border-color:rgba(143,208,255,.38);box-shadow:0 0 0 3px rgba(143,208,255,.07)}.copilot-input::placeholder{color:#5E7382}.copilot-send{width:74px;border-radius:14px;border:1px solid rgba(143,208,255,.24);background:linear-gradient(180deg,#183246,#102433);color:#EAF6FB;cursor:pointer;font:800 8px "Cascadia Mono",Consolas,monospace;letter-spacing:.14em;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:5px}.copilot-send-icon{font-size:18px;color:#8FD0FF}.copilot-footer{display:flex;justify-content:space-between;gap:10px;padding:0 14px 12px}.copilot-foot-note,.copilot-key-note{color:#536C7B;font:700 7px "Cascadia Mono",Consolas,monospace}
.overview-ai-btn{display:inline-flex;align-items:center;gap:7px;margin-top:14px;padding:9px 13px;border-radius:999px;border:1px solid rgba(143,208,255,.18);background:rgba(143,208,255,.045);color:#A9C7D8;cursor:pointer;font:800 8px "Cascadia Mono",Consolas,monospace;letter-spacing:.12em}.overview-ai-btn:hover{border-color:rgba(143,208,255,.42);color:#F0F7FB}.overview-ai-star,.deep-ask-icon{color:#F0C675}
.deep-panel-toolbar{display:flex;align-items:center;justify-content:space-between;gap:10px;padding-bottom:2px}.deep-ask-ai-btn{display:inline-flex;align-items:center;gap:5px;padding:5px 8px;border-radius:999px;border:1px solid rgba(143,208,255,.10);background:rgba(143,208,255,.025);color:#6F9AB0;cursor:pointer;font:800 7px "Cascadia Mono",Consolas,monospace;letter-spacing:.10em}.deep-ask-ai-btn:hover{color:#DCEEF8;border-color:rgba(143,208,255,.32)}.action-ai{border-color:rgba(143,208,255,.18)!important;background:rgba(143,208,255,.035)!important}.action-ai:hover{border-color:rgba(143,208,255,.42)!important}
@media(max-width:850px){.copilot-launcher{right:14px;bottom:14px;width:52px;min-width:52px;height:52px;padding:0;justify-content:center;border-radius:16px}.copilot-launch-label,.copilot-launch-live{display:none}.copilot-panel{right:12px;bottom:76px;width:calc(100vw - 24px);height:min(720px,calc(100vh - 92px));border-radius:20px}}
.copilot-context-scope{display:flex!important;flex-wrap:wrap!important;gap:6px!important}.copilot-context-scope-pill{padding:4px 7px;border-radius:999px;border:1px solid rgba(143,208,255,.08);background:rgba(255,255,255,.018);color:#6E8796;font:700 7px "Cascadia Mono",Consolas,monospace;letter-spacing:.02em}.copilot-quick-btn:active{transform:translateY(1px)}.copilot-input{transition:border-color .18s ease,box-shadow .18s ease}.copilot-panel.is-open .copilot-input{scroll-margin-bottom:12px}.copilot-md p:first-child{margin-top:0}.copilot-md p:last-child{margin-bottom:0}.copilot-md code{font-family:"Cascadia Mono",Consolas,monospace;color:#9ED7FA;background:rgba(143,208,255,.055);padding:1px 4px;border-radius:4px}.copilot-md strong{color:#F0F7FB}.copilot-welcome-copy{max-width:58ch}
@media(prefers-reduced-motion:reduce){.copilot-launcher,.copilot-launch-live,.copilot-panel{animation:none!important;transition:none!important}}
/* ============================================================
   COPILOT v2 — minimalist shell, deeper intelligence underneath
   ============================================================ */
.copilot-root{position:relative;z-index:1200}
.copilot-launcher{
  position:fixed;right:24px;bottom:24px;z-index:1210;
  display:inline-flex;align-items:center;gap:8px;min-width:138px;height:44px;padding:0 15px;
  border-radius:999px;border:1px solid rgba(143,208,255,.24);
  background:rgba(10,18,25,.92);color:#EAF5FC;cursor:pointer;
  box-shadow:0 16px 40px rgba(0,0,0,.38),0 0 26px rgba(126,200,255,.08);
  backdrop-filter:blur(18px);font:800 10px/1 Cascadia Mono,Consolas,monospace;letter-spacing:.12em;
  transition:transform .18s ease,border-color .18s ease,box-shadow .18s ease
}
.copilot-launcher:hover{transform:translateY(-2px);border-color:rgba(143,208,255,.48);box-shadow:0 20px 48px rgba(0,0,0,.44),0 0 34px rgba(126,200,255,.13)}
.copilot-launch-icon{font-size:15px;color:#9DD7FF}.copilot-launch-live{color:#6BCB77;font-size:8px}.copilot-launch-label{white-space:nowrap}
.copilot-panel{
  position:fixed;right:24px;bottom:78px;z-index:1220;
  width:min(480px,calc(100vw - 28px));height:min(690px,calc(100vh - 96px));
  display:flex;flex-direction:column;overflow:hidden;
  border:1px solid rgba(143,208,255,.16);border-radius:22px;
  background:linear-gradient(180deg,#0B131B 0%,#081016 100%);
  box-shadow:0 28px 80px rgba(0,0,0,.62),0 0 45px rgba(112,178,228,.08);
  backdrop-filter:blur(24px);transform-origin:bottom right;
  transition:opacity .18s ease,transform .20s ease,visibility .18s ease
}
.copilot-panel:after{content:"";position:absolute;inset:0;pointer-events:none;border-radius:inherit;box-shadow:0 0 0 1px rgba(255,255,255,.018) inset}
.copilot-panel.is-hidden{opacity:0;visibility:hidden;pointer-events:none;transform:translateY(10px) scale(.985)}
.copilot-panel.is-open{opacity:1;visibility:visible;pointer-events:auto;transform:none}
.copilot-header{min-height:56px;display:flex;align-items:center;justify-content:space-between;gap:12px;padding:10px 13px;border-bottom:1px solid rgba(255,255,255,.045)}
.copilot-heading-copy{display:flex;align-items:center;gap:9px;min-width:0}.copilot-title{margin:0;color:#F0F6FA;font:800 16px/1 Bahnschrift,"Segoe UI",sans-serif;letter-spacing:.01em}.copilot-status{display:flex;align-items:center;gap:5px;color:#71899A;font:700 7px/1 Cascadia Mono,Consolas,monospace;letter-spacing:.12em;white-space:nowrap}.copilot-status-dot{color:#67C77A;font-size:7px}.copilot-heading-actions{display:flex;gap:5px}.copilot-icon-btn{width:32px;height:32px;border-radius:10px;border:1px solid rgba(143,208,255,.10);background:rgba(255,255,255,.02);color:#849AAA;cursor:pointer;font-size:18px;line-height:1}.copilot-icon-btn:hover{border-color:rgba(143,208,255,.28);color:#EEF6FB}
.copilot-context-ribbon{margin:9px 12px 0;padding:8px 9px;border:1px solid rgba(143,208,255,.09);border-radius:12px;background:rgba(10,21,30,.72)}
.copilot-context-title{color:#698697;font:800 7px/1 Cascadia Mono,Consolas,monospace;letter-spacing:.16em}.copilot-context-row{display:flex;flex-wrap:wrap;gap:5px;margin-top:6px}.copilot-context-chip{padding:4px 7px;border-radius:999px;border:1px solid rgba(143,208,255,.08);background:rgba(143,208,255,.025);color:#A8BFCE;font:700 7px/1 Cascadia Mono,Consolas,monospace}.copilot-context-chip strong{color:#E6F1F7}.copilot-context-scope{display:flex;flex-wrap:wrap;gap:5px;margin-top:6px}.copilot-context-scope-pill{padding:3px 6px;border-radius:999px;color:#627C8D;font:700 7px/1 Cascadia Mono,Consolas,monospace;background:transparent;border:0}
.copilot-messages{flex:1;min-height:0;overflow-y:auto;padding:13px 12px 10px;scrollbar-width:thin;scroll-behavior:smooth}.copilot-welcome{padding:14px 13px;border:1px solid rgba(143,208,255,.08);border-radius:15px;background:rgba(255,255,255,.018)}.copilot-welcome-kicker{color:#78AECB;font:800 7px/1 Cascadia Mono,Consolas,monospace;letter-spacing:.16em}.copilot-welcome-title{margin-top:5px;color:#F0F6FA;font:750 20px/1.05 Bahnschrift,"Segoe UI",sans-serif}.copilot-welcome-copy{margin-top:7px;max-width:54ch;color:#7F97A6;font:11px/1.55 "Segoe UI",Arial,sans-serif}
.copilot-message{display:flex;margin:7px 0}.copilot-message-user{justify-content:flex-end}.copilot-bubble{max-width:91%;padding:9px 11px;border-radius:14px;font:12px/1.48 "Segoe UI",Arial,sans-serif}.copilot-message-user .copilot-bubble{background:#112534;border:1px solid rgba(143,208,255,.17);color:#EEF7FB;border-bottom-right-radius:5px}.copilot-message-assistant .copilot-bubble{background:rgba(255,255,255,.018);border:1px solid rgba(255,255,255,.06);color:#D7E3EA;border-bottom-left-radius:5px}.copilot-role{margin-bottom:4px;color:#5F7C8E;font:800 6.5px/1 Cascadia Mono,Consolas,monospace;letter-spacing:.16em}.copilot-sources{display:flex;flex-wrap:wrap;gap:4px;margin-top:6px}.copilot-source-chip{padding:3px 6px;border-radius:999px;border:1px solid rgba(143,208,255,.09);background:rgba(143,208,255,.025);color:#6F98AE;font:700 6.5px/1 Cascadia Mono,Consolas,monospace}.copilot-md p{margin:.38em 0}.copilot-md ul,.copilot-md ol{padding-left:18px;margin:.38em 0}.copilot-md code{font-family:Cascadia Mono,Consolas,monospace;color:#9ED7FA;background:rgba(143,208,255,.05);padding:1px 4px;border-radius:4px}.copilot-md strong{color:#F1F7FA}
.copilot-quick-wrap{padding:8px 12px 7px;border-top:1px solid rgba(255,255,255,.04)}.copilot-section-label{color:#5F7787;font:800 6.5px/1 Cascadia Mono,Consolas,monospace;letter-spacing:.15em;margin-bottom:6px}.copilot-quick-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:5px}.copilot-quick-btn{min-height:29px;padding:6px 7px;border-radius:9px;border:1px solid rgba(143,208,255,.08);background:rgba(255,255,255,.018);color:#8EA8B7;cursor:pointer;font:700 7px/1.1 Cascadia Mono,Consolas,monospace;text-align:left;transition:border-color .16s ease,color .16s ease,transform .16s ease}.copilot-quick-btn:hover{border-color:rgba(143,208,255,.25);color:#E2EEF4}.copilot-quick-btn:active{transform:translateY(1px)}
.copilot-composer{display:flex;align-items:stretch;gap:7px;padding:8px 12px 10px}.copilot-input{flex:1;min-height:50px;max-height:92px;resize:vertical;padding:10px 11px;border-radius:13px;border:1px solid rgba(143,208,255,.11);background:#09131B;color:#EEF6FA;outline:none;font:11px/1.42 "Segoe UI",Arial,sans-serif}.copilot-input:focus{border-color:rgba(143,208,255,.34);box-shadow:0 0 0 3px rgba(143,208,255,.045)}.copilot-input::placeholder{color:#536B7A}.copilot-send{width:48px;min-width:48px;border-radius:13px;border:1px solid rgba(143,208,255,.20);background:#102535;color:#EAF6FB;cursor:pointer;display:grid;place-items:center;transition:transform .16s ease,border-color .16s ease,background .16s ease}.copilot-send:hover{border-color:rgba(143,208,255,.38);background:#143046}.copilot-send:active{transform:translateY(1px)}.copilot-send-icon{font-size:18px;color:#8FD0FF}
.copilot-foot-note,.copilot-key-note{display:none}
@media(max-width:850px){.copilot-launcher{right:14px;bottom:14px;width:50px;min-width:50px;height:50px;padding:0;justify-content:center;border-radius:15px}.copilot-launch-label,.copilot-launch-live{display:none}.copilot-panel{right:10px;bottom:72px;width:calc(100vw - 20px);height:min(700px,calc(100vh - 84px));border-radius:19px}.copilot-quick-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
.copilot-backdrop{
  position:fixed;inset:0;z-index:1215;
  background:rgba(2,8,13,.16);
  backdrop-filter:blur(2px);-webkit-backdrop-filter:blur(2px);
  opacity:1;visibility:visible;pointer-events:auto;
  transition:opacity .16s ease,visibility .16s ease;
}
.copilot-backdrop.is-hidden{opacity:0;visibility:hidden;pointer-events:none}
.copilot-backdrop.is-open{opacity:1;visibility:visible;pointer-events:auto}
.copilot-heading-actions{align-items:center}
.copilot-icon-btn{width:36px;height:36px;border-radius:10px;font-size:19px;font-weight:700;line-height:1}
.copilot-icon-btn:last-child{background:rgba(255,255,255,.035);border-color:rgba(143,208,255,.18);color:#B7C8D4}
.copilot-icon-btn:last-child:hover{background:rgba(224,106,120,.10);border-color:rgba(224,106,120,.38);color:#F1AAB1}
.deep-map-inner{padding:0!important;background:transparent!important;border:0!important;box-shadow:none!important}
.deep-ask-ai-btn{transition:transform .15s ease,border-color .15s ease,color .15s ease,background .15s ease}
.deep-ask-ai-btn:active{transform:translateY(1px)}
@media(prefers-reduced-motion:reduce){.copilot-launcher,.copilot-panel,.copilot-backdrop{transition:none!important}}

/* ============================================================
   TEAM INFO v12 — PRECISION FIT / NO INNER SCROLL / CLEAN FOOTER
   Final presentation layer.  This override is intentionally last so it
   wins over the older Team Info experiments without touching other views.
   ============================================================ */
#team-info-modal .team-modal-dialog{
  width:min(1240px,96vw)!important;
  height:min(760px,91vh)!important;
  max-height:min(760px,91vh)!important;
  min-height:0!important;
  overflow:hidden!important;
}
#team-info-modal .team-modal-top{
  flex:0 0 98px!important;
  min-height:98px!important;
  height:98px!important;
  padding:15px 22px 14px!important;
}
#team-info-modal .team-modal-scroll{
  flex:1 1 auto!important;
  min-height:0!important;
  height:auto!important;
  max-height:none!important;
  overflow:hidden!important;
  overflow-y:hidden!important;
  padding:13px 20px 15px!important;
}
#team-info-modal .team-modal-pills{
  height:22px!important;
  margin:0 0 9px!important;
}
#team-info-modal .team-members-grid{
  grid-template-columns:repeat(3,minmax(0,1fr))!important;
  grid-template-rows:repeat(2,1fr)!important;
  gap:10px!important;
  height:calc(100% - 89px)!important;
  align-content:stretch!important;
}
#team-info-modal .team-member-card{
  grid-template-columns:132px minmax(0,1fr)!important;
  grid-template-rows:1fr!important;
  gap:12px!important;
  width:100%!important;
  min-width:0!important;
  min-height:0!important;
  height:100%!important;
  padding:9px!important;
  border-radius:19px!important;
  box-sizing:border-box!important;
  overflow:hidden!important;
}
#team-info-modal .team-member-visual{
  width:132px!important;
  min-width:132px!important;
  height:100%!important;
  min-height:0!important;
}
#team-info-modal .team-member-photo-frame{
  width:132px!important;
  height:100%!important;
  min-height:0!important;
  aspect-ratio:auto!important;
  border-radius:15px!important;
}
#team-info-modal .team-member-avatar,
#team-info-modal .team-member-avatar-photo,
#team-info-modal .team-member-photo{
  width:100%!important;
  height:100%!important;
  min-width:0!important;
  min-height:0!important;
}
#team-info-modal .team-member-photo{
  object-fit:cover!important;
  object-position:center 18%!important;
}
#team-info-modal .team-member-content{
  min-width:0!important;
  min-height:0!important;
  height:100%!important;
  overflow:hidden!important;
  padding:2px 2px 1px 0!important;
  justify-content:center!important;
}
#team-info-modal .team-member-name-row{
  min-width:0!important;
  width:100%!important;
  align-items:flex-start!important;
}
#team-info-modal .team-member-name{
  min-width:0!important;
  max-width:100%!important;
  overflow:visible!important;
  font-size:16px!important;
  line-height:1.05!important;
  white-space:normal!important;
}
#team-info-modal .team-member-role{
  font-size:7px!important;
  line-height:1.22!important;
  margin-top:5px!important;
  white-space:normal!important;
}
#team-info-modal .team-member-degree{
  font-size:8px!important;
  line-height:1.2!important;
  margin-top:5px!important;
}
#team-info-modal .team-member-bio{
  margin-top:7px!important;
  max-width:none!important;
  font-size:8.5px!important;
  line-height:1.34!important;
  overflow:visible!important;
  display:block!important;
}
#team-info-modal .team-member-tags{
  margin-top:auto!important;
  padding-top:7px!important;
}
#team-info-modal .team-capabilities{
  margin-top:9px!important;
  padding:8px 11px!important;
  min-height:52px!important;
  box-sizing:border-box!important;
}
#team-info-modal .team-capabilities-kicker{
  margin-bottom:5px!important;
}
#team-info-modal .team-capabilities-row{
  gap:5px!important;
}
#team-info-modal .team-capability-chip{
  height:20px!important;
  min-height:20px!important;
  padding:0 7px!important;
}
#team-info-modal .team-source-row,
#team-info-modal .team-source-label,
#team-info-modal .team-source-link{
  display:none!important;
}
#team-info-modal .team-modal-github{
  flex:0 0 auto!important;
  min-width:126px!important;
  justify-content:center!important;
}
#team-info-modal .team-modal-title{
  max-width:760px!important;
}

@media(max-width:1180px){
  #team-info-modal .team-modal-dialog{
    width:min(1100px,96vw)!important;
    height:min(760px,92vh)!important;
  }
  #team-info-modal .team-member-card{
    grid-template-columns:118px minmax(0,1fr)!important;
  }
  #team-info-modal .team-member-visual,
  #team-info-modal .team-member-photo-frame{
    width:118px!important;
    min-width:118px!important;
  }
}

@media(max-width:760px){
  #team-info-modal .team-modal-dialog{
    width:96vw!important;
    height:min(760px,94vh)!important;
  }
  #team-info-modal .team-modal-top{
    flex-basis:88px!important;
    min-height:88px!important;
    height:88px!important;
    padding:13px 15px!important;
  }
  #team-info-modal .team-modal-scroll{
    padding:10px!important;
  }
  #team-info-modal .team-members-grid{
    grid-template-columns:repeat(2,minmax(0,1fr))!important;
    grid-template-rows:repeat(3,minmax(0,1fr))!important;
    height:calc(100% - 84px)!important;
    gap:8px!important;
  }
  #team-info-modal .team-member-card{
    grid-template-columns:82px minmax(0,1fr)!important;
    gap:8px!important;
    padding:7px!important;
    border-radius:15px!important;
  }
  #team-info-modal .team-member-visual,
  #team-info-modal .team-member-photo-frame{
    width:82px!important;
    min-width:82px!important;
  }
  #team-info-modal .team-member-photo-frame{border-radius:12px!important}
  #team-info-modal .team-member-name{font-size:11.5px!important}
  #team-info-modal .team-member-role{font-size:5.3px!important}
  #team-info-modal .team-member-degree{font-size:6.8px!important}
  #team-info-modal .team-member-bio{font-size:7px!important;line-height:1.25!important}
  #team-info-modal .team-member-tags{display:none!important}
  #team-info-modal .team-capabilities{display:none!important}
}

</style>
<script>
(function(){
  function clickId(id){var e=document.getElementById(id);if(e){e.click();return true}return false}
  function focusInput(){var e=document.getElementById("copilot-input");if(e){setTimeout(function(){e.focus()},80)}}
  function scrollChat(){var e=document.getElementById("copilot-messages");if(!e)return;var messages=e.querySelectorAll(".copilot-message");if(!messages.length){e.scrollTop=0;return}requestAnimationFrame(function(){e.scrollTop=e.scrollHeight})}
  document.addEventListener("keydown",function(event){
    var key=(event.key||"").toLowerCase();
    if((event.ctrlKey||event.metaKey)&&key==="k"){event.preventDefault();if(clickId("copilot-launcher"))focusInput()}
    if(key==="escape"){var p=document.getElementById("copilot-panel");if(p&&p.classList.contains("is-open"))clickId("copilot-close")}
    if((key==="enter"||key==="return")&&!event.shiftKey&&document.activeElement&&document.activeElement.id==="copilot-input"){event.preventDefault();clickId("copilot-send")}
  });
  document.addEventListener("click",function(event){
    var target=event.target.closest&&event.target.closest("#copilot-launcher,#copilot-prompt-scope,#copilot-prompt-compare,#copilot-prompt-risk,#copilot-prompt-financial,#copilot-prompt-chart,#copilot-prompt-methodology");
    if(target) setTimeout(function(){focusInput();scrollChat()},100);
  });
  var observer=new MutationObserver(function(){var p=document.getElementById("copilot-panel");if(p&&p.classList.contains("is-open"))scrollChat()});
  setTimeout(function(){var m=document.getElementById("copilot-messages");if(m)observer.observe(m,{childList:true,subtree:true})},500);
})();
</script>
</head>""")

@app.callback(
    Output("health-store", "data"),
    Output("options-store", "data"),
    Output("national-store", "data"),
    Output("connection-state", "children"),
    Input("startup", "n_intervals"),
    Input("refresh-button", "n_clicks"),
    prevent_initial_call=False,
)
def load_runtime(_startup: int, _refresh: int | None):
    """Load runtime metadata without conflating endpoint failures.

    /health is the source of truth for FastAPI availability. The filter and
    dashboard-summary endpoints are fetched independently so one secondary
    failure cannot incorrectly paint a healthy API as OFFLINE.
    """
    try:
        health = api_get("/health")
    except requests.RequestException as exc:
        message = html.Div(
            [
                html.Span(className="connection-pulse is-offline"),
                html.Span("FASTAPI", className="connection-state-label"),
                html.Span("OFFLINE", className="connection-state-value"),
            ],
            className="connection-state-content connection-offline",
            title=f"FastAPI unavailable: {exc}",
        )
        return (
            {"error": str(exc)},
            {},
            {"error": "FastAPI unavailable"},
            message,
        )

    source = health.get("data_source", "unknown") if isinstance(health, dict) else "unknown"
    options = {}
    national = {}
    options_error = None
    national_error = None

    try:
        options = api_get("/api/v1/filter-options")
    except requests.RequestException as exc:
        options_error = str(exc)

    try:
        national = api_get("/api/v1/dashboard-summary")
    except requests.RequestException as exc:
        national_error = str(exc)

    degraded = options_error is not None or national_error is not None
    state = "DEGRADED" if degraded else "ONLINE"
    state_class = "connection-state-content connection-degraded" if degraded else "connection-state-content connection-online"
    detail = "health OK · supporting data issue" if degraded else f"health OK · {source}"
    message = html.Div(
        [
            html.Span(className="connection-pulse is-degraded" if degraded else "connection-pulse is-online"),
            html.Span("FASTAPI", className="connection-state-label"),
            html.Span(state, className="connection-state-value"),
        ],
        className=state_class,
        title=detail,
    )

    return health, options, national, message


@app.callback(
    Output("state-filter", "options"),
    Output("district-filter", "options"),
    Output("mp-filter", "options"),
    Output("constituency-filter", "options"),
    Output("category-filter", "options"),
    Output("status-filter", "options"),
    Output("risk-filter", "options"),
    Input("options-store", "data"),
)
def populate_filter_options(options):
    """Populate sidebar filters with clear All-* labels and sorted values."""
    all_labels = [
        "All States",
        "All Districts",
        "All MPs",
        "All Constituencies",
        "All Categories",
        "All Status",
        "All Risk Levels",
    ]
    if not isinstance(options, dict):
        return tuple([{"label": label, "value": "All"}] for label in all_labels)

    names = [
        "states",
        "districts",
        "mps",
        "constituencies",
        "work_categories",
        "work_statuses",
    ]
    result = []
    for label, name in zip(all_labels[:6], names):
        values = [label] + clean_options(options.get(name, []))
        # Keep value "All" for the first entry so filter logic stays stable.
        opts = [{"label": values[0], "value": "All"}]
        opts.extend({"label": v, "value": v} for v in values[1:])
        result.append(opts)

    risk_opts = [{"label": "All Risk Levels", "value": "All"}]
    for value in RISK_ORDER:
        if value in clean_options(options.get("risk_categories", [])):
            risk_opts.append({"label": value, "value": value})
    result.append(risk_opts)

    return tuple(result)


@app.callback(
    Output("district-filter", "value"),
    Input("state-filter", "value"),
    State("district-filter", "value"),
    prevent_initial_call=True,
)
def reset_district_on_state_change(state, current):
    # A state change invalidates the previously selected district.
    if state is None:
        return no_update
    return "All" if state != "All" else ("All" if current != "All" else no_update)


# ============================================================
# MAIN DATA SYNCHRONISATION
# ============================================================

@app.callback(
    Output("filtered-store", "data"),
    Output("works-store", "data"),
    Output("scope-banner", "children"),
    Input("state-filter", "value"),
    Input("district-filter", "value"),
    Input("mp-filter", "value"),
    Input("constituency-filter", "value"),
    Input("category-filter", "value"),
    Input("status-filter", "value"),
    Input("risk-filter", "value"),
    Input("completion-filter", "value"),
    Input("search-filter", "value"),
    Input("min-sanction", "value"),
    Input("max-sanction", "value"),
    Input("risk-range", "value"),
    Input("refresh-button", "n_clicks"),
)
def load_filtered_scope(
    state,
    district,
    mp,
    constituency,
    category,
    status,
    risk,
    completion,
    search,
    min_sanction,
    max_sanction,
    risk_range,
    _refresh,
):
    """Load the live KPI scope independently from deep chart generation.

    Keeping this callback independent means a transient failure in an expensive
    analytics route can never make the entire home page look static or offline.
    """
    if (
        min_sanction
        and max_sanction
        and min_sanction > 0
        and max_sanction > 0
        and min_sanction > max_sanction
    ):
        message = "Correct the sanction range to continue."
        return (
            {"error": "Minimum sanction cannot exceed maximum sanction."},
            {"items": []},
            html.Div(message, className="banner error"),
        )

    params = build_filter_params(
        state,
        district,
        mp,
        constituency,
        category,
        status,
        risk,
        completion,
        search,
        min_sanction,
        max_sanction,
        risk_range,
    )

    try:
        filtered = api_get("/api/v1/filtered-summary", params)
        works = api_get(
            "/api/v1/works",
            {
                **params,
                "page": 1,
                "page_size": QUEUE_LIMIT,
                "sort_by": "priority_score",
                "sort_order": "desc",
            },
        )

        total = safe_int(
            filtered.get("total_works"),
            safe_int(works.get("total_matching")),
        )

        return (
            filtered,
            works,
            html.Div(
                [
                    html.Div("FILTERED MONITORING UNIVERSE", className="banner-kicker"),
                    html.Div(f"{total:,} works in current scope", className="banner-main"),
                    html.Div(
                        f"High/Critical signal rate: {safe_float(filtered.get('high_or_critical_rate_pct')):.1f}%"
                        f" · Queue capped at {QUEUE_LIMIT:,} · Analytics update live",
                        className="banner-sub",
                    ),
                ],
                className="banner",
            ),
        )

    except requests.Timeout as exc:
        message = f"Scope request timed out after {REQUEST_TIMEOUT}s: {exc}"
        return {"error": message}, {"items": []}, html.Div(message, className="banner error")
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "HTTP"
        detail = ""
        try:
            payload = exc.response.json() if exc.response is not None else {}
            detail = payload.get("detail", "") if isinstance(payload, dict) else ""
        except Exception:
            pass
        message = f"FastAPI returned HTTP {status}" + (f": {detail}" if detail else "")
        return {"error": message}, {"items": []}, html.Div(message, className="banner error")
    except requests.RequestException as exc:
        message = f"FastAPI request failed: {exc}"
        return {"error": message}, {"items": []}, html.Div(message, className="banner error")


@app.callback(
    Output("analytics-store", "data"),
    Input("state-filter", "value"),
    Input("district-filter", "value"),
    Input("mp-filter", "value"),
    Input("constituency-filter", "value"),
    Input("category-filter", "value"),
    Input("status-filter", "value"),
    Input("risk-filter", "value"),
    Input("completion-filter", "value"),
    Input("search-filter", "value"),
    Input("min-sanction", "value"),
    Input("max-sanction", "value"),
    Input("risk-range", "value"),
    Input("refresh-button", "n_clicks"),
)
def load_deep_analytics(
    state,
    district,
    mp,
    constituency,
    category,
    status,
    risk,
    completion,
    search,
    min_sanction,
    max_sanction,
    risk_range,
    _refresh,
):
    """Load the complete filter-synchronised deep analytics payload.

    This callback deliberately runs separately from KPI/queue loading. The
    home page therefore remains populated with live data while deep graphs are
    being calculated, and HTTP failures are reported as analytics failures
    rather than incorrectly masquerading as API outages.
    """
    if (
        min_sanction
        and max_sanction
        and min_sanction > 0
        and max_sanction > 0
        and min_sanction > max_sanction
    ):
        return {"error": "Minimum sanction cannot exceed maximum sanction."}

    params = build_filter_params(
        state,
        district,
        mp,
        constituency,
        category,
        status,
        risk,
        completion,
        search,
        min_sanction,
        max_sanction,
        risk_range,
    )

    try:
        return api_get("/api/v1/deep-analytics", params)
    except (requests.Timeout, requests.ConnectionError):
        # One controlled retry handles transient startup/readiness races while
        # keeping the UI truthful if the analytics service is actually broken.
        try:
            return api_get("/api/v1/deep-analytics", params)
        except requests.Timeout as exc:
            return {
                "error": (
                    f"Deep analytics request timed out after {REQUEST_TIMEOUT}s "
                    f"even after one retry. FastAPI may be reachable, but the "
                    f"analytics computation did not finish. {exc}"
                )
            }
        except requests.ConnectionError as exc2:
            return {"error": f"Deep analytics connection failed after one retry: {exc2}"}
    except requests.Timeout as exc:
        return {
            "error": (
                f"Deep analytics request timed out after {REQUEST_TIMEOUT}s. "
                f"FastAPI is reachable; the analytics computation did not finish in time. {exc}"
            )
        }
    except requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else "HTTP"
        detail = ""
        try:
            payload = exc.response.json() if exc.response is not None else {}
            detail = payload.get("detail", "") if isinstance(payload, dict) else ""
        except Exception:
            pass
        return {
            "error": (
                f"Deep analytics endpoint returned HTTP {status_code}"
                + (f": {detail}" if detail else ".")
            )
        }
    except requests.RequestException as exc:
        return {"error": f"Deep analytics request failed: {exc}"}
    except Exception as exc:
        return {"error": f"Deep analytics client error: {type(exc).__name__}: {exc}"}


@app.callback(
    Output("kpi-grid", "children"),
    Output("secondary-kpi-grid", "children"),
    Output("runtime-badges", "children"),
    Input("filtered-store", "data"),
    Input("health-store", "data"),
    Input("view-mode", "value"),
)
def render_kpis(filtered, health, view):
    if not isinstance(filtered, dict) or filtered.get("error"):
        return (
            [kpi_card("STATUS", "API unavailable", "Start FastAPI on :8000", "danger")],
            [],
            [],
        )

    primary = [
        kpi_card("WORKS", f"{safe_int(filtered.get('total_works')):,}", "current scope"),
        kpi_card("COMPLETED", f"{safe_int(filtered.get('completed_works')):,}", "completed works"),
        kpi_card("OPEN", f"{safe_int(filtered.get('open_works')):,}", "open works"),
        kpi_card("SANCTIONED", inr(filtered.get("sanctioned_amount")), "recorded scope"),
        kpi_card("EXPENDITURE", inr(filtered.get("total_expenditure")), "recorded expenditure"),
        kpi_card("UTILISATION", pct(filtered.get("portfolio_utilization_pct")), "expenditure ÷ sanctioned"),
    ]

    secondary = [
        kpi_card("COMPLETION RATE", pct(filtered.get("completion_rate_pct")), "completed ÷ works"),
        kpi_card("MEAN RISK", f"{safe_float(filtered.get('mean_risk')):.1f}", "composite score / 100"),
        kpi_card("HIGH / CRITICAL", f"{safe_int(filtered.get('high_or_critical')):,}", pct(filtered.get("high_or_critical_rate_pct")), "warning"),
        kpi_card("CRITICAL", f"{safe_int(filtered.get('critical')):,}", "requires verification", "danger"),
        kpi_card("OVERDUE OPEN", f"{safe_int(filtered.get('overdue_open_works')):,}", "timing signal"),
        kpi_card("DUPLICATE CANDIDATES", f"{safe_int(filtered.get('duplicate_candidates')):,}", "similarity signal"),
    ]

    # Keep secondary risk indicators visible on analytical views while the
    # Overview gets a cleaner, citizen-facing hierarchy.
    if view != "Overview":
        secondary.extend([
            kpi_card("MULTI-SIGNAL", f"{safe_int(filtered.get('multi_signal_cases')):,}", "2+ independent signals"),
        ])

    source = health.get("data_source", "unknown") if isinstance(health, dict) else "unknown"
    api_ok = isinstance(health, dict) and not health.get("error")
    badges = [
        html.Div(
            [
                html.Span(className="badge-dot online" if api_ok else "badge-dot offline"),
                html.Span("API", className="badge-brand"),
                html.Span("ONLINE" if api_ok else "OFFLINE", className="badge-status"),
            ],
            className="runtime-badge badge-api" + (" is-online" if api_ok else " is-offline"),
            title="FastAPI health",
        ),
        html.Div(
            [
                html.Span("⬡", className="badge-logo source"),
                html.Span("SOURCE", className="badge-brand"),
                html.Span(str(source).upper(), className="badge-status"),
            ],
            className="runtime-badge badge-source",
            title="Analytical data source",
        ),
        html.Div(
            [
                html.Span("◆", className="badge-logo dash"),
                html.Span("DASH", className="badge-brand"),
                html.Span(str(APP_VERSION).replace("Dash-", ""), className="badge-status"),
            ],
            className="runtime-badge badge-dash",
            title="Dashboard build",
        ),
    ]
    return primary, secondary, badges
@app.callback(
    Output("team-footer-section", "className"),
    Input("filtered-store", "data"),
    Input("analytics-store", "data"),
)
def reveal_team_footer(filtered, analytics):
    """Keep the team footer hidden until real dashboard data has populated.

    This prevents the branded bottom section from flashing before the KPIs,
    charts and queue have loaded — the page reveals top-to-bottom.
    """
    if not isinstance(filtered, dict) or filtered.get("error"):
        return "team-footer team-footer-pending"
    if analytics is None or (isinstance(analytics, dict) and analytics.get("error")):
        return "team-footer team-footer-pending"
    return "team-footer team-footer-ready"


# ============================================================
# VIEW RENDERING
# ============================================================


def analytics_state_view(analytics: Any) -> html.Div:
    """Show a truthful loading/error state without blanking the live KPI layer."""
    if analytics is None:
        return html.Div(
            [
                html.Div("LIVE ANALYTICS", className="section-kicker"),
                html.H2("Building the filtered analytical view…", className="section-title"),
                html.Div(
                    "The KPI scope is live. Deep risk, financial, execution and geography graphics are loading from the same active filter set.",
                    className="section-note",
                ),
            ],
            className="panel analytics-state-panel",
        )

    if isinstance(analytics, dict) and analytics.get("error"):
        return html.Div(
            [
                html.Div("ANALYTICS ENGINE", className="section-kicker"),
                html.H2("Live scope loaded; deep charts need attention", className="section-title"),
                html.Div(str(analytics.get("error")), className="section-note analytics-error-text"),
                html.Div(
                    "FastAPI health, filter options, KPIs and work queue remain independent. Refresh after correcting the analytics endpoint.",
                    className="section-note",
                ),
            ],
            className="panel analytics-state-panel error",
        )

    return html.Div()

def overview_view(works, analytics):
    scope = analytics.get("scope", {}) if isinstance(analytics, dict) else {}
    return html.Div([
        html.Div([
            html.Div([
                html.Div("PORTFOLIO AT A GLANCE", className="overview-kicker"),
                html.Div("Understand the whole picture before opening a work.", className="overview-headline"),
                html.P(
                    "A filtered, evidence-oriented view of MPLADS works: scale first, then project status, financial flow, risk signals and geographic concentration. Every analytical card below uses the same active filter scope.",
                    className="overview-copy",
                ),
                html.Div([
                    html.Div("SCALE", className="overview-chip"),
                    html.Div("MONEY", className="overview-chip"),
                    html.Div("EXECUTION", className="overview-chip"),
                    html.Div("RISK SIGNALS", className="overview-chip"),
                    html.Div("HUMAN REVIEW", className="overview-chip"),
                ], className="overview-meta"),
            ], className="overview-intro"),
            html.Div(
                dcc.Graph(figure=overview_status_figure(scope), config={"displaylogo":False}),
                className="overview-hero-panel",
            ),
        ], className="overview-hero"),

        html.Div([
            html.Div([
                html.Div("PORTFOLIO STATUS", className="overview-section-label"),
                html.Div(
                    dcc.Graph(figure=overview_utilization_figure(scope), config={"displaylogo":False}),
                    className="panel",
                ),
            ]),
            html.Div([
                html.Div("RISK PROFILE", className="overview-section-label"),
                html.Div(
                    dcc.Graph(figure=overview_risk_spectrum_figure(analytics.get("risk_final")), config={"displaylogo":False}),
                    className="panel",
                ),
            ]),
        ], className="chart-grid"),

        html.Div(
            [
                kpi_card("MULTI-SIGNAL", f"{safe_int(scope.get('multi_signal_cases')):,}", "2+ independent signals", "accent"),
                kpi_card("OVERDUE OPEN", f"{safe_int(scope.get('overdue_open_works')):,}", "timing signal"),
                kpi_card("DUPLICATE CANDIDATES", f"{safe_int(scope.get('duplicate_candidates')):,}", "similarity signal"),
            ],
            className="overview-signal-grid",
        ),

        html.Div([
            html.Div(
                [html.Div("FINANCIAL + EXECUTION", className="overview-section-label"),
                 html.Div(dcc.Graph(figure=overview_financial_flow_figure(analytics.get("time")), config={"displaylogo":False}), className="panel")]
            ),
            html.Div(
                [html.Div("SIGNAL PROFILE", className="overview-section-label"),
                 html.Div(dcc.Graph(figure=overview_signal_figure(analytics.get("risk_reasons")), config={"displaylogo":False}), className="panel")]
            ),
        ], className="chart-grid"),

        html.Div([
            html.Div(dcc.Graph(figure=overview_state_attention_figure(analytics.get("state")), config={"displaylogo":False}), className="panel"),
            html.Div([
                html.Div("READ THIS VIEW", className="section-kicker"),
                html.Div("From population to review", className="overview-queue-title"),
                html.P(
                    "Start with the six primary KPIs. Project status shows completed versus open works. Utilisation compares recorded expenditure with sanctioned scope. The risk spectrum shows how the composite score is distributed. Signal prevalence counts analytical flags; it does not establish misconduct.",
                    className="section-note",
                ),
                html.Div([
                    html.Div([html.B("1 · Scope"), html.Span(" — what is in the current filter")], className="band-row"),
                    html.Div([html.B("2 · Flow"), html.Span(" — sanctioned → expenditure")], className="band-row"),
                    html.Div([html.B("3 · Execution"), html.Span(" — completion and open work")], className="band-row"),
                    html.Div([html.B("4 · Signals"), html.Span(" — independent analytical flags")], className="band-row"),
                    html.Div([html.B("5 · Review"), html.Span(" — inspect individual works")], className="band-row"),
                ], style={"marginTop":"14px"}),
            ], className="panel", style={"padding":"18px"}),
        ], className="chart-grid"),

        collapsible_queue_panel(
            works,
            table_id="overview-queue-table",
            title="Priority investigation queue",
            subtitle="Review candidates from the active scope · tap to expand",
        ),
    ])

def risk_view(works, analytics):
    return html.Div([
        section_header("DETECT + EXPLAIN", "Risk Intelligence", "Separate the composite score from its underlying rule and machine-learning signals."),
        html.Div([
            html.Div(dcc.Graph(figure=risk_basis_figure(analytics), config={"displaylogo":False}), className="panel"),
            html.Div(dcc.Graph(figure=risk_score_histogram(works), config={"displaylogo":False}), className="panel"),
        ], className="chart-grid"),
        html.Div([
            html.Div(dcc.Graph(figure=risk_reason_rate_figure(analytics.get("risk_reasons")), config={"displaylogo":False}), className="panel"),
            html.Div(dcc.Graph(figure=priority_risk_figure(works), config={"displaylogo":False}), className="panel"),
        ], className="chart-grid"),
        html.Div([
            html.Div(dcc.Graph(figure=completion_risk_figure(works), config={"displaylogo":False}), className="panel"),
            html.Div(dcc.Graph(figure=state_figure(analytics.get("state")), config={"displaylogo":False}), className="panel"),
        ], className="chart-grid"),
        html.Div(dcc.Graph(figure=sector_risk_figure(analytics.get("sector")), config={"displaylogo":False}), className="panel"),
        collapsible_queue_panel(works, table_id="risk-queue-table", title="Current investigation queue", subtitle=f"{len(works):,} displayed records · tap to expand"),
    ])
def finance_view(works, analytics):
    return html.Div([
        section_header("MONEY + EXECUTION", "Financial & Execution Intelligence", "Track sanctioned scope, recorded expenditure, utilisation, completion and execution age without conflating signals with audit conclusions."),
        html.Div([
            html.Div(dcc.Graph(figure=time_figure(analytics.get("time")), config={"displaylogo":False}), className="panel"),
            html.Div(dcc.Graph(figure=time_risk_figure(analytics.get("time")), config={"displaylogo":False}), className="panel"),
        ], className="chart-grid"),
        html.Div([
            html.Div(dcc.Graph(figure=financial_gap_figure(works), config={"displaylogo":False}), className="panel"),
            html.Div(dcc.Graph(figure=utilization_distribution_figure(works), config={"displaylogo":False}), className="panel"),
        ], className="chart-grid"),
        html.Div([
            html.Div(dcc.Graph(figure=execution_age_figure(works), config={"displaylogo":False}), className="panel"),
            html.Div(dcc.Graph(figure=execution_figure(works), config={"displaylogo":False}), className="panel"),
        ], className="chart-grid"),
        html.Div([
            html.Div(dcc.Graph(figure=sector_completion_figure(analytics.get("sector")), config={"displaylogo":False}), className="panel"),
            html.Div(dcc.Graph(figure=sector_figure(analytics.get("sector")), config={"displaylogo":False}), className="panel"),
        ], className="chart-grid"),
    ])
def geography_view(works, analytics):
    district_data=analytics.get("district")
    district_frame=to_polars(district_data)
    if district_frame.is_empty() or not {"ida","high_or_critical_rate_pct"}.issubset(district_frame.columns):
        district_panel=empty_panel("District analytics unavailable for the active scope.")
    else:
        district_frame=district_frame.with_columns(pl.col("high_or_critical_rate_pct").cast(pl.Float64,strict=False).fill_null(0)).sort("high_or_critical_rate_pct",descending=True).head(30).sort("high_or_critical_rate_pct")
        rows=district_frame.to_dicts(); fig=base_figure("District / IDA high-critical signal rate",540)
        fig.add_trace(go.Bar(x=[safe_float(r.get("high_or_critical_rate_pct")) for r in rows],y=[str(r.get("ida","Unknown")) for r in rows],orientation="h",customdata=[[safe_int(r.get("works")),safe_int(r.get("completed_works"))] for r in rows],hovertemplate="%{y}<br>High/Critical: %{x:.1f}%<br>Works: %{customdata[0]:,}<br>Completed: %{customdata[1]:,}<extra></extra>"))
        fig.update_layout(xaxis_title="Rate (%)",yaxis_title="")
        district_panel=html.Div(dcc.Graph(figure=fig,config={"displaylogo":False}),className="panel")
    return html.Div([
        section_header("WHERE", "Geographic Intelligence", "Compare descriptive geographic concentration, exposure and completion patterns. Map points are shown only when valid source coordinates exist."),
        html.Div([
            html.Div(dcc.Graph(figure=state_figure(analytics.get("state")),config={"displaylogo":False}),className="panel"),
            html.Div(dcc.Graph(figure=state_exposure_figure(analytics.get("state")),config={"displaylogo":False}),className="panel"),
        ],className="chart-grid"),
        html.Div([
            html.Div(dcc.Graph(figure=state_completion_figure(analytics.get("state")),config={"displaylogo":False}),className="panel"),
            district_panel,
        ],className="chart-grid"),
        html.Div([
            html.Div("WORK-LEVEL RISK MAP",className="section-kicker"),
            html.Div("The map uses the filtered priority queue and valid latitude/longitude fields. It is descriptive, not a geographic proof of anomaly.",className="section-note"),
            dcc.Graph(figure=map_figure_from_works(works),config={"displaylogo":False}),
        ],className="panel"),
    ])
def map_figure_from_works(works):
    frame = to_polars(works)
    if frame.is_empty():
        return empty_figure("No work records available for mapping", 560)

    lat_col = (
        "lat"
        if "lat" in frame.columns
        else "latitude"
        if "latitude" in frame.columns
        else None
    )
    lon_col = (
        "lon"
        if "lon" in frame.columns
        else "longitude"
        if "longitude" in frame.columns
        else None
    )

    if not lat_col or not lon_col:
        return empty_figure(
            "No latitude/longitude fields in the current API response",
            560,
        )

    points = []
    for row in frame.to_dicts():
        lat = safe_float(row.get(lat_col), float("nan"))
        lon = safe_float(row.get(lon_col), float("nan"))
        if (
            math.isfinite(lat)
            and math.isfinite(lon)
            and -90 <= lat <= 90
            and -180 <= lon <= 180
        ):
            points.append((lat, lon, row))

    if not points:
        return empty_figure("No valid coordinates available", 560)

    fig = go.Figure(
        go.Scattermapbox(
            lat=[p[0] for p in points],
            lon=[p[1] for p in points],
            mode="markers",
            marker={"size": 9},
            customdata=[
                [
                    p[2].get("work_uid"),
                    p[2].get("risk_category"),
                    safe_float(p[2].get("final_risk_score")),
                ]
                for p in points
            ],
            hovertemplate=(
                "Work %{customdata[0]}"
                "<br>Band: %{customdata[1]}"
                "<br>Risk: %{customdata[2]:.1f}"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        mapbox={"style": "open-street-map", "center": {"lat": 22.5, "lon": 79.0}, "zoom": 3.8},
        height=560,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def explorer_view(works):
    total = len(works)
    return html.Div(
        [
            section_header(
                "EXPLAIN + REVIEW",
                "Work Explorer",
                "Select a row to open the work-level evidence profile.",
            ),
            html.Div(
                [
                    html.Div(
                        f"{total:,} records returned by the active API query.",
                        className="queue-meta",
                    ),
                    collapsible_queue_panel(works, table_id="explorer-queue-table", title="Work explorer queue", subtitle="Select a row to inspect · tap to expand", selectable=True),
                ],
                className="panel",
            ),
        ]
    )



def methodology_view(health):
    """Trust / methodology surface with lucid multi-colour flow."""
    health = health if isinstance(health, dict) else {}
    status = [
        {"Item": "API version", "Value": str(health.get("version") or "—")},
        {"Item": "Data source", "Value": str(health.get("data_source") or "csv")},
        {"Item": "Works loaded", "Value": f"{safe_int(health.get('works_loaded')):,}"},
        {"Item": "Data as-of", "Value": str(health.get("data_as_of") or "—")},
        {"Item": "Dashboard", "Value": APP_VERSION},
        {"Item": "Team", "Value": "Fresh Minds · 120613"},
    ]

    steps = [
        ("01", "Official source data", "MPLADS works, sanctions, expenditure and status fields enter the analytical store.", "#5B9FD4"),
        ("02", "Validation + features", "Dates, amounts and peers are cleaned; transparent features are engineered for review.", "#6BCB77"),
        ("03", "Rule signals", "Over-sanction, overdue, stalled, duplicate, cost/duration outliers and date integrity flags.", "#F5C56B"),
        ("04", "Isolation Forest", "Population-relative anomaly percentile — abnormality, not a fraud probability.", "#C792EA"),
        ("05", "Composite risk", "ML + financial + execution + similarity + integrity layers fused into a 0–100 score.", "#FF8B7B"),
        ("06", "Priority + exposure", "Score is tempered by financial exposure and evidence support for triage.", "#8FD0FF"),
        ("07", "Human review", "Investigators verify signals. The dashboard never adjudicates misconduct.", "#9EE6A0"),
    ]

    flow_children = []
    for i, (num, title, note, color) in enumerate(steps):
        flow_children.append(
            html.Div(
                [
                    html.Div(num, className="flow-num", style={"color": color, "borderColor": color}),
                    html.Div(
                        [
                            html.Div(title, className="flow-title"),
                            html.Div(note, className="flow-note"),
                        ],
                        className="flow-body",
                    ),
                ],
                className="flow-step",
            )
        )
        if i < len(steps) - 1:
            flow_children.append(html.Div("↓", className="flow-arrow", style={"color": color}))

    bands = [
        ("LOW", "0–25", "#3D8B6E"),
        ("MEDIUM", ">25–50", "#C9A227"),
        ("HIGH", ">50–75", "#D97B2D"),
        ("CRITICAL", ">75–100", "#D64545"),
    ]

    return html.Div(
        [
            section_header(
                "TRUST",
                "Methodology & Guardrails",
                "Explainable hierarchy from official data to human review. Signals support verification — they do not replace it.",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("ANALYTICAL HIERARCHY", className="section-kicker"),
                            html.Div("From population evidence to investigation priority", className="overview-queue-title"),
                            html.Div(flow_children, className="method-flow-visual"),
                        ],
                        className="panel method-panel",
                    ),
                    html.Div(
                        [
                            html.Div("RISK BANDS", className="section-kicker"),
                            html.Div("Composite score ranges used for triage", className="overview-queue-title"),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div(name, className="band-name", style={"color": color}),
                                            html.Div(rng, className="band-range"),
                                            html.Div(className="band-bar", style={"background": color}),
                                        ],
                                        className="band-card",
                                    )
                                    for name, rng, color in bands
                                ],
                                className="band-grid",
                            ),
                            html.Div("SCORE SEMANTICS", className="section-kicker", style={"marginTop": "22px"}),
                            html.P(
                                "ML anomaly percentile ranks how unusual a work looks relative to peers. "
                                "It is not a probability of fraud, corruption, or legal non-compliance.",
                                className="section-note",
                            ),
                            html.P(
                                "Rule signals (over-sanction, overdue, stalled, duplicate, outliers, bad dates) "
                                "are transparent boolean flags. Composite risk blends them with financial and "
                                "execution context for prioritisation only.",
                                className="section-note",
                            ),
                            html.Div("GUARDRAILS", className="section-kicker", style={"marginTop": "18px"}),
                            html.Ul(
                                [
                                    html.Li("No fabricated map coordinates when source lat/long is absent."),
                                    html.Li("Queue is capped for usability; analytics use the full filtered population."),
                                    html.Li("Human review remains the final decision layer."),
                                    html.Li("Filters synchronise KPIs, charts and the investigation queue."),
                                ],
                                className="guard-list",
                            ),
                        ],
                        className="panel method-panel",
                    ),
                ],
                className="overview-two-col",
            ),
            html.Div(
                [
                    html.Div("RUNTIME STATUS", className="section-kicker"),
                    dash_table.DataTable(
                        data=status,
                        columns=[{"name": "Item", "id": "Item"}, {"name": "Value", "id": "Value"}],
                        style_table={"overflowX": "auto"},
                        style_header={
                            "backgroundColor": "#151D26",
                            "color": "#A7B5C2",
                            "fontWeight": "700",
                            "border": "1px solid #24303A",
                        },
                        style_cell={
                            "backgroundColor": "#0F151C",
                            "color": "#DDE6ED",
                            "border": "1px solid #1C2630",
                            "fontFamily": FONT_BODY,
                            "fontSize": "13px",
                            "padding": "10px 12px",
                        },
                    ),
                ],
                className="panel",
                style={"marginTop": "16px"},
            ),
        ],
        className="deep-view",
    )



# ============================================================
# DEEP DASHBOARD EXPERIENCE — 35+ FILTER-AWARE VISUALS
# ============================================================
# The original analytical functions remain above. This extension
# deliberately adds a second, richer visualization layer rather than
# deleting or simplifying existing functionality.
#
# All deep figures are fed from /api/v1/deep-analytics. Therefore:
#   • KPI scope = filtered population
#   • aggregated charts = filtered population
#   • queue = top 500 review records
#   • scatter samples = deterministic seeded subset when needed
#   • no chart silently treats the 500-row queue as the whole universe
# ============================================================

# Risk colours are categorical and aligned with the project's existing
# RISK_RANGES: LOW 0–25, MEDIUM >25–50, HIGH >50–75, CRITICAL >75–100.
DEEP_RISK_COLORS = {
    "LOW": "#2BB673",
    "MEDIUM": "#F2C94C",
    "HIGH": "#F2994A",
    "CRITICAL": "#E55353",
}
DEEP_RISK_NEUTRAL = "#26343E"
DEEP_RISK_BORDER = "#8AA6B5"

DEEP_SIGNAL_COLORS = {
    "financial": "#8FD0FF",
    "timing": "#F5C56B",
    "execution": "#8CCB9B",
    "similarity": "#C39CFF",
    "data_quality": "#E58E8E",
}

DEEP_PALETTE = [
    "#8FD0FF",
    "#5B6FEF",
    "#14B89A",
    "#F5C56B",
    "#C39CFF",
    "#E58E8E",
    "#8CCB9B",
    "#D6A76D",
    "#71B7D6",
    "#A6B6C5",
]

DEEP_GRAPH_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


def deep_base_figure(title: str, height: int = 420) -> go.Figure:
    """High-density dark analytical canvas with consistent MPLADS styling."""
    fig = go.Figure()
    fig.update_layout(
        title={
            "text": title,
            "font": {"family": FONT_HEAD, "size": 17, "color": "#EEF4F8"},
            "x": 0.02,
            "xanchor": "left",
            "y": 0.96,
            "yanchor": "top",
        },
        height=height,
        margin={"l": 28, "r": 26, "t": 62, "b": 45},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": FONT_BODY, "size": 11, "color": "#DDE7EE"},
        hoverlabel={
            "bgcolor": "#111A23",
            "bordercolor": "#355267",
            "font": {"family": FONT_BODY, "size": 11, "color": "#EEF4F8"},
        },
        legend={
            "orientation": "h",
            "y": 1.01,
            "x": 1,
            "xanchor": "right",
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"size": 9},
        },
        hovermode="closest",
        transition={"duration": 180},
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#1E2A35",
        gridwidth=1,
        zeroline=False,
        linecolor="#2B3A47",
        tickfont={"color": "#8697A5", "size": 9},
        title_font={"color": "#9CAEBB", "size": 10},
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#1E2A35",
        gridwidth=1,
        zeroline=False,
        linecolor="#2B3A47",
        tickfont={"color": "#8697A5", "size": 9},
        title_font={"color": "#9CAEBB", "size": 10},
    )
    return fig


def deep_graph_panel(
    kicker: str,
    fig: go.Figure,
    note: str = "",
    class_name: str = "panel deep-panel",
) -> html.Div:
    # Force readable height so Plotly never collapses to a dwarfed strip.
    if getattr(fig.layout, "height", None) in (None, 0):
        fig.update_layout(height=420)
    elif int(fig.layout.height or 0) < 360:
        fig.update_layout(height=max(380, int(fig.layout.height or 0)))
    try:
        chart_title = str(fig.layout.title.text or "").strip()
    except Exception:
        chart_title = ""
    chart_label = chart_title or kicker.upper()
    chart_key = re.sub(r"[^a-z0-9]+", "-", f"{kicker}-{chart_label}".casefold()).strip("-")[:120] or "chart"
    children = [
        html.Div(
            [
                html.Div(kicker.upper(), className="deep-kicker"),
            ],
            className="deep-panel-toolbar",
        ),
        dcc.Graph(
            figure=fig,
            config=DEEP_GRAPH_CONFIG,
            style={"height": f"{int(fig.layout.height or 420)}px", "minHeight": "360px"},
        ),
    ]
    if note:
        children.append(html.Div(note, className="deep-note"))
    return html.Div(children, className=class_name)


def deep_records(analytics: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = analytics.get(key) if isinstance(analytics, dict) else None
    return value if isinstance(value, list) else []


def _deep_frame(data: Any) -> pl.DataFrame:
    return to_polars(data)


def _deep_rows(data: Any) -> list[dict[str, Any]]:
    frame = _deep_frame(data)
    return frame.to_dicts() if not frame.is_empty() else []


def _deep_sort_rows(rows: list[dict[str, Any]], key: str, reverse: bool = True, limit: int = 20) -> list[dict[str, Any]]:
    rows = [row for row in rows if row.get(key) is not None]
    rows.sort(key=lambda row: safe_float(row.get(key), float("-inf")), reverse=reverse)
    return rows[:limit]


def deep_status_share_figure(data: Any) -> go.Figure:
    rows = deep_records(data if isinstance(data, dict) else {}, "status_distribution")
    if not rows:
        return empty_figure("Work-status distribution unavailable", 340)
    fig = deep_base_figure("Work status across the filtered population", 360)
    labels = [str(row.get("work_status") or "Unknown") for row in rows]
    values = [safe_int(row.get("works")) for row in rows]
    fig.add_trace(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=DEEP_PALETTE[0],
            customdata=[[safe_float(row.get("share_pct"))] for row in rows],
            hovertemplate="%{y}<br>Works: %{x:,}<br>Share: %{customdata[0]:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_title="Works", yaxis_title="")
    return fig


def deep_status_risk_heatmap_figure(data: Any) -> go.Figure:
    rows = data if isinstance(data, list) else []
    if not rows:
        return empty_figure("Status × risk matrix unavailable", 400)
    df = pl.DataFrame(rows)
    if not {"work_status", "risk_category", "count"}.issubset(df.columns):
        return empty_figure("Status × risk matrix unavailable", 400)
    statuses = [str(value) for value in df.get_column("work_status").unique().to_list()]
    statuses = sorted(statuses, key=str.casefold)
    matrix = []
    for status in statuses:
        lookup = {
            str(row.get("risk_category")): safe_int(row.get("count"))
            for row in df.filter(pl.col("work_status") == status).to_dicts()
        }
        matrix.append([lookup.get(band, 0) for band in RISK_ORDER])
    fig = deep_base_figure("Risk-band composition by recorded work status", max(390, 65 + len(statuses) * 28))
    fig.add_trace(
        go.Heatmap(
            z=matrix,
            x=RISK_ORDER,
            y=statuses,
            colorscale=[[0, "#0D141B"], [0.25, "#23394A"], [0.60, "#51738D"], [1, "#8FD0FF"]],
            hovertemplate="Status: %{y}<br>Risk: %{x}<br>Works: %{z:,}<extra></extra>",
            colorbar={"title": "Works", "tickfont": {"size": 9}},
        )
    )
    fig.update_layout(xaxis_title="Risk band", yaxis_title="Work status")
    return fig


def deep_risk_financial_figure(data: Any) -> go.Figure:
    rows = data if isinstance(data, list) else []
    if not rows:
        return empty_figure("Risk-band financial exposure unavailable", 420)
    order = {band: idx for idx, band in enumerate(RISK_ORDER)}
    rows = sorted(rows, key=lambda row: order.get(str(row.get("risk_category")), 99))
    fig = deep_base_figure("Financial exposure by final risk band", 410)
    for band in RISK_ORDER:
        row = next((item for item in rows if str(item.get("risk_category")) == band), None)
        amount = safe_float(row.get("sanctioned_amount")) if row else 0.0
        fig.add_trace(
            go.Bar(
                x=[band],
                y=[amount],
                name=band,
                marker_color=DEEP_RISK_COLORS[band],
                customdata=[[safe_int(row.get("works")) if row else 0, safe_float(row.get("sanction_share_pct")) if row else 0.0]],
                hovertemplate=f"{band}<br>Sanctioned: ₹%{{y:,.0f}}<br>Works: %{{customdata[0]:,}}<br>Financial share: %{{customdata[1]:.1f}}%<extra></extra>",
            )
        )
    fig.update_layout(barmode="group", xaxis_title="Risk band", yaxis_title="Sanctioned amount (₹)")
    return fig


def deep_risk_hist_scope_figure(data: Any) -> go.Figure:
    rows = deep_records(data if isinstance(data, dict) else {}, "distributions")
    # The helper is called with analytics, not the distribution itself.
    return _deep_bucket_bar(analytics=data, key="risk_score", title="Final-risk score distribution · complete filtered scope", x_title="Score band", y_title="Works", height=390)


def _deep_bucket_bar(
    analytics: Any,
    key: str,
    title: str,
    x_title: str,
    y_title: str,
    height: int = 390,
    horizontal: bool = False,
) -> go.Figure:
    rows = deep_records(analytics if isinstance(analytics, dict) else {}, "distributions")
    # This key is intentionally fetched from the nested distribution object.
    nested = analytics.get("distributions", {}).get(key, []) if isinstance(analytics, dict) else []
    if not nested:
        return empty_figure(f"{title} unavailable", height)
    labels = [str(row.get("bucket")) for row in nested]
    values = [safe_int(row.get("count")) for row in nested]
    shares = [safe_float(row.get("share_pct")) for row in nested]
    fig = deep_base_figure(title, height)
    if horizontal:
        fig.add_trace(
            go.Bar(
                x=values,
                y=labels,
                orientation="h",
                marker_color=DEEP_PALETTE[0],
                customdata=[[shares[i]] for i in range(len(values))],
                hovertemplate="%{y}<br>Works: %{x:,}<br>Share: %{customdata[0]:.1f}%<extra></extra>",
            )
        )
        fig.update_layout(yaxis={"categoryorder": "array", "categoryarray": labels})
    else:
        fig.add_trace(
            go.Bar(
                x=labels,
                y=values,
                marker_color=DEEP_PALETTE[0],
                customdata=[[shares[i]] for i in range(len(values))],
                hovertemplate="%{x}<br>Works: %{y:,}<br>Share: %{customdata[0]:.1f}%<extra></extra>",
            )
        )
    fig.update_layout(xaxis_title=x_title, yaxis_title=y_title)
    return fig


def deep_risk_utilization_figure(data: Any) -> go.Figure:
    return _deep_scatter_figure(
        data,
        "utilization_pct",
        "final_risk_score",
        "Risk score × utilisation · filtered work sample",
        "Recorded expenditure / sanction (%)",
        "Final risk score (0–100)",
        "Utilisation",
        "Risk",
        420,
    )


def deep_risk_sanction_figure(data: Any) -> go.Figure:
    return _deep_scatter_figure(
        data,
        "sanction_amount",
        "final_risk_score",
        "Risk score × sanctioned value · filtered work sample",
        "Sanction amount (₹)",
        "Final risk score (0–100)",
        "Sanction",
        "Risk",
        420,
        log_x=True,
    )


def deep_risk_age_figure(data: Any) -> go.Figure:
    return _deep_scatter_figure(
        data,
        "days_open_since_sanction",
        "final_risk_score",
        "Risk score × open-work age · filtered work sample",
        "Days since sanction",
        "Final risk score (0–100)",
        "Open age",
        "Risk",
        420,
    )


def deep_priority_exposure_figure(data: Any) -> go.Figure:
    return _deep_scatter_figure(
        data,
        "financial_exposure_percentile",
        "priority_score",
        "Investigation priority × financial exposure percentile",
        "Financial exposure percentile",
        "Priority score",
        "Exposure",
        "Priority",
        420,
    )


def deep_cost_risk_figure(data: Any) -> go.Figure:
    return _deep_scatter_figure(
        data,
        "cost_robust_z",
        "final_risk_score",
        "Cost peer deviation × final risk",
        "Robust cost z-score",
        "Final risk score",
        "Cost deviation",
        "Risk",
        420,
    )


def deep_duration_risk_figure(data: Any) -> go.Figure:
    return _deep_scatter_figure(
        data,
        "duration_robust_z",
        "final_risk_score",
        "Duration peer deviation × final risk",
        "Robust duration z-score",
        "Final risk score",
        "Duration deviation",
        "Risk",
        420,
    )


def deep_confidence_priority_figure(data: Any) -> go.Figure:
    return _deep_scatter_figure(
        data,
        "confidence_score",
        "priority_score",
        "Evidence confidence × investigation priority",
        "Confidence score",
        "Priority score",
        "Confidence",
        "Priority",
        420,
    )


def _deep_scatter_figure(
    analytics: Any,
    x_col: str,
    y_col: str,
    title: str,
    x_title: str,
    y_title: str,
    x_label: str,
    y_label: str,
    height: int = 420,
    log_x: bool = False,
) -> go.Figure:
    sample = deep_records(analytics if isinstance(analytics, dict) else {}, "scatter_sample")
    valid = []
    for row in sample:
        x = safe_float(row.get(x_col), float("nan"))
        y = safe_float(row.get(y_col), float("nan"))
        if math.isfinite(x) and math.isfinite(y):
            if log_x and x <= 0:
                continue
            valid.append(row)
    if not valid:
        return empty_figure(f"{title} unavailable", height)
    fig = deep_base_figure(title, height)
    risk_values = [str(row.get("risk_category") or "UNKNOWN") for row in valid]
    symbols = {"LOW": "circle", "MEDIUM": "circle", "HIGH": "diamond", "CRITICAL": "x"}
    for band in RISK_ORDER:
        subset = [row for row in valid if str(row.get("risk_category")) == band]
        if not subset:
            continue
        customdata = [
            [
                row.get("work_uid"),
                row.get("state"),
                row.get("work_category"),
                safe_float(row.get("final_risk_score")),
                safe_float(row.get("priority_score")),
                safe_float(row.get("utilization_pct")),
            ]
            for row in subset
        ]
        fig.add_trace(
            go.Scatter(
                x=[safe_float(row.get(x_col)) for row in subset],
                y=[safe_float(row.get(y_col)) for row in subset],
                mode="markers",
                name=band,
                marker={
                    "size": 8 if band not in ("CRITICAL", "HIGH") else 9,
                    "opacity": 0.72,
                    "color": DEEP_RISK_COLORS[band],
                    "symbol": symbols[band],
                    "line": {"width": 0.6, "color": "#071017"},
                },
                customdata=customdata,
                hovertemplate=(
                    "Work %{customdata[0]}<br>State: %{customdata[1]}<br>Category: %{customdata[2]}"
                    "<br>Risk: %{customdata[3]:.1f}<br>Priority: %{customdata[4]:.1f}"
                    f"<br>Utilisation: %{{customdata[5]:.1f}}%<br>{x_label}: %{{x:,.2f}}"
                    f"<br>{y_label}: %{{y:,.2f}}<extra></extra>"
                ),
            )
        )
    fig.update_layout(xaxis_title=x_title, yaxis_title=y_title)
    if log_x:
        fig.update_xaxes(type="log")
    return fig
    fig.update_layout(xaxis_title=x_title, yaxis_title=y_title)
    if log_x:
        fig.update_xaxes(type="log")
    return fig


def _deep_group_bar(
    rows: list[dict[str, Any]],
    label_col: str,
    value_col: str,
    title: str,
    x_title: str,
    color: str | None = None,
    limit: int = 15,
    height: int = 450,
    percent: bool = False,
) -> go.Figure:
    if not rows or label_col not in rows[0] or value_col not in rows[0]:
        return empty_figure(f"{title} unavailable", height)
    clean = [row for row in rows if row.get(value_col) is not None]
    clean = _deep_sort_rows(clean, value_col, True, limit)
    clean.reverse()
    if not clean:
        return empty_figure(f"{title} unavailable", height)
    values = [safe_float(row.get(value_col)) for row in clean]
    labels = [str(row.get(label_col) or "Unknown")[:28] for row in clean]
    fig = deep_base_figure(title, height)
    fig.add_trace(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=color or DEEP_PALETTE[0],
            customdata=[[safe_int(row.get("works")), safe_float(row.get("sanctioned_amount", row.get("high_critical_sanction")))] for row in clean],
            hovertemplate=(
                "%{y}<br>%{x:.1f}%" if percent else "%{y}<br>%{x:,.0f}"
            )
            + "<br>Works: %{customdata[0]:,}<br>Sanctioned: ₹%{customdata[1]:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title=x_title, yaxis_title="", yaxis={"categoryorder": "array", "categoryarray": labels})
    return fig


def deep_sector_completion_figure(data: Any) -> go.Figure:
    return _deep_group_bar(deep_records(data if isinstance(data, dict) else {}, "sector"), "work_category", "completion_rate_pct", "Completion rate by work category", "Completion rate (%)", DEEP_PALETTE[2], 15, 460, True)


def deep_sector_risk_figure(data: Any) -> go.Figure:
    return _deep_group_bar(deep_records(data if isinstance(data, dict) else {}, "sector"), "work_category", "high_or_critical_rate_pct", "High / critical signal rate by work category", "High / critical rate (%)", DEEP_RISK_COLORS["HIGH"], 15, 460, True)


def deep_sector_utilization_figure(data: Any) -> go.Figure:
    return _deep_group_bar(deep_records(data if isinstance(data, dict) else {}, "sector"), "work_category", "utilization_pct", "Utilisation by work category", "Expenditure / sanction (%)", DEEP_PALETTE[0], 15, 460, True)


def deep_sector_portfolio_figure(data: Any) -> go.Figure:
    return _deep_group_bar(deep_records(data if isinstance(data, dict) else {}, "sector"), "work_category", "sanctioned_amount", "Sanctioned financial portfolio by category", "Sanctioned amount (₹)", DEEP_PALETTE[4], 15, 460)


def deep_sector_priority_figure(data: Any) -> go.Figure:
    return _deep_group_bar(deep_records(data if isinstance(data, dict) else {}, "sector"), "work_category", "mean_priority", "Mean investigation priority by work category", "Mean priority score", DEEP_PALETTE[1], 15, 460)


def deep_sector_risk_mean_figure(data: Any) -> go.Figure:
    return _deep_group_bar(deep_records(data if isinstance(data, dict) else {}, "sector"), "work_category", "mean_risk", "Mean final risk score by work category", "Mean final risk score", DEEP_PALETTE[5], 15, 460)


def deep_state_risk_rate_figure(data: Any) -> go.Figure:
    return _deep_group_bar(deep_records(data if isinstance(data, dict) else {}, "state"), "state", "high_or_critical_rate_pct", "High / critical signal rate by state", "Rate (%)", DEEP_RISK_COLORS["HIGH"], 20, 550, True)


def deep_state_completion_rate_figure(data: Any) -> go.Figure:
    return _deep_group_bar(deep_records(data if isinstance(data, dict) else {}, "state"), "state", "completion_rate_pct", "Completion rate by state", "Completion rate (%)", DEEP_PALETTE[2], 20, 550, True)


def deep_state_exposure_amount_figure(data: Any) -> go.Figure:
    return _deep_group_bar(deep_records(data if isinstance(data, dict) else {}, "state"), "state", "sanctioned_amount", "Sanctioned financial exposure by state", "Sanctioned amount (₹)", DEEP_PALETTE[0], 20, 550)


def deep_state_overdue_rate_figure(data: Any) -> go.Figure:
    return _deep_group_bar(deep_records(data if isinstance(data, dict) else {}, "state"), "state", "overdue_rate_pct", "Overdue-open rate by state", "Overdue rate (%)", DEEP_PALETTE[3], 20, 550, True)


def deep_state_duplicate_rate_figure(data: Any) -> go.Figure:
    return _deep_group_bar(deep_records(data if isinstance(data, dict) else {}, "state"), "state", "duplicate_rate_pct", "Potential duplicate-candidate rate by state", "Candidate rate (%)", DEEP_PALETTE[4], 20, 550, True)


def deep_state_priority_figure(data: Any) -> go.Figure:
    return _deep_group_bar(deep_records(data if isinstance(data, dict) else {}, "state"), "state", "mean_priority", "Mean investigation priority by state", "Mean priority score", DEEP_PALETTE[1], 20, 550)


def deep_state_signal_profile_figure(data: Any) -> go.Figure:
    rows = data if isinstance(data, list) else []
    if not rows:
        return empty_figure("State signal profile unavailable", 570)
    df = pl.DataFrame(rows)
    if not {"state", "signal", "count"}.issubset(df.columns):
        return empty_figure("State signal profile unavailable", 570)
    top_states = (
        df.group_by("state")
        .agg(pl.col("count").sum().alias("signal_total"))
        .sort("signal_total", descending=True)
        .head(12)
        .get_column("state")
        .to_list()
    )
    fig = deep_base_figure("Risk-signal volume by state · top signal concentration", 570)
    for idx, signal in enumerate(sorted({str(value) for value in df.get_column("signal").to_list()})):
        subset = df.filter(pl.col("signal") == signal)
        lookup = {str(row.get("state")): safe_int(row.get("count")) for row in subset.to_dicts()}
        fig.add_trace(
            go.Bar(
                x=[str(state) for state in top_states],
                y=[lookup.get(str(state), 0) for state in top_states],
                name=signal,
                marker_color=DEEP_PALETTE[idx % len(DEEP_PALETTE)],
                hovertemplate=f"{signal}<br>%{{x}}<br>%{{y:,}} flagged works<extra></extra>",
            )
        )
    fig.update_layout(barmode="stack", xaxis_title="State", yaxis_title="Flagged works", xaxis_tickangle=-35)
    return fig


def deep_time_performance_figure(data: Any) -> go.Figure:
    rows = data if isinstance(data, list) else []
    if not rows:
        return empty_figure("Time performance unavailable", 430)
    rows = sorted(rows, key=lambda row: str(row.get("analysis_year")))
    fig = deep_base_figure("Completion, utilisation and signal rate across sanction years", 430)
    x = [str(row.get("analysis_year")) for row in rows]
    fig.add_trace(go.Scatter(x=x, y=[safe_float(row.get("completion_rate_pct")) for row in rows], mode="lines+markers", name="Completion rate", line={"width": 3, "color": DEEP_PALETTE[2]}, hovertemplate="Year %{x}<br>Completion: %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=[safe_float(row.get("utilization_pct")) for row in rows], mode="lines+markers", name="Utilisation", line={"width": 3, "color": DEEP_PALETTE[0]}, hovertemplate="Year %{x}<br>Utilisation: %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=[safe_float(row.get("high_or_critical_rate_pct")) for row in rows], mode="lines+markers", name="High/Critical rate", line={"width": 3, "color": DEEP_RISK_COLORS["HIGH"]}, hovertemplate="Year %{x}<br>High/Critical: %{y:.1f}%<extra></extra>"))
    fig.update_layout(xaxis_title="Sanction year", yaxis_title="Rate (%)", yaxis={"range": [0, 100]}, hovermode="x unified")
    return fig


def deep_time_risk_figure(data: Any) -> go.Figure:
    rows = data if isinstance(data, list) else []
    rows = sorted(rows, key=lambda row: str(row.get("analysis_year")))
    if not rows:
        return empty_figure("Risk trend unavailable", 410)
    fig = deep_base_figure("Mean and median final risk across sanction years", 410)
    x = [str(row.get("analysis_year")) for row in rows]
    fig.add_trace(go.Scatter(x=x, y=[safe_float(row.get("mean_risk")) for row in rows], mode="lines+markers", name="Mean risk", line={"width": 3, "color": DEEP_PALETTE[1]}, hovertemplate="Year %{x}<br>Mean risk: %{y:.1f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=[safe_float(row.get("median_risk")) for row in rows], mode="lines+markers", name="Median risk", line={"width": 3, "color": DEEP_PALETTE[5]}, hovertemplate="Year %{x}<br>Median risk: %{y:.1f}<extra></extra>"))
    fig.update_layout(xaxis_title="Sanction year", yaxis_title="Risk score (0–100)", yaxis={"range": [0, 100]}, hovermode="x unified")
    return fig


def deep_time_volume_figure(data: Any) -> go.Figure:
    rows = data if isinstance(data, list) else []
    rows = sorted(rows, key=lambda row: str(row.get("analysis_year")))
    if not rows:
        return empty_figure("Yearly volume unavailable", 400)
    fig = deep_base_figure("Work volume and sanctioned value across sanction years", 400)
    x = [str(row.get("analysis_year")) for row in rows]
    fig.add_trace(go.Bar(x=x, y=[safe_int(row.get("works")) for row in rows], name="Works", marker_color=DEEP_PALETTE[0], hovertemplate="Year %{x}<br>Works: %{y:,}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=[safe_float(row.get("sanctioned_amount")) for row in rows], name="Sanctioned amount", mode="lines+markers", yaxis="y2", line={"width": 3, "color": DEEP_PALETTE[4]}, hovertemplate="Year %{x}<br>Sanctioned: ₹%{y:,.0f}<extra></extra>"))
    fig.update_layout(xaxis_title="Sanction year", yaxis={"title": "Works"}, yaxis2={"title": "Sanctioned amount (₹)", "overlaying": "y", "side": "right", "showgrid": False})
    return fig


def deep_sanction_year_expenditure_figure(data: Any) -> go.Figure:
    rows = data if isinstance(data, list) else []
    rows = sorted(rows, key=lambda row: str(row.get("analysis_year")))
    if not rows:
        return empty_figure("Sanction-year financial flow unavailable", 410)
    fig = deep_base_figure("Sanctioned versus recorded expenditure by sanction year", 410)
    x = [str(row.get("analysis_year")) for row in rows]
    fig.add_trace(go.Bar(x=x, y=[safe_float(row.get("sanctioned_amount")) for row in rows], name="Sanctioned", marker_color=DEEP_PALETTE[0], hovertemplate="Year %{x}<br>Sanctioned: ₹%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Bar(x=x, y=[safe_float(row.get("total_expenditure")) for row in rows], name="Expenditure", marker_color=DEEP_PALETTE[2], hovertemplate="Year %{x}<br>Expenditure: ₹%{y:,.0f}<extra></extra>"))
    fig.update_layout(barmode="group", xaxis_title="Sanction year", yaxis_title="Amount (₹)")
    return fig


def deep_lag_figure(data: Any) -> go.Figure:
    return _deep_bucket_bar(data, "rec_to_sanction", "Recommendation → sanction lag distribution", "Days", "Works", 400)


def deep_completion_duration_figure(data: Any) -> go.Figure:
    return _deep_bucket_bar(data, "sanction_to_complete", "Sanction → completion duration distribution", "Days", "Works", 400)


def deep_expenditure_gap_figure(data: Any) -> go.Figure:
    return _deep_bucket_bar(data, "expenditure_gap", "Days since last recorded expenditure", "Days", "Works", 400)


def deep_financial_band_figure(data: Any) -> go.Figure:
    rows = deep_records(data if isinstance(data, dict) else {}, "financial_bands")
    if not rows:
        return empty_figure("Financial-band analytics unavailable", 460)
    rows = [row for row in rows if safe_int(row.get("works")) > 0]
    fig = deep_base_figure("Portfolio structure by sanctioned-value band", 460)
    fig.add_trace(go.Bar(x=[str(row.get("band")) for row in rows], y=[safe_int(row.get("works")) for row in rows], name="Works", marker_color=DEEP_PALETTE[0], customdata=[[safe_float(row.get("completion_rate_pct")), safe_float(row.get("high_or_critical_rate_pct"))] for row in rows], hovertemplate="Band %{x}<br>Works: %{y:,}<br>Completion: %{customdata[0]:.1f}%<br>High/Critical: %{customdata[1]:.1f}%<extra></extra>"))
    fig.update_layout(xaxis_title="Sanction band", yaxis_title="Works", xaxis_tickangle=-25)
    return fig


def deep_financial_band_utilization_figure(data: Any) -> go.Figure:
    rows = deep_records(data if isinstance(data, dict) else {}, "financial_bands")
    rows = [row for row in rows if safe_int(row.get("works")) > 0]
    if not rows:
        return empty_figure("Financial-band utilisation unavailable", 460)
    fig = deep_base_figure("Utilisation by sanctioned-value band", 460)
    fig.add_trace(go.Bar(x=[str(row.get("band")) for row in rows], y=[safe_float(row.get("utilization_pct")) for row in rows], marker_color=DEEP_PALETTE[2], hovertemplate="Band %{x}<br>Utilisation: %{y:.1f}%<extra></extra>"))
    fig.update_layout(xaxis_title="Sanction band", yaxis_title="Utilisation (%)", xaxis_tickangle=-25)
    return fig


def deep_quality_figure(data: Any) -> go.Figure:
    rows = deep_records(data if isinstance(data, dict) else {}, "quality_distribution")
    if not rows:
        return empty_figure("Data-quality status unavailable", 380)
    fig = deep_base_figure("Data-quality status across the filtered population", 380)
    fig.add_trace(go.Bar(x=[str(row.get("data_quality_status")) for row in rows], y=[safe_int(row.get("works")) for row in rows], marker_color=DEEP_PALETTE[5], customdata=[[safe_float(row.get("share_pct"))] for row in rows], hovertemplate="%{x}<br>Works: %{y:,}<br>Share: %{customdata[0]:.1f}%<extra></extra>"))
    fig.update_layout(xaxis_title="Data-quality status", yaxis_title="Works")
    return fig


def deep_component_profile_figure(data: Any) -> go.Figure:
    rows = data if isinstance(data, list) else []
    clean = [row for row in rows if row.get("mean") is not None]
    if not clean:
        return empty_figure("Risk-component profile unavailable", 440)
    labels = []
    means = []
    medians = []
    for row in clean:
        label = str(row.get("column"))
        labels.append(label.replace("_", " ").title().replace(" Ml ", " ML "))
        means.append(safe_float(row.get("mean")))
        medians.append(safe_float(row.get("median")))
    fig = deep_base_figure("Analytical score profile · mean versus median", 450)
    fig.add_trace(go.Bar(x=means, y=labels, orientation="h", name="Mean", marker_color=DEEP_PALETTE[0], hovertemplate="%{y}<br>Mean: %{x:.1f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=medians, y=labels, mode="markers", name="Median", marker={"size": 10, "color": DEEP_PALETTE[4]}, hovertemplate="%{y}<br>Median: %{x:.1f}<extra></extra>"))
    fig.update_layout(barmode="overlay", xaxis_title="Score / value", yaxis_title="Component")
    return fig


def deep_signal_count_figure(data: Any, key: str, title: str) -> go.Figure:
    return _deep_bucket_bar(data, key, title, "Number of signals", "Works", 390)


def deep_signal_cooccurrence_figure(data: Any) -> go.Figure:
    rows = data if isinstance(data, list) else []
    if not rows:
        return empty_figure("Signal co-occurrence unavailable", 470)
    labels = sorted({str(row.get("signal_a")) for row in rows} | {str(row.get("signal_b")) for row in rows})
    idx = {label: i for i, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for row in rows:
        a = str(row.get("signal_a")); b = str(row.get("signal_b")); count = safe_int(row.get("count"))
        if a in idx and b in idx:
            matrix[idx[a]][idx[b]] = count
            matrix[idx[b]][idx[a]] = count
    fig = deep_base_figure("Risk-signal co-occurrence · exact filtered-scope counts", 500)
    fig.add_trace(go.Heatmap(z=matrix, x=labels, y=labels, colorscale=[[0, "#0B1117"], [0.3, "#213748"], [0.7, "#51728A"], [1, "#8FD0FF"]], hovertemplate="%{y} + %{x}<br>Works: %{z:,}<extra></extra>", colorbar={"title": "Works"}))
    fig.update_layout(xaxis_tickangle=-35, xaxis_title="Signal", yaxis_title="Signal")
    return fig


def deep_category_risk_heatmap_figure(data: Any) -> go.Figure:
    rows = data if isinstance(data, list) else []
    if not rows:
        return empty_figure("Category × risk heatmap unavailable", 560)
    labels = []
    for row in rows:
        category = str(row.get("work_category") or "Unknown")
        if category not in labels:
            labels.append(category)
    matrix = []
    for category in labels:
        lookup = {str(row.get("risk_category")): safe_int(row.get("count")) for row in rows if str(row.get("work_category")) == category}
        matrix.append([lookup.get(band, 0) for band in RISK_ORDER])
    fig = deep_base_figure("Top work categories × final risk bands", max(500, 75 + len(labels) * 24))
    fig.add_trace(go.Heatmap(z=matrix, x=RISK_ORDER, y=labels, colorscale=[[0, "#0D141B"], [0.25, "#253A4B"], [0.65, "#587B95"], [1, "#9EDBFF"]], hovertemplate="Category: %{y}<br>Risk: %{x}<br>Works: %{z:,}<extra></extra>"))
    fig.update_layout(xaxis_title="Risk band", yaxis_title="Work category")
    return fig


def deep_highrisk_state_exposure_figure(data: Any) -> go.Figure:
    rows = deep_records(data if isinstance(data, dict) else {}, "high_critical_state_exposure")
    return _deep_group_bar(rows, "state", "high_critical_sanction", "High / critical sanctioned exposure by state", "High / critical sanctioned amount (₹)", DEEP_RISK_COLORS["HIGH"], 15, 480)


def deep_highrisk_category_exposure_figure(data: Any) -> go.Figure:
    rows = deep_records(data if isinstance(data, dict) else {}, "high_critical_category_exposure")
    return _deep_group_bar(rows, "work_category", "high_critical_sanction", "High / critical sanctioned exposure by category", "High / critical sanctioned amount (₹)", DEEP_RISK_COLORS["HIGH"], 15, 480)


def deep_state_treemap_figure(data: Any) -> go.Figure:
    rows = deep_records(data if isinstance(data, dict) else {}, "state")
    rows = [row for row in rows if safe_float(row.get("sanctioned_amount")) > 0]
    if not rows:
        return empty_figure("State exposure treemap unavailable", 500)
    fig = go.Figure(
        go.Treemap(
            labels=[str(row.get("state") or "Unknown") for row in rows],
            parents=["" for _ in rows],
            values=[safe_float(row.get("sanctioned_amount")) for row in rows],
            text=[f"{safe_int(row.get('works')):,} works" for row in rows],
            marker={"colors": [safe_float(row.get("high_or_critical_rate_pct")) for row in rows], "colorscale": [[0, "#18242E"], [0.45, "#385C70"], [1, "#A66CFF"]], "colorbar": {"title": "High/Critical %"}},
            hovertemplate="%{label}<br>Sanctioned: ₹%{value:,.0f}<br>%{text}<br>High/Critical: %{marker.color:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(title={"text": "State financial concentration", "font": {"family": FONT_HEAD, "size": 17}, "x": 0.02}, height=500, margin={"l": 10, "r": 10, "t": 55, "b": 10}, paper_bgcolor="rgba(0,0,0,0)")
    return fig


def deep_category_treemap_figure(data: Any) -> go.Figure:
    rows = deep_records(data if isinstance(data, dict) else {}, "sector")
    rows = [row for row in rows if safe_float(row.get("sanctioned_amount")) > 0]
    if not rows:
        return empty_figure("Category financial concentration unavailable", 500)
    fig = go.Figure(go.Treemap(labels=[str(row.get("work_category") or "Unknown") for row in rows], parents=["" for _ in rows], values=[safe_float(row.get("sanctioned_amount")) for row in rows], text=[f"{safe_int(row.get('works')):,} works" for row in rows], marker={"colors": [safe_float(row.get("mean_risk")) for row in rows], "colorscale": [[0, "#16222B"], [0.45, "#3A5568"], [1, "#F05A47"]], "colorbar": {"title": "Mean risk"}}, hovertemplate="%{label}<br>Sanctioned: ₹%{value:,.0f}<br>%{text}<br>Mean risk: %{marker.color:.1f}<extra></extra>"))
    fig.update_layout(title={"text": "Work-category financial concentration", "font": {"family": FONT_HEAD, "size": 17}, "x": 0.02}, height=500, margin={"l": 10, "r": 10, "t": 55, "b": 10}, paper_bgcolor="rgba(0,0,0,0)")
    return fig


def deep_risk_basis_decomposition_figure(data: Any) -> go.Figure:
    rows = data if isinstance(data, list) else []
    if not rows:
        return empty_figure("Risk-basis comparison unavailable", 430)
    fig = deep_base_figure("Final, rule-based and ML risk-band composition", 430)
    basis = [
        ("Final composite", "Final", DEEP_PALETTE[0]),
        ("Rule-based", "Rule", DEEP_PALETTE[2]),
        ("ML anomaly", "ML", DEEP_PALETTE[4]),
    ]
    # If caller sends an explicit three-element basis list, preserve it.
    if all(isinstance(item, dict) and "basis" in item for item in rows):
        values = rows
        for item in values:
            fig.add_trace(go.Bar(x=RISK_ORDER, y=[safe_float(item.get(band)) for band in RISK_ORDER], name=str(item.get("basis")), marker_color=DEEP_PALETTE[len(fig.data) % len(DEEP_PALETTE)]))
    else:
        # Fallback to the same input shape used by the legacy analytics object.
        pass
    fig.update_layout(barmode="group", xaxis_title="Risk band", yaxis_title="Share of works (%)", yaxis={"range": [0, 100]})
    return fig


def deep_date_quality_heatmap_figure(data: Any) -> go.Figure:
    rows = deep_records(data if isinstance(data, dict) else {}, "state")
    rows = rows[:20]
    if not rows:
        return empty_figure("State data-quality profile unavailable", 520)
    labels = [str(row.get("state") or "Unknown") for row in rows]
    matrix = [
        [
            safe_float(row.get("bad_date_rate_pct")),
            safe_float(row.get("duplicate_rate_pct")),
            safe_float(row.get("overdue_rate_pct")),
            safe_float(row.get("stalled_rate_pct")),
        ]
        for row in rows
    ]
    fig = deep_base_figure("State operational-signal profile", max(500, 75 + len(labels) * 23))
    fig.add_trace(go.Heatmap(z=matrix, x=["Bad dates", "Duplicate candidates", "Overdue open", "Stalled"], y=labels, colorscale=[[0, "#0C131A"], [0.4, "#2B485C"], [1, "#F5C56B"]], zmin=0, zmax=100, hovertemplate="State: %{y}<br>%{x}: %{z:.1f}%<extra></extra>"))
    fig.update_layout(xaxis_title="Signal rate", yaxis_title="State")
    return fig


def deep_priority_state_figure(data: Any) -> go.Figure:
    return deep_state_priority_figure(data)


def deep_financial_vs_risk_figure(data: Any) -> go.Figure:
    return deep_risk_sanction_figure(data)


def deep_utilization_distribution_figure(data: Any) -> go.Figure:
    return _deep_bucket_bar(data, "utilization", "Utilisation distribution · exact filtered population", "Utilisation band", "Works", 410)


def deep_open_age_distribution_figure(data: Any) -> go.Figure:
    return _deep_bucket_bar(data, "open_age", "Open-work age distribution · exact filtered population", "Open age", "Works", 410)


def deep_confidence_distribution_figure(data: Any) -> go.Figure:
    return _deep_bucket_bar(data, "confidence", "Confidence-score distribution", "Confidence band", "Works", 390)


def deep_anomaly_distribution_figure(data: Any) -> go.Figure:
    return _deep_bucket_bar(data, "risk_score", "Final-risk distribution", "Risk band", "Works", 390)


def deep_model_missing_figure(data: Any) -> go.Figure:
    return _deep_bucket_bar(data, "model_missing", "Model-input missingness distribution", "Missingness band", "Works", 390)


def deep_duplicate_distribution_figure(data: Any) -> go.Figure:
    return _deep_bucket_bar(data, "duplicate_group", "Duplicate-group size distribution", "Group size", "Works", 390)


def deep_anomaly_percentile_figure(data: Any) -> go.Figure:
    return _deep_scored_histogram_from_profiles(data, "ml_anomaly_percentile", "ML anomaly percentile distribution", "Percentile", 390)


def _deep_scored_histogram_from_profiles(analytics: Any, column: str, title: str, x_title: str, height: int = 390) -> go.Figure:
    rows = deep_records(analytics if isinstance(analytics, dict) else {}, "scatter_sample")
    values = [safe_float(row.get(column), float("nan")) for row in rows]
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return empty_figure(f"{title} unavailable", height)
    bins = [i for i in range(0, 101, 10)]
    counts = [0 for _ in range(10)]
    for value in values:
        idx = min(9, max(0, int(value // 10)))
        counts[idx] += 1
    labels = [f"{i}–{i+10}" for i in range(0, 100, 10)]
    fig = deep_base_figure(title + " · sample for browser performance", height)
    fig.add_trace(go.Bar(x=labels, y=counts, marker_color=DEEP_PALETTE[4], hovertemplate="%{x}<br>Sample rows: %{y:,}<extra></extra>"))
    fig.update_layout(xaxis_title=x_title, yaxis_title="Sample rows")
    return fig


def deep_multisignal_quality_figure(analytics: Any) -> go.Figure:
    scope = analytics.get("scope", {}) if isinstance(analytics, dict) else {}
    values = [
        safe_int(scope.get("multi_signal_cases")),
        safe_int(scope.get("duplicate_candidates")),
        safe_int(scope.get("overdue_open_works")),
        safe_int(scope.get("critical")),
    ]
    labels = ["2+ signals", "Duplicate candidates", "Overdue open", "Critical"]
    fig = deep_base_figure("Priority review signal volumes", 370)
    fig.add_trace(go.Bar(x=labels, y=values, marker_color=[DEEP_PALETTE[4], DEEP_PALETTE[4], DEEP_PALETTE[3], DEEP_RISK_COLORS["CRITICAL"]], hovertemplate="%{x}<br>%{y:,} works<extra></extra>"))
    fig.update_layout(xaxis_title="Review signal", yaxis_title="Works")
    return fig


def deep_signal_prevalence_figure(data: Any) -> go.Figure:
    rows = data if isinstance(data, list) else []
    rows = sorted(rows, key=lambda row: safe_float(row.get("rate_pct")))
    if not rows:
        return empty_figure("Signal prevalence unavailable", 420)
    labels = [str(row.get("label") or row.get("signal")) for row in rows]
    values = [safe_float(row.get("rate_pct")) for row in rows]
    fig = deep_base_figure("Signal prevalence · counts and rates, not findings", 430)
    fig.add_trace(go.Bar(x=values, y=labels, orientation="h", marker_color=[DEEP_SIGNAL_COLORS.get(str(row.get("family")), DEEP_PALETTE[0]) for row in rows], customdata=[[safe_int(row.get("count"))] for row in rows], hovertemplate="%{y}<br>Rate: %{x:.1f}%<br>Works: %{customdata[0]:,}<extra></extra>"))
    fig.update_layout(xaxis_title="Share of filtered works (%)", yaxis_title="")
    return fig


def deep_risk_spectrum_figure(data: Any) -> go.Figure:
    rows = data if isinstance(data, list) else []
    counts = {str(row.get("risk_category")): safe_int(row.get("count")) for row in rows}
    total = sum(counts.values())
    if total <= 0:
        return empty_figure("Risk spectrum unavailable", 360)
    fig = deep_base_figure("Final risk spectrum · complete filtered population", 360)
    for band in RISK_ORDER:
        count = counts.get(band, 0)
        fig.add_trace(go.Bar(x=[count], y=["Current scope"], orientation="h", name=band, marker_color=DEEP_RISK_COLORS[band], customdata=[[safe_float(count / total * 100)]], hovertemplate=f"{band}<br>%{{x:,}} works<br>%{{customdata[0]:.1f}}%<extra></extra>"))
    fig.update_layout(barmode="stack", xaxis_title="Works", yaxis_title="", yaxis={"showgrid": False})
    return fig


def deep_risk_basis_chart_figure(analytics: Any) -> go.Figure:
    # Exact percentages by scoring layer, preserving the existing analytics contract.
    basis_specs = [("Final composite", analytics.get("risk_final")), ("Rule-based", analytics.get("risk_rule")), ("ML anomaly", analytics.get("risk_ml"))]
    fig = deep_base_figure("Risk-band composition across scoring layers", 430)
    added = False
    for name, rows in basis_specs:
        if not isinstance(rows, list) or not rows:
            continue
        lookup = {str(row.get("risk_category")): safe_float(row.get("pct")) for row in rows}
        fig.add_trace(go.Bar(x=RISK_ORDER, y=[lookup.get(band, 0.0) for band in RISK_ORDER], name=name, hovertemplate=f"{name}<br>%{{x}}<br>%{{y:.1f}}%<extra></extra>"))
        added = True
    if not added:
        return empty_figure("Risk-basis comparison unavailable", 430)
    fig.update_layout(barmode="group", xaxis_title="Risk band", yaxis_title="Share of works (%)", yaxis={"range": [0, 100]})
    return fig


def deep_queue_summary_card(scope: dict[str, Any]) -> html.Div:
    return html.Div(
        [
            html.Div("FILTER-AWARE ANALYTICAL UNIVERSE", className="deep-kicker"),
            html.Div(f"{safe_int(scope.get('total_works')):,}", className="deep-hero-number"),
            html.Div("works represented by exact aggregated analytics", className="deep-hero-note"),
            html.Div(
                [
                    html.Span(f"{safe_float(scope.get('completion_rate_pct')):.1f}% complete"),
                    html.Span(f"{safe_float(scope.get('portfolio_utilization_pct')):.1f}% utilisation"),
                    html.Span(f"{safe_float(scope.get('high_or_critical_rate_pct')):.1f}% high/critical"),
                ],
                className="deep-stat-strip",
            ),
        ],
        className="deep-scope-card",
    )


def deep_insight_cards(scope: dict[str, Any]) -> html.Div:
    cards = [
        ("COMPLETION", f"{safe_float(scope.get('completion_rate_pct')):.1f}%", "completed works", "accent"),
        ("UTILISATION", f"{safe_float(scope.get('portfolio_utilization_pct')):.1f}%", "recorded expenditure ÷ sanction", "accent"),
        ("HIGH / CRITICAL", f"{safe_int(scope.get('high_or_critical')):,}", f"{safe_float(scope.get('high_or_critical_rate_pct')):.1f}% of scope", "warning"),
        ("CRITICAL", f"{safe_int(scope.get('critical')):,}", "requires verification", "danger"),
        ("OVERDUE OPEN", f"{safe_int(scope.get('overdue_open_works')):,}", "timing signal", "warning"),
        ("MULTI-SIGNAL", f"{safe_int(scope.get('multi_signal_cases')):,}", "2+ independent signals", "accent"),
    ]
    return html.Div([kpi_card(label, value, note, tone) for label, value, note, tone in cards], className="deep-insight-grid")


def _deep_section(title: str, note: str, *children: Any) -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Div("ANALYTICS MODULE", className="deep-section-kicker"),
                    html.H3(title, className="deep-section-title"),
                    html.P(note, className="deep-section-note"),
                ],
                className="deep-section-head",
            ),
            html.Div(list(children), className="deep-chart-grid"),
        ],
        className="deep-section",
    )



def deep_overview_view(works: list[dict[str, Any]], analytics: dict[str, Any]) -> html.Div:
    """Clean executive Overview — ample space, no graph bombardment.

    Dense multi-chart analytics live in Risk / Financial / Geography tabs.
    Overview answers: scale → status → money → risk → where to review next.
    """
    scope = analytics.get("scope", {}) if isinstance(analytics, dict) else {}
    analytics = analytics if isinstance(analytics, dict) else {}

    return html.Div(
        [
            # ---- Hero: match uploaded target style ----
            html.Div(
                [
                    # LEFT — portfolio copy
                    html.Div(
                        [
                            html.Div("PORTFOLIO AT A GLANCE", className="overview-kicker"),
                            html.Div(
                                "Understand the whole picture before opening a work.",
                                className="overview-headline",
                            ),
                            html.P(
                                "A filtered, evidence-oriented view of MPLADS works: scale first, then project status, "
                                "financial flow, risk signals and geographic concentration. Every analytical card below "
                                "uses the same active filter scope.",
                                className="overview-copy",
                            ),
                            html.Div(
                                [
                                    html.Div("SCALE", className="overview-chip"),
                                    html.Div("MONEY", className="overview-chip"),
                                    html.Div("EXECUTION", className="overview-chip"),
                                    html.Div("RISK SIGNALS", className="overview-chip"),
                                    html.Div("HUMAN REVIEW", className="overview-chip"),
                                ],
                                className="overview-meta",
                            ),
                        ],
                        className="overview-intro",
                    ),

                    # RIGHT — pie chart box (label must be first child here)
                    html.Div(
                        [
                            html.Div("PROJECT STATUS", className="overview-panel-kicker"),
                            dcc.Graph(
                                figure=overview_status_figure(scope),
                                config={"displaylogo": False, "responsive": True},
                                style={"height": "300px", "width": "100%"},
                            ),
                        ],
                        className="overview-hero-panel",
                    ),
                ],
                className="overview-hero",
            ),

            # ---- Row 1: Utilisation + Risk spectrum (ample height) ----
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("PORTFOLIO STATUS", className="overview-section-kicker"),
                            html.Div(
                                dcc.Graph(
                                    figure=overview_utilization_figure(scope),
                                    config={"displaylogo": False, "responsive": True},
                                    style={"height": "400px"},
                                ),
                                className="panel overview-chart-panel",
                            ),
                        ],
                        className="overview-chart-cell",
                    ),
                    html.Div(
                        [
                            html.Div("RISK PROFILE", className="overview-section-kicker"),
                            html.Div(
                                dcc.Graph(
                                    figure=overview_risk_spectrum_figure(analytics.get("risk_final")),
                                    config={"displaylogo": False, "responsive": True},
                                    style={"height": "400px"},
                                ),
                                className="panel overview-chart-panel",
                            ),
                        ],
                        className="overview-chart-cell",
                    ),
                ],
                className="overview-two-col",
            ),

            # ---- Signal strip ----
            html.Div(
                [
                    kpi_card(
                        "MULTI-SIGNAL",
                        f"{safe_int(scope.get('multi_signal_cases')):,}",
                        "2+ independent signals",
                        "accent",
                    ),
                    kpi_card(
                        "OVERDUE OPEN",
                        f"{safe_int(scope.get('overdue_open_works')):,}",
                        "timing signal",
                    ),
                    kpi_card(
                        "DUPLICATE CANDIDATES",
                        f"{safe_int(scope.get('duplicate_candidates')):,}",
                        "similarity signal",
                    ),
                ],
                className="overview-signal-grid",
            ),

            # ---- Row 2: Financial flow + Signal prevalence ----
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("FINANCIAL + EXECUTION", className="overview-section-kicker"),
                            html.Div(
                                dcc.Graph(
                                    figure=overview_financial_flow_figure(analytics.get("time")),
                                    config={"displaylogo": False, "responsive": True},
                                    style={"height": "420px"},
                                ),
                                className="panel overview-chart-panel",
                            ),
                        ],
                        className="overview-chart-cell",
                    ),
                    html.Div(
                        [
                            html.Div("SIGNAL PROFILE", className="overview-section-kicker"),
                            html.Div(
                                dcc.Graph(
                                    figure=overview_signal_figure(analytics.get("risk_reasons")),
                                    config={"displaylogo": False, "responsive": True},
                                    style={"height": "420px"},
                                ),
                                className="panel overview-chart-panel",
                            ),
                        ],
                        className="overview-chart-cell",
                    ),
                ],
                className="overview-two-col",
            ),

            # ---- Row 3: State attention + reading guide ----
            html.Div(
                [
                    html.Div(
                        dcc.Graph(
                            figure=overview_state_attention_figure(analytics.get("state")),
                            config={"displaylogo": False, "responsive": True},
                            style={"height": "440px"},
                        ),
                        className="panel overview-chart-panel",
                    ),
                    html.Div(
                        [
                            html.Div("READ THIS VIEW", className="section-kicker"),
                            html.Div("From population to review", className="overview-queue-title"),
                            html.P(
                                "Start with the six primary KPIs. Project status shows completed versus open works. "
                                "Utilisation compares recorded expenditure with sanctioned scope. The risk spectrum "
                                "shows how the composite score is distributed. Signal prevalence counts analytical "
                                "flags; it does not establish misconduct.",
                                className="section-note",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [html.B("1 · Scope"), html.Span(" — what is in the current filter")],
                                        className="band-row",
                                    ),
                                    html.Div(
                                        [html.B("2 · Flow"), html.Span(" — sanctioned → expenditure")],
                                        className="band-row",
                                    ),
                                    html.Div(
                                        [html.B("3 · Execution"), html.Span(" — completion and open work")],
                                        className="band-row",
                                    ),
                                    html.Div(
                                        [html.B("4 · Signals"), html.Span(" — independent analytical flags")],
                                        className="band-row",
                                    ),
                                    html.Div(
                                        [html.B("5 · Review"), html.Span(" — inspect individual works below")],
                                        className="band-row",
                                    ),
                                ],
                                style={"marginTop": "16px"},
                            ),
                            html.P(
                                "Open Risk Intelligence, Financial & Execution, or Geography for denser multi-chart analytics. "
                                "Overview stays deliberate and readable.",
                                className="section-note",
                                style={"marginTop": "14px"},
                            ),
                        ],
                        className="panel overview-guide-panel",
                    ),
                ],
                className="overview-two-col",
            ),

            # ---- Priority queue ----
            collapsible_queue_panel(
                works,
                table_id="overview-queue-table",
                title="Priority investigation queue",
                subtitle="Review candidates from the active scope · tap to expand",
            ),
        ],
        className="deep-view overview-clean",
    )



def deep_risk_view(works: list[dict[str, Any]], analytics: dict[str, Any]) -> html.Div:
    return html.Div(
        [
            section_header("DETECT + EXPLAIN", "Risk Intelligence", "A dense risk-control surface separating composite scores, model layers, transparent flags, relationships and investigation priority."),
            _deep_section(
                "Risk composition",
                "Use the complete filtered population for distributions; use the queue only for work-level review.",
                deep_graph_panel("SCORING LAYERS", deep_risk_basis_chart_figure(analytics), "Final, rule-based and ML anomaly risk bands."),
                deep_graph_panel("RISK SPECTRUM", deep_risk_spectrum_figure(analytics.get("risk_final")), "Exact final risk composition."),
                deep_graph_panel("RISK × UTILISATION", deep_risk_utilization_figure(analytics), "Risk versus recorded expenditure / sanction."),
                deep_graph_panel("RISK × SANCTION", deep_risk_sanction_figure(analytics), "Financial scale versus final risk."),
                deep_graph_panel("RISK × OPEN AGE", deep_risk_age_figure(analytics), "Execution age versus risk."),
                deep_graph_panel("COST × RISK", deep_cost_risk_figure(analytics), "Peer-relative cost deviation versus risk."),
                deep_graph_panel("DURATION × RISK", deep_duration_risk_figure(analytics), "Peer-relative duration deviation versus risk."),
                deep_graph_panel("PRIORITY × EXPOSURE", deep_priority_exposure_figure(analytics), "Triage priority versus financial exposure percentile."),
            ),
            _deep_section(
                "Risk-signal transparency",
                "Signal prevalence is reported with exact filtered denominators and pairwise co-occurrence counts.",
                deep_graph_panel("SIGNAL PREVALENCE", deep_signal_prevalence_figure(analytics.get("risk_reasons")), "Rule-signal rates and counts."),
                deep_graph_panel("SIGNAL CO-OCCURRENCE", deep_signal_cooccurrence_figure(analytics.get("signal_cooccurrence")), "Counts of records where signal pairs appear together."),
                deep_graph_panel("CATEGORY × RISK", deep_category_risk_heatmap_figure(analytics.get("category_risk")), "Category concentration across risk bands."),
                deep_graph_panel("STATUS × RISK", deep_status_risk_heatmap_figure(analytics.get("status_risk")), "Status and risk composition."),
                deep_graph_panel("HIGH/CRITICAL STATE EXPOSURE", deep_highrisk_state_exposure_figure(analytics), "Financial exposure contained in higher risk bands."),
                deep_graph_panel("HIGH/CRITICAL CATEGORY EXPOSURE", deep_highrisk_category_exposure_figure(analytics), "Category-level high/critical financial exposure."),
            ),
            _deep_section(
                "Model and confidence diagnostics",
                "Diagnostic charts reveal score shape and evidence confidence; they are not model-validation claims by themselves.",
                deep_graph_panel("FINAL RISK DISTRIBUTION", deep_risk_hist_scope_figure(analytics), "Exact score buckets."),
                deep_graph_panel("ML ANOMALY PERCENTILE", deep_anomaly_percentile_figure(analytics), "Seeded sample of anomaly-percentile distribution."),
                deep_graph_panel("CONFIDENCE", deep_confidence_distribution_figure(analytics), "Confidence score distribution."),
                deep_graph_panel("MODEL MISSINGNESS", deep_model_missing_figure(analytics), "Model-input missingness buckets."),
                deep_graph_panel("COMPONENT PROFILE", deep_component_profile_figure(analytics.get("component_profiles")), "Mean versus median analytical component profile."),
                deep_graph_panel("CONFIDENCE × PRIORITY", deep_confidence_priority_figure(analytics), "Confidence score versus investigation priority."),
            ),
            collapsible_queue_panel(works, table_id="risk-queue-table", title="Current investigation queue", subtitle="Displayed records · tap to expand"),
        ],
        className="deep-view",
    )


def deep_finance_view(works: list[dict[str, Any]], analytics: dict[str, Any]) -> html.Div:
    return html.Div(
        [
            section_header("MONEY + EXECUTION", "Financial & Execution Intelligence", "Financial flow, utilisation, transaction freshness, duration, portfolio bands, risk concentration and longitudinal execution patterns."),
            _deep_section(
                "Financial portfolio",
                "Monetary totals are calculated from the filtered analytical universe, not from the capped work queue.",
                deep_graph_panel("FINANCIAL FLOW", deep_sanction_year_expenditure_figure(analytics.get("time")), "Sanctioned versus recorded expenditure."),
                deep_graph_panel("VALUE BANDS", deep_financial_band_figure(analytics), "Work count by sanctioned-value band."),
                deep_graph_panel("VALUE-BAND UTILISATION", deep_financial_band_utilization_figure(analytics), "Utilisation by sanctioned-value band."),
                deep_graph_panel("RISK FINANCIAL EXPOSURE", deep_risk_financial_figure(analytics.get("risk_financial")), "Sanctioned-value exposure by final risk band."),
                deep_graph_panel("RISK × SANCTION", deep_financial_vs_risk_figure(analytics), "Work-level sampled relationship between financial scale and risk."),
                deep_graph_panel("CATEGORY PORTFOLIO", deep_sector_portfolio_figure(analytics), "Sanctioned value by category."),
                deep_graph_panel("STATE PORTFOLIO", deep_state_exposure_amount_figure(analytics), "Sanctioned value by state."),
                deep_graph_panel("STATE EXPOSURE TREEMAP", deep_state_treemap_figure(analytics), "Geographic financial concentration."),
            ),
            _deep_section(
                "Execution dynamics",
                "Timing signals use explicit dates and duration fields from the analytical dataset. The one-year benchmark remains a rule-specific review signal.",
                deep_graph_panel("TIME PERFORMANCE", deep_time_performance_figure(analytics.get("time")), "Completion, utilisation and high/critical rate by sanction year."),
                deep_graph_panel("TIME RISK", deep_time_risk_figure(analytics.get("time")), "Mean and median risk by sanction year."),
                deep_graph_panel("OPEN AGE", deep_open_age_distribution_figure(analytics), "Open-work age distribution."),
                deep_graph_panel("EXPENDITURE GAP", deep_expenditure_gap_figure(analytics), "Days since last recorded expenditure."),
                deep_graph_panel("SANCTION → COMPLETION", deep_completion_duration_figure(analytics), "Completion duration distribution."),
                deep_graph_panel("RECOMMENDATION → SANCTION", deep_lag_figure(analytics), "Recommendation-to-sanction lag distribution."),
                deep_graph_panel("CATEGORY COMPLETION", deep_sector_completion_figure(analytics), "Completion rate by category."),
                deep_graph_panel("CATEGORY UTILISATION", deep_sector_utilization_figure(analytics), "Utilisation by category."),
            ),
            _deep_section(
                "Financial/execution risk relationships",
                "Relationships support review and prioritisation; they are descriptive rather than causal.",
                deep_graph_panel("RISK × UTILISATION", deep_risk_utilization_figure(analytics), "Composite risk versus recorded utilisation."),
                deep_graph_panel("COST DEVIATION × RISK", deep_cost_risk_figure(analytics), "Peer-relative cost outlier score versus risk."),
                deep_graph_panel("DURATION DEVIATION × RISK", deep_duration_risk_figure(analytics), "Peer-relative duration score versus risk."),
                deep_graph_panel("PRIORITY × EXPOSURE", deep_priority_exposure_figure(analytics), "Priority score versus financial exposure percentile."),
                deep_graph_panel("HIGH/CRITICAL STATE EXPOSURE", deep_highrisk_state_exposure_figure(analytics), "Financial exposure among high/critical works by state."),
                deep_graph_panel("HIGH/CRITICAL CATEGORY EXPOSURE", deep_highrisk_category_exposure_figure(analytics), "Financial exposure among high/critical works by category."),
            ),
        ],
        className="deep-view",
    )


def deep_geography_view(works: list[dict[str, Any]], analytics: dict[str, Any]) -> html.Div:
    return html.Div(
        [
            section_header("WHERE", "Geographic Intelligence", "State-level financial concentration, completion, execution signals and district context — all synchronized to the active analytical scope."),
            _deep_section(
                "State landscape",
                "Every chart uses the same active filter scope. Larger states are not directly comparable without considering denominator and volume.",
                deep_graph_panel("STATE HIGH/CRITICAL", deep_state_risk_rate_figure(analytics), "High/critical signal rate by state."),
                deep_graph_panel("STATE COMPLETION", deep_state_completion_rate_figure(analytics), "Completion rate by state."),
                deep_graph_panel("STATE EXPOSURE", deep_state_exposure_amount_figure(analytics), "Sanctioned financial exposure by state."),
                deep_graph_panel("STATE OVERDUE", deep_state_overdue_rate_figure(analytics), "Overdue-open rate by state."),
                deep_graph_panel("STATE DUPLICATE RATE", deep_state_duplicate_rate_figure(analytics), "Potential duplicate-candidate rate by state."),
                deep_graph_panel("STATE PRIORITY", deep_state_priority_figure(analytics), "Mean investigation priority by state."),
                deep_graph_panel("STATE SIGNAL MIX", deep_state_signal_profile_figure(analytics.get("state_signal")), "Stacked counts of signal families."),
                deep_graph_panel("STATE TREEMAP", deep_state_treemap_figure(analytics), "Sanctioned amount as area; high/critical rate as colour."),
                deep_graph_panel("STATE OPERATIONAL PROFILE", deep_date_quality_heatmap_figure(analytics), "Rates for data/duplicate/timing/execution signals."),
                deep_graph_panel("HIGH/CRITICAL EXPOSURE", deep_highrisk_state_exposure_figure(analytics), "High/critical financial exposure by state."),
            ),
            _deep_section(
                "District / IDA depth",
                "District/IDA charts expand the geographic view when a district or state filter is selected; the same population denominator is retained.",
                deep_graph_panel("DISTRICT HIGH/CRITICAL", _deep_group_bar(deep_records(analytics, "district"), "ida", "high_or_critical_rate_pct", "Top district / IDA high-critical signal rates", "Rate (%)", DEEP_RISK_COLORS["HIGH"], 25, 620, True), "High/critical signal rate by district/IDA."),
                deep_graph_panel("DISTRICT COMPLETION", _deep_group_bar(deep_records(analytics, "district"), "ida", "completion_rate_pct", "Top district / IDA completion rates", "Completion rate (%)", DEEP_PALETTE[2], 25, 620, True), "Completion rate by district/IDA."),
                deep_graph_panel("DISTRICT EXPOSURE", _deep_group_bar(deep_records(analytics, "district"), "ida", "sanctioned_amount", "Top district / IDA sanctioned exposure", "Sanctioned amount (₹)", DEEP_PALETTE[0], 25, 620), "Sanctioned amount by district/IDA."),
                deep_graph_panel("DISTRICT PRIORITY", _deep_group_bar(deep_records(analytics, "district"), "ida", "mean_priority", "Top district / IDA mean priority", "Mean priority score", DEEP_PALETTE[1], 25, 620), "Mean investigation priority by district/IDA."),
            ),
            html.Div(
                [
                    html.Div("GEOGRAPHIC CONCENTRATION", className="section-kicker"),
                    html.Div(
                        "State and district concentration is shown through synchronized analytical charts. No geographic boundary rendering or inferred location is used.",
                        className="section-note",
                    ),
                    deep_graph_panel(
                        "STATE CONCENTRATION",
                        _geography_primary_figure(works, analytics),
                        "State financial concentration follows the active analytical filter scope. Use the state and district panels above for volume, exposure, completion and risk context.",
                        class_name="deep-map-inner",
                    ),
                ],
                className="panel deep-map-panel",
            ),
        ],
        className="deep-view geography-final",
    )


def _risk_band_from_score(score: float) -> str:
    """Use the project's existing 0–100 RISK_RANGES exactly."""
    if not math.isfinite(score):
        return ""
    if score <= 25:
        return "LOW"
    if score <= 50:
        return "MEDIUM"
    if score <= 75:
        return "HIGH"
    return "CRITICAL"



def _geography_primary_figure(works: list[dict[str, Any]], analytics: dict[str, Any]) -> go.Figure:
    """Clean geographic concentration view without a geographic map layer."""
    return deep_state_treemap_figure(analytics)

def deep_explorer_view(works: list[dict[str, Any]], analytics: dict[str, Any]) -> html.Div:
    scope = analytics.get("scope", {}) if isinstance(analytics, dict) else {}
    return html.Div(
        [
            section_header("EXPLAIN + REVIEW", "Work Explorer", "Work-level records remain intentionally capped, while all analytical summaries remain complete for the active filter scope."),
            html.Div(
                [
                    kpi_card("FILTERED WORKS", f"{safe_int(scope.get('total_works')):,}", "complete analytical population"),
                    kpi_card("QUEUE DISPLAY", f"{len(works):,}", f"capped at {QUEUE_LIMIT:,}"),
                    kpi_card("MEAN PRIORITY", f"{safe_float(scope.get('mean_risk')):.1f}", "mean final risk shown as scope context"),
                    kpi_card("HIGH / CRITICAL", f"{safe_int(scope.get('high_or_critical')):,}", pct(scope.get("high_or_critical_rate_pct"))),
                ],
                className="deep-insight-grid four",
            ),
            collapsible_queue_panel(works, table_id="explorer-queue-table", title="Work explorer queue", subtitle="Select a row to inspect · tap to expand", selectable=True),
        ],
        className="deep-view",
    )


# ============================================================
# OVERRIDE VIEW FUNCTIONS — APPEND-ONLY ENHANCEMENT
# ============================================================
# These names are intentionally defined immediately before the existing
# render_view callback, so the already-registered callback resolves to the
# richer versions without deleting the original implementation above.
# ============================================================


def overview_view(works, analytics):
    return deep_overview_view(works, analytics if isinstance(analytics, dict) else {})


def risk_view(works, analytics):
    return deep_risk_view(works, analytics if isinstance(analytics, dict) else {})


def finance_view(works, analytics):
    return deep_finance_view(works, analytics if isinstance(analytics, dict) else {})


def geography_view(works, analytics):
    return deep_geography_view(works, analytics if isinstance(analytics, dict) else {})


def explorer_view(works, analytics=None):
    return deep_explorer_view(works, analytics if isinstance(analytics, dict) else {})




@app.callback(
    Output("sidebar-shell", "className"),
    Output("sidebar-collapsed", "data"),
    Output("sidebar-toggle", "children"),
    Input("sidebar-toggle", "n_clicks"),
    State("sidebar-collapsed", "data"),
    prevent_initial_call=True,
)
def toggle_sidebar(n_clicks, collapsed):
    collapsed = not bool(collapsed)
    if collapsed:
        return "sidebar sidebar-collapsed", True, "»"
    return "sidebar", False, "«"



@app.callback(
    Output("seal-viewer", "className"),
    Input("brand-seal", "n_clicks"),
    Input("seal-viewer-close", "n_clicks"),
    Input("seal-viewer-backdrop", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_seal_viewer(_seal_clicks, _close_clicks, _backdrop_clicks):
    trigger = ctx.triggered_id
    if trigger == "brand-seal":
        return "seal-viewer is-open"
    return "seal-viewer is-hidden"


# ============================================================
# AI COPILOT CALLBACKS
# ============================================================













@app.callback(
    Output("active-view", "children"),
    Input("view-mode", "value"),
    Input("filtered-store", "data"),
    Input("works-store", "data"),
    Input("analytics-store", "data"),
    Input("health-store", "data"),
)
def render_view(view, filtered, works_payload, analytics, health):
    if not isinstance(filtered, dict) or filtered.get("error"):
        return empty_panel("Monitoring data is unavailable.")

    works = records(works_payload)

    if analytics is None:
        return analytics_state_view(None)

    if isinstance(analytics, dict) and analytics.get("error"):
        # Geography gets a specific diagnostic because its choropleth depends
        # on the deep-analytics state aggregates. Do not substitute an
        # unfiltered national state table, which would break scope integrity.
        if view == "Geography":
            return html.Div(
                [
                    html.Div("GEOGRAPHIC INTELLIGENCE", className="section-kicker"),
                    html.H2("Waiting for the filtered geography layer", className="section-title"),
                    html.Div(
                        "The View control and all sidebar filters are wired correctly. "
                        "The geography panel is waiting for the same filter-synchronised "
                        "deep-analytics payload used by the other analytical views.",
                        className="section-note",
                    ),
                    html.Div(
                        [
                            html.Span("VIEW · Geography", className="analytics-status-chip"),
                            html.Span("FILTERS · SYNCHRONISED", className="analytics-status-chip"),
                            html.Span("MAP · PENDING", className="analytics-status-chip warning"),
                        ],
                        className="analytics-status-row",
                    ),
                    html.Div(
                        str(analytics.get("error")),
                        className="section-note analytics-error-text",
                    ),
                    html.Div(
                        "Correct the FastAPI /api/v1/deep-analytics error and refresh. "
                        "No unfiltered fallback is shown because that would make the map disagree with the active scope.",
                        className="section-note",
                    ),
                ],
                className="panel analytics-state-panel error",
            )

        return analytics_state_view(analytics)

    analytics = analytics if isinstance(analytics, dict) else {}

    if view == "Risk Intelligence":
        return risk_view(works, analytics)

    if view == "Financial & Execution":
        return finance_view(works, analytics)

    if view == "Geography":
        return geography_view(works, analytics)

    if view == "Work Explorer":
        return explorer_view(works, analytics)

    if view == "Methodology":
        return methodology_view(health)

    return overview_view(works, analytics)


# ============================================================
# WORK DRILL-DOWN
# ============================================================

# ============================================================
# WORK PROFILE LOOKUP
# ============================================================

@app.callback(
    Output("selected-work-detail", "children"),
    Input("work-lookup-button", "n_clicks"),
    Input("demo-work-button", "n_clicks"),
    State("work-id-input", "value"),
    State("works-store", "data"),
    prevent_initial_call=True,
)
def lookup_work(_clicks, _demo_clicks, work_uid, works_payload):
    # Demo mode intentionally reuses the live filtered queue; no hard-coded
    # Work ID is introduced, so the demo stays valid as the dataset changes.
    if ctx.triggered_id == "demo-work-button":
        demo_works = records(works_payload)
        if not demo_works:
            return empty_panel("No work is available in the current scope for the live demo.")
        # /api/v1/works is already requested in priority_score DESC order.
        work_uid = demo_works[0].get("work_uid")

    uid = str(work_uid or "").strip()

    if not uid:
        return empty_panel("Enter a Work ID to open its evidence profile.")

    try:
        detail = api_get(f"/api/v1/works/{uid}")
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status == 404:
            return empty_panel(f"No work was found for Work ID: {uid}")
        return empty_panel(f"Work profile request failed: HTTP {status or 'error'}")
    except requests.RequestException as exc:
        return empty_panel(f"FastAPI request failed: {exc}")

    row = detail.get("work", {}) if isinstance(detail, dict) else {}
    components = detail.get("risk_components", {}) if isinstance(detail, dict) else {}
    evidence = detail.get("evidence", {}) if isinstance(detail, dict) else {}

    if isinstance(evidence, str):
        try:
            evidence = json.loads(evidence)
        except json.JSONDecodeError:
            evidence = {}

    return html.Div(
        [
            html.Div("WORK INTELLIGENCE", className="section-kicker"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(str(row.get("work", uid)), className="detail-title"),
                            html.Div(f"Work ID · {uid}", className="mono"),
                            html.Div(
                                f"{row.get('work_category', 'Unclassified')} · "
                                f"{row.get('state', 'Unknown')} · "
                                f"{row.get('ida', 'Unknown')}",
                                className="detail-meta",
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Div(
                                f"{safe_float(components.get('final_risk_score')):.1f}",
                                className="score-number",
                            ),
                            html.Div(
                                str(row.get("risk_category", "UNSCORED")),
                                className="score-band",
                            ),
                        ],
                        className="score-orb",
                    ),
                ],
                className="detail-head",
            ),
            html.Div(
                [
                    kpi_card("RECOMMENDED", inr(row.get("recommended_amount"))),
                    kpi_card("SANCTIONED", inr(row.get("sanction_amount"))),
                    kpi_card("EXPENDITURE", inr(row.get("total_expenditure"))),
                    kpi_card("UTILISATION", pct(row.get("utilization_pct"))),
                ],
                className="detail-kpis",
            ),
            html.Div(
                [
                    html.Div(
                        dcc.Graph(
                            figure=component_figure(components),
                            config={"displaylogo": False},
                        ),
                        className="panel",
                    ),
                    html.Div(
                        [
                            html.Div("WHY FLAGGED?", className="section-kicker"),
                            html.Div(
                                str(row.get("risk_explanation") or "No strong rule-based signal."),
                                className="explanation",
                            ),
                            html.Div("EVIDENCE", className="section-kicker"),
                            html.Pre(
                                json.dumps(
                                    evidence,
                                    indent=2,
                                    ensure_ascii=False,
                                    default=str,
                                )[:6000],
                                className="evidence",
                            ),
                        ],
                        className="panel",
                    ),
                ],
                className="chart-grid",
            ),
        ],
        className="detail-card",
    )


# ============================================================
# DOWNLOAD
# ============================================================

@app.callback(
    Output("download-queue", "data"),
    Input("download-button", "n_clicks"),
    State("works-store", "data"),
    prevent_initial_call=True,
)
def download_queue(_clicks, works_payload):
    works = records(works_payload)
    frame = to_polars(works)

    if frame.is_empty():
        return no_update

    return dcc.send_string(
        frame.write_csv(),
        "mplads_current_queue.csv",
    )


# ============================================================
# LOGO INTERACTIVE CLICK — clientside callback
# ============================================================
app.clientside_callback(
    """
    function(n) {
        if (!n) return '';
        var el = document.getElementById('model-logo-interactive');
        if (!el) return '';
        el.classList.remove('mlg-clicked');
        void el.offsetWidth;          // force reflow so animation restarts
        el.classList.add('mlg-clicked');
        setTimeout(function () { el.classList.remove('mlg-clicked'); }, 800);
        return '';
    }
    """,
    Output("_logo-click-sink", "children"),
    Input("model-logo-interactive", "n_clicks"),
    prevent_initial_call=True,
)


# ============================================================
# ENTRY POINT
# ============================================================
# ============================================================
# ENTRY POINT
# ============================================================


# ============================================================
# FINAL TEAM CARD + COPILOT GLASS UX OVERRIDES
# ============================================================

app.index_string = app.index_string.replace("</head>", r"""
<style>
/* Team cards: framed portrait column + editorial information column. */
.team-member-card{
  display:grid!important;
  grid-template-columns:minmax(118px,42%) minmax(0,1fr)!important;
  gap:15px!important;
  min-height:194px!important;
  padding:10px!important;
  border-radius:22px!important;
  background:linear-gradient(135deg,rgba(19,31,43,.97),rgba(8,15,22,.98))!important;
  border:1px solid rgba(143,208,255,.14)!important;
  box-shadow:0 16px 38px rgba(0,0,0,.22),0 1px 0 rgba(255,255,255,.045) inset!important;
}
.team-member-card:hover{transform:translateY(-3px)!important;border-color:rgba(143,208,255,.34)!important;box-shadow:0 22px 44px rgba(0,0,0,.30),0 0 24px rgba(126,200,255,.08)!important}
.team-member-visual{position:relative!important;min-width:0!important;min-height:172px!important}
.team-member-photo-frame{
  position:relative!important;height:100%!important;min-height:172px!important;
  border-radius:30px!important;overflow:hidden!important;
  border:1px solid rgba(143,208,255,.24)!important;
  background:linear-gradient(160deg,#152A3C,#0C151E)!important;
  box-shadow:0 0 0 5px rgba(143,208,255,.035),0 13px 28px rgba(0,0,0,.28)!important;
}
.team-member-photo-frame:before{content:"";position:absolute;inset:7px;border-radius:24px;border:1px solid rgba(255,255,255,.08);pointer-events:none;z-index:3}
.team-member-photo-frame:after{content:"";position:absolute;inset:-25% -15%;background:linear-gradient(120deg,transparent 35%,rgba(255,255,255,.09) 48%,transparent 60%);transform:translateX(-70%);transition:transform .55s ease;pointer-events:none;z-index:4}
.team-member-card:hover .team-member-photo-frame:after{transform:translateX(70%)}
.team-member-avatar{width:100%!important;height:100%!important;border:0!important;border-radius:28px!important;box-shadow:none!important;background:transparent!important}
.team-member-avatar-initials{position:relative!important;display:flex!important;flex-direction:column!important;justify-content:center!important;align-items:center!important;gap:5px!important;background:radial-gradient(circle at 32% 22%,rgba(143,208,255,.26),transparent 35%),linear-gradient(155deg,#18344B 0%,#0B141D 76%)!important}
.team-member-avatar-initials-main{font:800 36px/1 Bahnschrift,"Segoe UI",sans-serif!important;letter-spacing:-.05em!important;color:#F4F8FB!important;text-shadow:0 6px 22px rgba(143,208,255,.16)!important}
.team-member-avatar-caption{font:800 7px/1 Cascadia Mono,Consolas,monospace!important;letter-spacing:.18em!important;color:#79A7BE!important}
.team-member-photo{
  width:100%!important;height:100%!important;object-fit:cover!important;object-position:50% 18%!important;display:block!important;
  transform:translateZ(0)!important;backface-visibility:hidden!important;
  image-rendering:auto!important;
  filter:none!important;transition:transform .35s cubic-bezier(.2,.8,.2,1)!important;
}
.team-member-card:hover .team-member-photo{transform:translateZ(0) scale(1.04)!important;filter:none!important}
.team-member-photo-frame{transform:translateZ(0)!important;will-change:transform!important;isolation:isolate!important}
.team-member-photo-corner{position:absolute!important;left:11px!important;bottom:11px!important;z-index:5!important;padding:5px 7px!important;border-radius:999px!important;background:rgba(5,12,18,.74)!important;border:1px solid rgba(143,208,255,.18)!important;backdrop-filter:blur(8px)!important}
.team-member-photo-index{font:800 7px/1 Cascadia Mono,Consolas,monospace!important;letter-spacing:.14em!important;color:#A8D4EB!important}
.team-member-content{min-width:0!important;display:flex!important;flex-direction:column!important;justify-content:center!important;padding:7px 5px 7px 0!important}
.team-member-name-row{display:flex!important;align-items:flex-start!important;justify-content:space-between!important;gap:8px!important}
.team-member-name{font-size:17px!important;line-height:1.08!important;margin:0!important;min-width:0!important}
.team-member-role{margin-top:5px!important;font-size:8px!important;line-height:1.25!important;letter-spacing:.13em!important;max-width:95%!important}
.team-member-degree{margin-top:7px!important;font-size:9px!important}
.team-member-bio{margin-top:9px!important;font-size:9.5px!important;line-height:1.48!important;max-width:none!important;color:#8094A1!important}
.team-member-tags{display:flex!important;gap:5px!important;flex-wrap:wrap!important;margin-top:auto!important;padding-top:9px!important}
.team-member-tag{padding:4px 6px!important;border-radius:999px!important;border:1px solid rgba(143,208,255,.10)!important;background:rgba(143,208,255,.025)!important;color:#6E9AB0!important;font:800 6.5px/1 Cascadia Mono,Consolas,monospace!important;letter-spacing:.12em!important}
.team-member-tag-gold{color:#CDB27A!important;border-color:rgba(240,198,117,.14)!important;background:rgba(240,198,117,.025)!important}
.team-member-jmi-badge{position:relative!important;right:auto!important;bottom:auto!important;flex:0 0 32px!important;width:32px!important;height:32px!important;min-width:32px!important;border-radius:11px!important;padding:2px!important;border:1px solid rgba(255,255,255,.24)!important;background:rgba(255,255,255,.96)!important;box-shadow:0 6px 14px rgba(0,0,0,.18)!important;overflow:hidden!important}
.team-member-jmi-logo{width:100%!important;height:100%!important;object-fit:contain!important;object-position:center!important;border-radius:8px!important;background:#fff!important;display:block!important}
.team-member-jmi-badge-fallback{display:flex!important;flex-direction:column!important;justify-content:center!important;align-items:center!important;color:#114B31!important;background:linear-gradient(145deg,#FFFFFF,#ECF7F0)!important}
.team-member-jmi-fallback-main{font:900 8px/1 Bahnschrift,"Segoe UI",sans-serif!important;letter-spacing:.06em!important}.team-member-jmi-fallback-sub{font:700 5px/1 Cascadia Mono,Consolas,monospace!important;color:#6C7C75!important;margin-top:2px!important}


</style>
""")

# ============================================================

# ============================================================
# AI COPILOT FINAL COMPATIBILITY + SHARP MEDIA PATCH
# Compatible with Dash 3.4.x html.Img (no HTML-only loading/decoding props).
# ============================================================

app.index_string = app.index_string.replace("</head>", r"""
<style>
/* Keep portrait pixels native-sharp. No filters, blur, transforms, or GPU resampling. */
.team-member-photo,
.team-member-jmi-logo,
.team-member-photo-frame img{
  filter:none!important;
  -webkit-filter:none!important;
  transform:none!important;
  -webkit-transform:none!important;
  backface-visibility:visible!important;
  -webkit-backface-visibility:visible!important;
  image-rendering:auto!important;
  display:block!important;
}
.team-member-photo{
  width:100%!important;height:100%!important;
  object-fit:cover!important;object-position:50% 50%!important;
}
.team-member-photo-frame,
.team-member-avatar-photo{
  overflow:hidden!important;
  transform:none!important;
  filter:none!important;
  backdrop-filter:none!important;
  -webkit-backdrop-filter:none!important;
}
/* The page remains clear behind the Copilot; only the panel itself gets glass. */
.copilot-backdrop{backdrop-filter:none!important;-webkit-backdrop-filter:none!important;filter:none!important}
.copilot-panel,.copilot-launcher{
  backdrop-filter:blur(16px) saturate(112%)!important;
  -webkit-backdrop-filter:blur(16px) saturate(112%)!important;
}
.copilot-panel{will-change:opacity,transform;}
.copilot-messages{font-family:"Segoe UI Variable","Segoe UI",Arial,sans-serif!important;font-size:14px!important;line-height:1.62!important}
.copilot-bubble{font-size:14px!important;line-height:1.62!important}
.copilot-md{font-size:14px!important;line-height:1.62!important}
.copilot-welcome-copy{font-size:13.5px!important;line-height:1.68!important}
</style>
""" + "</head>")


# ============================================================
# TEAM INFO — LIVE MODAL INTERACTION
# ============================================================
@app.callback(
    Output("team-info-modal", "className"),
    Input("team-info-button", "n_clicks"),
    Input("team-info-close", "n_clicks"),
    Input("team-info-backdrop", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_team_info_modal(_open_clicks, _close_clicks, _backdrop_clicks):
    """Open Team Info on demand and close it deterministically.

    The modal lives in the static Dash layout, so this callback only switches
    its state class; it never rebuilds or hides the six member cards.
    """
    trigger = ctx.triggered_id
    if trigger == "team-info-button":
        return "team-modal is-open"
    return "team-modal is-hidden"


# FINAL AI COPILOT + TEAM PHOTO HD UX LAYER
# Scoped to Copilot and team portraits. The analytical engine remains upstream.
# ============================================================

app.index_string = app.index_string.replace("</head>", r"""
<style>
:root{
  --cop-bg:#07111A;--cop-surface:#0B1722;--cop-surface-2:#0E1D29;
  --cop-line:rgba(142,205,235,.16);--cop-line-strong:rgba(142,205,235,.30);
  --cop-text:#F2F7FA;--cop-muted:#91A7B4;--cop-soft:#B9CAD4;
  --cop-accent:#71C9F2;--cop-accent-2:#A88AF5;--cop-success:#4CD39A;
}

/* ---------- AI COPILOT ---------- */
.copilot-root{position:relative!important;z-index:1400!important;font-family:"Segoe UI",Aptos,Arial,sans-serif!important}
.copilot-launcher{
  position:fixed!important;right:24px!important;bottom:24px!important;z-index:1410!important;
  display:inline-flex!important;align-items:center!important;justify-content:center!important;gap:10px!important;
  min-width:168px!important;height:50px!important;padding:0 16px!important;border-radius:14px!important;
  border:1px solid var(--cop-line-strong)!important;
  background:linear-gradient(145deg,rgba(15,34,48,.97),rgba(7,18,27,.985))!important;
  color:var(--cop-text)!important;cursor:pointer!important;
  box-shadow:0 18px 44px rgba(0,0,0,.40),inset 0 1px 0 rgba(255,255,255,.055)!important;
  backdrop-filter:blur(14px) saturate(115%)!important;-webkit-backdrop-filter:blur(14px) saturate(115%)!important;
  font:850 10px/1 "Cascadia Mono",Consolas,monospace!important;letter-spacing:.11em!important;
  transition:transform .18s ease,border-color .18s ease,box-shadow .18s ease!important;
}
.copilot-launcher:hover{transform:translateY(-2px)!important;border-color:rgba(113,201,242,.52)!important;box-shadow:0 22px 52px rgba(0,0,0,.46),0 0 34px rgba(76,189,235,.09),inset 0 1px 0 rgba(255,255,255,.07)!important}
.copilot-launch-label{white-space:nowrap!important;color:#F1F7FA!important}
.copilot-launch-live{font-size:8px!important;color:var(--cop-success)!important;letter-spacing:.08em!important}

.copilot-backdrop{
  position:fixed!important;inset:0!important;z-index:1405!important;
  background:rgba(2,8,13,.32)!important;
  backdrop-filter:none!important;-webkit-backdrop-filter:none!important;
  opacity:1!important;visibility:visible!important;pointer-events:auto!important;
  transition:opacity .16s ease,visibility .16s ease!important;
}
.copilot-backdrop.is-hidden{opacity:0!important;visibility:hidden!important;pointer-events:none!important}

.copilot-panel{
  position:fixed!important;right:24px!important;bottom:86px!important;z-index:1420!important;
  width:min(560px,calc(100vw - 36px))!important;height:min(790px,calc(100vh - 112px))!important;
  min-height:520px!important;display:flex!important;flex-direction:column!important;overflow:hidden!important;
  border:1px solid rgba(132,207,239,.22)!important;border-radius:24px!important;
  background:
    linear-gradient(180deg,rgba(11,27,39,.985),rgba(7,16,25,.985)) padding-box,
    linear-gradient(145deg,rgba(113,201,242,.38),rgba(168,138,245,.16),rgba(76,211,154,.13)) border-box!important;
  box-shadow:0 32px 90px rgba(0,0,0,.62),0 0 58px rgba(79,183,226,.08),inset 0 1px 0 rgba(255,255,255,.06)!important;
  transform-origin:bottom right!important;transition:opacity .18s ease,transform .20s ease,visibility .18s ease!important;
}
.copilot-panel:before{content:"";position:absolute;inset:0;pointer-events:none;z-index:0;background:radial-gradient(circle at 91% 5%,rgba(83,184,235,.10),transparent 24%),radial-gradient(circle at 9% 92%,rgba(135,105,236,.075),transparent 22%);}
.copilot-panel:after{content:"";position:absolute;inset:0;border-radius:inherit;pointer-events:none;z-index:8;box-shadow:inset 0 0 0 1px rgba(255,255,255,.022),inset 0 1px 0 rgba(255,255,255,.05)}
.copilot-panel.is-hidden{opacity:0!important;visibility:hidden!important;pointer-events:none!important;transform:translateY(12px) scale(.978)!important}
.copilot-panel.is-open{opacity:1!important;visibility:visible!important;pointer-events:auto!important;transform:none!important}

.copilot-header,.copilot-context-ribbon,.copilot-loading,.copilot-quick-wrap,.copilot-composer,.copilot-footer{position:relative!important;z-index:2!important}
.copilot-header{display:flex!important;justify-content:space-between!important;gap:16px!important;padding:17px 18px 14px!important;background:rgba(7,18,28,.74)!important;border-bottom:1px solid rgba(142,205,235,.09)!important}
.copilot-heading-copy{display:flex!important;align-items:center!important;gap:12px!important;min-width:0!important}
.copilot-heading-text{min-width:0!important}
.copilot-title{color:var(--cop-text)!important;font:900 16px/1 "Bahnschrift","Segoe UI",Arial,sans-serif!important;letter-spacing:.10em!important}
.copilot-subtitle{margin-top:5px!important;color:#8EA6B4!important;font:600 9px/1.2 "Segoe UI",Arial,sans-serif!important;letter-spacing:.02em!important}
.copilot-status{display:flex!important;align-items:center!important;gap:6px!important;margin-top:7px!important;color:#718B9B!important;font:850 7px/1 "Cascadia Mono",Consolas,monospace!important;letter-spacing:.12em!important}
.copilot-status-dot{width:6px!important;height:6px!important;border-radius:50%!important;background:var(--cop-success)!important;box-shadow:0 0 10px rgba(76,211,154,.55)!important}
.copilot-heading-actions{display:flex!important;gap:7px!important;align-items:center!important}
.copilot-icon-btn,.copilot-close-btn{height:36px!important;min-width:36px!important;padding:0 11px!important;border-radius:10px!important;border:1px solid rgba(142,205,235,.14)!important;background:rgba(255,255,255,.027)!important;color:#B7C9D2!important;cursor:pointer!important;font:850 8px/1 "Cascadia Mono",Consolas,monospace!important;letter-spacing:.08em!important}
.copilot-icon-btn:hover,.copilot-close-btn:hover{border-color:rgba(113,201,242,.36)!important;background:rgba(113,201,242,.06)!important;color:#F1F8FB!important}
.copilot-close-btn{min-width:58px!important}

.copilot-context-ribbon{margin:10px 13px 7px!important;padding:10px 11px!important;border-radius:16px!important;border:1px solid rgba(142,205,235,.11)!important;background:linear-gradient(145deg,rgba(14,36,50,.78),rgba(10,24,36,.78))!important;box-shadow:inset 0 1px 0 rgba(255,255,255,.028),0 8px 22px rgba(0,0,0,.12)!important}
.copilot-context-topline{display:flex!important;align-items:center!important;gap:7px!important;margin-bottom:7px!important}
.copilot-context-live{color:var(--cop-success)!important;font:900 7px/1 "Cascadia Mono",Consolas,monospace!important;letter-spacing:.12em!important}
.copilot-context-kicker{color:#7694A5!important;font:850 7px/1 "Cascadia Mono",Consolas,monospace!important;letter-spacing:.13em!important}
.copilot-context-metrics{display:flex!important;flex-wrap:wrap!important;gap:9px 14px!important;align-items:center!important}
.copilot-context-stat{display:flex!important;align-items:baseline!important;gap:5px!important}
.copilot-context-count{color:#ECF6FA!important;font:900 14px/1 "Bahnschrift","Segoe UI",sans-serif!important}
.copilot-context-count-alert{color:#F2C5A4!important}
.copilot-context-muted{color:#7892A1!important;font:750 7px/1.2 "Cascadia Mono",Consolas,monospace!important;letter-spacing:.03em!important}
.copilot-context-connection-dot{color:var(--cop-success)!important;font-size:8px!important}
.copilot-context-asof{color:#AFC3CD!important;font:800 7px/1.2 "Cascadia Mono",Consolas,monospace!important}
.copilot-context-filters{display:flex!important;flex-wrap:wrap!important;gap:5px!important;margin-top:8px!important}
.copilot-context-filter{padding:5px 7px!important;border-radius:999px!important;border:1px solid rgba(142,205,235,.09)!important;background:rgba(255,255,255,.022)!important;color:#91AAB7!important;font:750 7px/1 "Cascadia Mono",Consolas,monospace!important}
.copilot-context-filter strong{color:#DCEAF0!important;font-weight:900!important}

.copilot-loading{flex:1 1 auto!important;min-height:0!important;display:flex!important;overflow:hidden!important}
.copilot-loading>div{flex:1 1 auto!important;min-height:0!important;display:flex!important;overflow:hidden!important}
.copilot-messages{flex:1 1 auto!important;min-height:0!important;overflow-y:auto!important;overflow-x:hidden!important;padding:10px 13px 12px!important;scrollbar-width:thin!important;scroll-behavior:smooth!important}
.copilot-message{margin:8px 0!important;display:flex!important}.copilot-message-user{justify-content:flex-end!important}.copilot-message-assistant{justify-content:flex-start!important}
.copilot-bubble{position:relative!important;max-width:91%!important;padding:12px 14px!important;border-radius:16px!important;box-shadow:0 9px 24px rgba(0,0,0,.13)!important}
.copilot-message-user .copilot-bubble{background:linear-gradient(145deg,#173D54,#125C7B)!important;color:#F8FDFF!important;border:1px solid rgba(113,201,242,.24)!important;border-bottom-right-radius:6px!important}
.copilot-message-assistant .copilot-bubble{background:linear-gradient(145deg,rgba(15,37,53,.94),rgba(9,25,37,.96))!important;color:#E4EFF4!important;border:1px solid rgba(142,205,235,.11)!important;border-bottom-left-radius:6px!important}
.copilot-bubble-main{min-width:0!important;flex:1!important}
.copilot-role{margin-bottom:6px!important;color:#7895A5!important;font:900 7px/1 "Cascadia Mono",Consolas,monospace!important;letter-spacing:.13em!important}
.copilot-message-user .copilot-role{color:#C5EDF9!important}
.copilot-md{font:600 11.9px/1.65 "Segoe UI",Aptos,Arial,sans-serif!important;color:#DCEAF1!important}
.copilot-message-user .copilot-md{color:#FBFEFF!important}
.copilot-md p{margin:.40em 0!important}.copilot-md ul,.copilot-md ol{margin:.45em 0 .45em 1.05em!important;padding-left:13px!important}.copilot-md li{margin:.22em 0!important}
.copilot-md strong{color:#F7FBFD!important;font-weight:850!important}.copilot-md code{font:700 10px/1.4 "Cascadia Mono",Consolas,monospace!important;color:#A9E4FA!important;background:rgba(113,201,242,.07)!important;border:1px solid rgba(113,201,242,.08)!important;padding:2px 5px!important;border-radius:5px!important}
.copilot-md table{width:100%!important;border-collapse:separate!important;border-spacing:0!important;margin:9px 0!important;overflow:hidden!important;border:1px solid rgba(142,205,235,.10)!important;border-radius:10px!important;font-size:10px!important}.copilot-md th{padding:7px 7px!important;color:#A6D4E4!important;text-align:left!important;background:rgba(113,201,242,.045)!important;border-bottom:1px solid rgba(142,205,235,.09)!important}.copilot-md td{padding:6px 7px!important;color:#D9E8EF!important;border-bottom:1px solid rgba(142,205,235,.055)!important;vertical-align:top!important}.copilot-md tr:last-child td{border-bottom:0!important}
.copilot-sources{display:flex!important;flex-wrap:wrap!important;gap:5px!important;margin-top:8px!important}.copilot-source-chip{padding:4px 7px!important;border-radius:999px!important;border:1px solid rgba(142,205,235,.10)!important;background:rgba(113,201,242,.026)!important;color:#86A8B7!important;font:800 7px/1 "Cascadia Mono",Consolas,monospace!important}
.copilot-copy{position:absolute!important;right:8px!important;bottom:8px!important;padding:4px 6px!important;border-radius:7px!important;border:1px solid rgba(142,205,235,.10)!important;background:rgba(255,255,255,.025)!important;color:#7895A3!important;cursor:pointer!important;font:850 6.5px/1 "Cascadia Mono",Consolas,monospace!important;opacity:.0!important;transition:opacity .14s ease,color .14s ease,border-color .14s ease!important}.copilot-message-assistant:hover .copilot-copy{opacity:1!important}.copilot-copy.copied{color:var(--cop-success)!important;border-color:rgba(76,211,154,.25)!important}

.copilot-welcome{box-sizing:border-box!important;margin:5px 0 10px!important;padding:18px!important;border-radius:18px!important;background:linear-gradient(145deg,rgba(13,37,52,.86),rgba(8,22,33,.92))!important;border:1px solid rgba(142,205,235,.11)!important;box-shadow:0 13px 30px rgba(0,0,0,.14),inset 0 1px 0 rgba(255,255,255,.035)!important}
.copilot-welcome-bot-wrap{display:flex!important;justify-content:flex-start!important;margin-bottom:10px!important}
.copilot-welcome-kicker{color:#69C7ED!important;font:900 7px/1 "Cascadia Mono",Consolas,monospace!important;letter-spacing:.15em!important}.copilot-welcome-title{margin-top:6px!important;color:#F3F8FA!important;font:900 26px/1.04 "Bahnschrift","Segoe UI",sans-serif!important;letter-spacing:-.02em!important}.copilot-welcome-copy{margin-top:8px!important;color:#91A7B3!important;font:600 11px/1.56 "Segoe UI",Aptos,Arial,sans-serif!important;max-width:62ch!important}
.copilot-welcome-suggestions{display:grid!important;grid-template-columns:repeat(3,minmax(0,1fr))!important;gap:7px!important;margin-top:14px!important}.copilot-welcome-suggestion{min-height:38px!important;padding:7px 9px!important;border-radius:10px!important;border:1px solid rgba(142,205,235,.10)!important;background:rgba(255,255,255,.025)!important;color:#B7CCD5!important;cursor:pointer!important;font:800 7.5px/1.2 "Cascadia Mono",Consolas,monospace!important;text-align:left!important}.copilot-welcome-suggestion:hover{border-color:rgba(113,201,242,.28)!important;background:rgba(113,201,242,.045)!important;color:#EFF8FC!important}
.copilot-welcome-live{display:flex!important;align-items:center!important;gap:7px!important;margin-top:13px!important;padding-top:10px!important;border-top:1px solid rgba(142,205,235,.08)!important}.copilot-welcome-live-dot{width:6px!important;height:6px!important;border-radius:50%!important;background:var(--cop-success)!important;box-shadow:0 0 10px rgba(76,211,154,.4)!important}.copilot-welcome-live-label{color:#5DCE9B!important;font:900 7px/1 "Cascadia Mono",Consolas,monospace!important;letter-spacing:.12em!important}.copilot-welcome-live-text{color:#748F9D!important;font:700 7px/1.2 "Cascadia Mono",Consolas,monospace!important}

.copilot-quick-wrap{flex:0 0 auto!important;padding:10px 13px 8px!important;border-top:1px solid rgba(142,205,235,.09)!important;background:linear-gradient(180deg,rgba(8,23,34,.94),rgba(7,18,28,.97))!important}
.copilot-section-label{margin-bottom:7px!important;color:#7796A6!important;font:900 7px/1 "Cascadia Mono",Consolas,monospace!important;letter-spacing:.14em!important}.copilot-quick-grid{display:grid!important;grid-template-columns:repeat(3,minmax(0,1fr))!important;gap:6px!important}.copilot-quick-btn{min-height:34px!important;padding:7px 8px!important;border-radius:10px!important;border:1px solid rgba(142,205,235,.09)!important;background:rgba(255,255,255,.025)!important;color:#A9C2CD!important;cursor:pointer!important;text-align:left!important;font:800 7.7px/1.12 "Cascadia Mono",Consolas,monospace!important;transition:transform .14s ease,border-color .14s ease,color .14s ease,background .14s ease!important}.copilot-quick-btn:hover{transform:translateY(-1px)!important;border-color:rgba(113,201,242,.28)!important;background:rgba(113,201,242,.045)!important;color:#ECF7FB!important}

.copilot-composer{display:flex!important;align-items:stretch!important;gap:7px!important;padding:10px 13px 8px!important;background:rgba(6,17,26,.97)!important}
.copilot-input-wrap{position:relative!important;display:flex!important;flex:1 1 auto!important;min-width:0!important}.copilot-input-icon{position:absolute!important;left:12px!important;top:12px!important;z-index:3!important;color:#76B7D1!important;font-size:12px!important}.copilot-input{width:100%!important;min-height:58px!important;max-height:108px!important;resize:none!important;padding:12px 12px 11px 31px!important;border-radius:14px!important;border:1px solid rgba(142,205,235,.14)!important;background:linear-gradient(145deg,rgba(9,29,42,.96),rgba(7,21,32,.98))!important;color:#F0F7FA!important;outline:none!important;font:600 11.5px/1.5 "Segoe UI",Aptos,Arial,sans-serif!important;box-shadow:inset 0 1px 0 rgba(255,255,255,.035)!important}
.copilot-input::placeholder{color:#6F8998!important}.copilot-input:focus{border-color:rgba(113,201,242,.40)!important;box-shadow:0 0 0 3px rgba(113,201,242,.065),inset 0 1px 0 rgba(255,255,255,.05)!important}
.copilot-mic{width:40px!important;min-width:40px!important;border-radius:12px!important;border:1px solid rgba(142,205,235,.12)!important;background:rgba(255,255,255,.025)!important;color:#9BBFCC!important;cursor:pointer!important;font-size:18px!important;transition:background .14s ease,border-color .14s ease,transform .14s ease!important}.copilot-mic:hover{transform:translateY(-1px)!important;border-color:rgba(113,201,242,.28)!important;background:rgba(113,201,242,.045)!important}
.copilot-send{width:76px!important;min-width:76px!important;border-radius:13px!important;border:1px solid rgba(113,201,242,.30)!important;background:linear-gradient(145deg,#16719A,#226C8A)!important;color:#FFFFFF!important;cursor:pointer!important;display:flex!important;flex-direction:column!important;align-items:center!important;justify-content:center!important;gap:4px!important;box-shadow:0 12px 25px rgba(17,104,139,.20)!important}.copilot-send:hover{transform:translateY(-1px)!important;filter:saturate(1.05)!important}.copilot-send-icon{font-size:18px!important;line-height:1!important}.copilot-send-label{font:900 7px/1 "Cascadia Mono",Consolas,monospace!important;letter-spacing:.13em!important;color:#DDF6FF!important}

.copilot-footer{display:flex!important;align-items:center!important;justify-content:space-between!important;gap:8px!important;min-height:28px!important;padding:0 13px 8px!important;background:rgba(6,17,26,.97)!important}.copilot-footer-left{display:flex!important;align-items:center!important;gap:5px!important;min-width:0!important}.copilot-footer-chip{padding:4px 6px!important;border-radius:7px!important;border:1px solid rgba(142,205,235,.08)!important;background:rgba(255,255,255,.018)!important;color:#718B99!important;font:850 6px/1 "Cascadia Mono",Consolas,monospace!important;letter-spacing:.08em!important}.copilot-footer-tip{color:#718B99!important;font:800 6.5px/1 "Cascadia Mono",Consolas,monospace!important;white-space:nowrap!important}

/* ---------- CSS AI COPILOT LOGO: crisp vector, no raster blur ---------- */
.copilot-logo{position:relative!important;display:inline-block!important;flex:0 0 auto!important;width:36px!important;height:36px!important;isolation:isolate!important}
.copilot-logo:before{content:"";position:absolute!important;inset:5px!important;border-radius:10px!important;border:1px solid rgba(136,214,244,.58)!important;background:linear-gradient(145deg,rgba(24,79,109,.92),rgba(13,34,51,.98))!important;box-shadow:inset 0 1px 0 rgba(255,255,255,.28),0 0 20px rgba(113,201,242,.10)!important;transform:rotate(45deg)!important}
.copilot-logo:after{content:"AI";position:absolute!important;inset:0!important;display:grid!important;place-items:center!important;color:#F3FAFD!important;font:900 10px/1 "Bahnschrift","Segoe UI",sans-serif!important;letter-spacing:.06em!important;text-shadow:0 1px 8px rgba(113,201,242,.18)!important}
.copilot-logo-orbit{display:none!important}.copilot-logo-aura,.copilot-logo-node,.copilot-logo-core,.copilot-logo-glint{position:absolute!important;display:none!important}.copilot-logo-header{width:40px!important;height:40px!important}.copilot-logo-launcher{width:28px!important;height:28px!important}.copilot-logo-launcher:after{font-size:8px!important}.copilot-logo-launcher:before{inset:4px!important;border-radius:8px!important}
.copilot-logo-welcome{width:44px!important;height:44px!important}.copilot-logo-message{width:26px!important;height:26px!important;margin-right:8px!important}.copilot-logo-message:before{inset:4px!important;border-radius:8px!important}.copilot-logo-message:after{font-size:7px!important}

/* ---------- Team portraits: true sharpness, no CSS blur / no image-rendering hacks ---------- */
.team-member-photo-frame{transform:translateZ(0)!important;isolation:isolate!important;backdrop-filter:none!important;-webkit-backdrop-filter:none!important}
.team-member-photo-frame:after{display:none!important}
.team-member-photo,.team-member-jmi-logo{filter:none!important;image-rendering:auto!important;backface-visibility:hidden!important;-webkit-backface-visibility:hidden!important}
.team-member-photo{width:100%!important;height:100%!important;display:block!important;object-fit:cover!important;object-position:50% 18%!important;transform:none!important;will-change:auto!important;-webkit-user-drag:none!important}
.team-member-card:hover .team-member-photo{transform:scale(1.012)!important;transition:transform .16s ease!important}
.team-member-photo-frame:before{inset:6px!important;border-radius:23px!important;border:1px solid rgba(255,255,255,.07)!important;z-index:3!important}
.team-member-avatar-caption{font-size:6.5px!important;letter-spacing:.10em!important;text-align:center!important;max-width:90%!important;line-height:1.25!important}

@media(max-width:850px){
  .copilot-launcher{right:14px!important;bottom:14px!important;min-width:0!important;width:54px!important;height:54px!important;padding:0!important;border-radius:15px!important}
  .copilot-launch-label,.copilot-launch-live{display:none!important}
  .copilot-panel{right:10px!important;bottom:78px!important;width:calc(100vw - 20px)!important;height:min(760px,calc(100vh - 90px))!important;min-height:500px!important;border-radius:20px!important}
  .copilot-welcome-suggestions{grid-template-columns:1fr 1fr!important}
}
@media(max-width:560px){
  .copilot-panel{right:0!important;bottom:0!important;width:100vw!important;height:100dvh!important;max-height:100dvh!important;min-height:0!important;border-radius:0!important}
  .copilot-header{padding:13px 12px!important}.copilot-context-ribbon{margin:8px 9px 7px!important}.copilot-messages{padding:8px 9px 10px!important}.copilot-quick-wrap,.copilot-composer{padding-left:9px!important;padding-right:9px!important}.copilot-footer{padding-left:9px!important;padding-right:9px!important}.copilot-footer-tip{display:none!important}
  .copilot-md{font-size:11.2px!important}.copilot-welcome{padding:15px 12px!important}.copilot-welcome-title{font-size:23px!important}.copilot-send{width:63px!important;min-width:63px!important}.copilot-mic{width:38px!important;min-width:38px!important}.copilot-context-filter{font-size:6.5px!important}
}
@media(prefers-reduced-motion:reduce){.copilot-root *,.copilot-root *:before,.copilot-root *:after{animation:none!important;transition:none!important;scroll-behavior:auto!important}.copilot-launcher:hover,.copilot-send:hover{transform:none!important}.team-member-card:hover .team-member-photo{transform:none!important}}
</style>
<script>
(function(){
  function el(id){return document.getElementById(id)}
  function click(id){var e=el(id);if(e){e.click();return true}return false}
  function focusInput(){var e=el("copilot-input");if(e){setTimeout(function(){e.focus()},80)}}
  function scrollChat(){var e=el("copilot-messages");if(!e)return;requestAnimationFrame(function(){e.scrollTop=e.scrollHeight})}
  function writeTextarea(value){var e=el("copilot-input");if(!e)return;var d=Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype,"value");if(d&&d.set)d.set.call(e,value);else e.value=value;e.dispatchEvent(new Event("input",{bubbles:true}));e.dispatchEvent(new Event("change",{bubbles:true}));e.focus()}
  function copyReply(btn){if(btn.dataset.bound)return;btn.dataset.bound="1";btn.addEventListener("click",function(ev){ev.preventDefault();var bubble=btn.closest(".copilot-bubble-assistant");if(!bubble)return;var main=bubble.querySelector(".copilot-bubble-main");var value=main?main.innerText.trim():bubble.innerText.trim();var done=function(){btn.textContent="Copied";btn.classList.add("copied");setTimeout(function(){btn.textContent="Copy";btn.classList.remove("copied")},900)};if(navigator.clipboard&&navigator.clipboard.writeText){navigator.clipboard.writeText(value).then(done).catch(function(){})}else{var ta=document.createElement("textarea");ta.value=value;document.body.appendChild(ta);ta.select();try{document.execCommand("copy");done()}catch(_e){}ta.remove()}})}
  function bind(){document.querySelectorAll(".copilot-copy").forEach(copyReply)}
  document.addEventListener("keydown",function(event){
    var key=(event.key||"").toLowerCase();
    var p=el("copilot-panel");
    if((event.ctrlKey||event.metaKey)&&key==="k"){event.preventDefault();if(p&&p.classList.contains("is-open")){click("copilot-close")}else{click("copilot-launcher");focusInput()};return}
    if(key==="escape"&&p&&p.classList.contains("is-open")){event.preventDefault();click("copilot-close");return}
    if((key==="enter"||key==="return")&&!event.shiftKey&&document.activeElement&&document.activeElement.id==="copilot-input"){event.preventDefault();click("copilot-send")}
    if(key==="arrowup"&&document.activeElement&&document.activeElement.id==="copilot-input"&&!(document.activeElement.value||"")){var last=null;if(last){event.preventDefault();writeTextarea(last)}}
  });
  document.addEventListener("click",function(event){
    var t=event.target.closest&&event.target.closest("#copilot-launcher,#copilot-prompt-scope,#copilot-prompt-compare,#copilot-prompt-risk,#copilot-prompt-financial,#copilot-prompt-chart,#copilot-prompt-methodology,#copilot-welcome-compare,#copilot-welcome-risk,#copilot-welcome-work");
    if(t){setTimeout(function(){focusInput();scrollChat()},120)}
    if(event.target.closest&&event.target.closest("#copilot-send")){var input=el("copilot-input");var q=input&&input.value?input.value.trim():"";if(q)localStorage.setItem("mplads_ai_copilot_last_question",q)}
  });
  var obs=new MutationObserver(function(){bind();var p=el("copilot-panel");if(p&&p.classList.contains("is-open"))scrollChat()});
  setTimeout(function(){bind();var m=el("copilot-messages");if(m)obs.observe(m,{childList:true,subtree:true})},450);
})();
</script>
""" + "</head>")



# ============================================================
# AI COPILOT PRO — FINAL AUTHORITATIVE UI/UX OVERRIDE
# This is intentionally appended last so it wins over legacy Copilot styles.
# ============================================================

app.index_string = app.index_string.replace("</head>", r"""
<style>
/* ---------- GLOBAL RENDER / SHARPNESS ---------- */
html, body, #react-entry-point, .app-root, ._dash-loading { -webkit-font-smoothing: antialiased!important; -moz-osx-font-smoothing: grayscale!important; text-rendering: optimizeLegibility!important; }
.team-member-photo-frame,.team-member-avatar,.team-member-photo,.team-member-jmi-logo{filter:none!important;backdrop-filter:none!important;-webkit-backdrop-filter:none!important;image-rendering:auto!important}
.team-member-photo{transform:none!important;will-change:auto!important;backface-visibility:visible!important;-webkit-backface-visibility:visible!important}
.team-member-photo-frame{transform:none!important;will-change:auto!important}
.team-member-card:hover .team-member-photo{transform:none!important}

/* ---------- AI COPILOT LAUNCHER ---------- */
.copilot-root{position:relative!important;z-index:2140!important;font-family:"Segoe UI",Arial,sans-serif!important}
.copilot-launcher{
  position:fixed!important;right:24px!important;bottom:24px!important;z-index:2145!important;
  width:auto!important;min-width:164px!important;height:52px!important;padding:0 16px 0 12px!important;
  display:inline-flex!important;align-items:center!important;justify-content:flex-start!important;gap:10px!important;
  border:1px solid rgba(143,208,255,.30)!important;border-radius:16px!important;
  background:linear-gradient(180deg,rgba(18,30,40,.96),rgba(8,16,24,.97))!important;
  color:#F4FAFD!important;box-shadow:0 16px 42px rgba(0,0,0,.36),0 0 0 1px rgba(255,255,255,.025) inset!important;
  backdrop-filter:blur(18px) saturate(120%)!important;-webkit-backdrop-filter:blur(18px) saturate(120%)!important;
  cursor:pointer!important;transition:transform .16s ease,border-color .16s ease,box-shadow .16s ease!important;
}
.copilot-launcher:hover{transform:translateY(-1px)!important;border-color:rgba(143,208,255,.50)!important;box-shadow:0 20px 52px rgba(0,0,0,.42),0 0 28px rgba(91,184,235,.09)!important}
.copilot-launcher:focus-visible{outline:2px solid rgba(143,208,255,.75)!important;outline-offset:3px!important}
.copilot-launch-label{font:800 10px/1 "Bahnschrift","Segoe UI",sans-serif!important;letter-spacing:.12em!important;color:#F4FAFD!important}
.copilot-launch-live{font:800 8px/1 "Cascadia Mono",Consolas,monospace!important;letter-spacing:.12em!important;color:#70D19B!important;padding:4px 6px!important;border:1px solid rgba(112,209,155,.20)!important;border-radius:999px!important;background:rgba(112,209,155,.05)!important;animation:none!important}

/* ---------- AI COPILOT PANEL ---------- */
.copilot-panel{
  position:fixed!important;right:24px!important;bottom:88px!important;z-index:2150!important;
  width:min(560px,calc(100vw - 34px))!important;height:min(760px,calc(100vh - 112px))!important;min-height:560px!important;
  display:flex!important;flex-direction:column!important;overflow:hidden!important;
  border:1px solid rgba(143,208,255,.22)!important;border-radius:24px!important;
  background:linear-gradient(180deg,rgba(11,22,31,.985),rgba(6,14,21,.99))!important;
  box-shadow:0 34px 100px rgba(0,0,0,.56),0 0 0 1px rgba(255,255,255,.026) inset,0 0 40px rgba(92,176,228,.07)!important;
  backdrop-filter:blur(24px) saturate(125%)!important;-webkit-backdrop-filter:blur(24px) saturate(125%)!important;
  transform-origin:bottom right!important;
  transition:opacity .14s ease,transform .16s ease,visibility .14s ease!important;
}
.copilot-panel.is-hidden{opacity:0!important;visibility:hidden!important;pointer-events:none!important;transform:translateY(8px) scale(.99)!important}
.copilot-panel.is-open{opacity:1!important;visibility:visible!important;pointer-events:auto!important;transform:none!important}
.copilot-header{min-height:76px!important;padding:14px 16px!important;border-bottom:1px solid rgba(255,255,255,.06)!important;background:linear-gradient(180deg,rgba(255,255,255,.022),rgba(255,255,255,0))!important}
.copilot-heading-copy{gap:11px!important}.copilot-title{font:800 18px/1.05 "Bahnschrift","Segoe UI",sans-serif!important;letter-spacing:.015em!important;color:#F6FBFE!important}.copilot-subtitle{margin-top:5px!important;font:500 11px/1.3 "Segoe UI",Arial,sans-serif!important;color:#8DA6B5!important}.copilot-status{margin-top:5px!important;font:700 8px/1 "Cascadia Mono",Consolas,monospace!important;letter-spacing:.10em!important;color:#79A0B2!important}.copilot-status-dot{color:#6ED19B!important}
.copilot-heading-actions{gap:6px!important}.copilot-icon-btn,.copilot-close-btn{min-width:36px!important;height:36px!important;padding:0 10px!important;border-radius:10px!important;border:1px solid rgba(143,208,255,.13)!important;background:rgba(255,255,255,.025)!important;color:#AFC5D0!important;font:800 12px/1 "Segoe UI",Arial,sans-serif!important;cursor:pointer!important}.copilot-icon-btn:hover,.copilot-close-btn:hover{border-color:rgba(143,208,255,.34)!important;background:rgba(143,208,255,.05)!important;color:#F3FAFD!important}

/* ---------- CONTEXT RIBBON ---------- */
.copilot-context-ribbon{margin:10px 14px 0!important;padding:10px 11px!important;border:1px solid rgba(143,208,255,.11)!important;border-radius:14px!important;background:linear-gradient(180deg,rgba(143,208,255,.035),rgba(255,255,255,.012))!important;box-shadow:inset 0 1px 0 rgba(255,255,255,.03)!important}
.copilot-context-topline{display:flex!important;align-items:center!important;gap:7px!important}.copilot-context-live{font:800 7px/1 "Cascadia Mono",Consolas,monospace!important;letter-spacing:.13em!important;color:#69D39C!important}.copilot-context-kicker{font:800 7px/1 "Cascadia Mono",Consolas,monospace!important;letter-spacing:.13em!important;color:#718D9D!important}.copilot-context-metrics{display:flex!important;flex-wrap:wrap!important;gap:9px!important;margin-top:8px!important}.copilot-context-stat{display:flex!important;align-items:baseline!important;gap:5px!important}.copilot-context-count{font:800 13px/1 "Bahnschrift","Segoe UI",sans-serif!important;color:#E9F4F8!important}.copilot-context-count-alert{color:#F3B66F!important}.copilot-context-muted{font:600 8px/1.2 "Segoe UI",Arial,sans-serif!important;color:#78909E!important}.copilot-context-asof{font:700 8px/1.2 "Cascadia Mono",Consolas,monospace!important;color:#A9C3D0!important}.copilot-context-filters{display:flex!important;flex-wrap:wrap!important;gap:5px!important;margin-top:8px!important}.copilot-context-filter{padding:5px 7px!important;border-radius:8px!important;border:1px solid rgba(143,208,255,.10)!important;background:rgba(255,255,255,.018)!important;color:#91A9B6!important;font:700 8px/1.15 "Cascadia Mono",Consolas,monospace!important}.copilot-context-filter strong{color:#DCEAF1!important}

/* ---------- MESSAGE AREA ---------- */
.copilot-messages{flex:1!important;min-height:0!important;overflow-y:auto!important;padding:16px 14px 12px!important;scrollbar-width:thin!important;scroll-behavior:smooth!important;overscroll-behavior:contain!important}
.copilot-welcome{padding:18px!important;border:1px solid rgba(143,208,255,.10)!important;border-radius:17px!important;background:rgba(255,255,255,.018)!important}.copilot-welcome-title{font:800 24px/1.08 "Bahnschrift","Segoe UI",sans-serif!important;color:#F5FAFC!important}.copilot-welcome-copy{max-width:61ch!important;margin-top:8px!important;font:400 13px/1.62 "Segoe UI",Arial,sans-serif!important;color:#98AEB9!important}.copilot-welcome-kicker{font:800 8px/1 "Cascadia Mono",Consolas,monospace!important;letter-spacing:.16em!important;color:#7AB9D4!important}
.copilot-message{margin:9px 0!important}.copilot-bubble{max-width:92%!important;padding:12px 13px!important;border-radius:15px!important;font:400 13px/1.58 "Segoe UI",Arial,sans-serif!important}.copilot-message-user .copilot-bubble,.copilot-bubble-user{background:#102533!important;border:1px solid rgba(143,208,255,.18)!important;color:#F0F7FA!important}.copilot-message-assistant .copilot-bubble,.copilot-bubble-assistant{background:rgba(255,255,255,.018)!important;border:1px solid rgba(255,255,255,.065)!important;color:#DCE8ED!important}.copilot-role{margin-bottom:6px!important;font:800 7px/1 "Cascadia Mono",Consolas,monospace!important;letter-spacing:.15em!important;color:#6F8B9A!important}.copilot-md{font-size:13px!important;line-height:1.58!important}.copilot-md strong{color:#F0F7FA!important}.copilot-md code{font-family:"Cascadia Mono",Consolas,monospace!important;font-size:.93em!important;color:#9FD7F2!important;background:rgba(143,208,255,.055)!important}.copilot-sources{margin-top:8px!important}.copilot-source-chip{font:700 7px/1 "Cascadia Mono",Consolas,monospace!important;padding:4px 7px!important;border-radius:999px!important;color:#7CA6BA!important;border-color:rgba(143,208,255,.10)!important;background:rgba(143,208,255,.025)!important}.copilot-copy{margin-top:8px!important;padding:4px 8px!important;border-radius:7px!important;border:1px solid rgba(255,255,255,.07)!important;background:transparent!important;color:#6F8795!important;font:700 7px/1 "Cascadia Mono",Consolas,monospace!important;cursor:pointer!important}.copilot-copy:hover{color:#CFE4EC!important;border-color:rgba(143,208,255,.18)!important}

/* ---------- QUICK ACTIONS ---------- */
.copilot-quick-wrap{padding:10px 14px 8px!important;border-top:1px solid rgba(255,255,255,.05)!important}.copilot-section-label{margin-bottom:7px!important;font:800 8px/1 "Cascadia Mono",Consolas,monospace!important;letter-spacing:.15em!important;color:#6E8795!important}.copilot-quick-grid{grid-template-columns:repeat(3,minmax(0,1fr))!important;gap:6px!important}.copilot-quick-btn{min-height:34px!important;padding:8px 9px!important;border-radius:10px!important;border:1px solid rgba(143,208,255,.09)!important;background:rgba(255,255,255,.018)!important;color:#A2BAC5!important;text-align:left!important;font:700 8px/1.18 "Segoe UI",Arial,sans-serif!important;cursor:pointer!important;transition:border-color .14s ease,background .14s ease,color .14s ease!important}.copilot-quick-btn:hover{border-color:rgba(143,208,255,.24)!important;background:rgba(143,208,255,.035)!important;color:#EDF6FA!important;transform:none!important}

/* ---------- COMPOSER ---------- */
.copilot-composer{display:flex!important;align-items:stretch!important;gap:7px!important;padding:10px 14px 10px!important}.copilot-input-wrap{position:relative!important;flex:1!important;display:flex!important}.copilot-input{flex:1!important;min-height:58px!important;max-height:150px!important;resize:vertical!important;padding:12px 13px!important;border-radius:14px!important;border:1px solid rgba(143,208,255,.14)!important;background:#08131C!important;color:#F1F8FB!important;outline:none!important;font:500 13px/1.5 "Segoe UI",Arial,sans-serif!important;box-shadow:none!important}.copilot-input:focus{border-color:rgba(143,208,255,.42)!important;box-shadow:0 0 0 3px rgba(143,208,255,.06)!important}.copilot-input::placeholder{color:#607784!important;opacity:1!important}.copilot-mic{width:42px!important;min-width:42px!important;border-radius:12px!important;border:1px solid rgba(143,208,255,.11)!important;background:rgba(255,255,255,.018)!important;color:#8FA8B5!important;cursor:pointer!important}.copilot-send{width:76px!important;min-width:76px!important;border-radius:13px!important;border:1px solid rgba(143,208,255,.30)!important;background:linear-gradient(180deg,#1D6B8B,#154A62)!important;color:#F4FBFE!important;cursor:pointer!important;display:flex!important;flex-direction:column!important;align-items:center!important;justify-content:center!important;gap:4px!important;box-shadow:0 10px 24px rgba(19,89,120,.18)!important;transition:background .14s ease,border-color .14s ease!important}.copilot-send:hover{background:linear-gradient(180deg,#2481A6,#1A5872)!important;border-color:rgba(143,208,255,.46)!important;transform:none!important}.copilot-send-label{font:900 7px/1 "Cascadia Mono",Consolas,monospace!important;letter-spacing:.13em!important}.copilot-send-icon{font-size:18px!important;line-height:1!important}
.copilot-footer{min-height:28px!important;padding:0 14px 10px!important;background:transparent!important}.copilot-footer-brand{font:700 7px/1 "Cascadia Mono",Consolas,monospace!important;letter-spacing:.06em!important;color:#647D8A!important}

/* ---------- CSS VECTOR AI COPILOT MARK ---------- */
.copilot-logo{position:relative!important;display:inline-block!important;width:38px!important;height:38px!important;flex:0 0 38px!important;isolation:isolate!important}.copilot-logo:before{content:"";position:absolute!important;inset:4px!important;border-radius:11px!important;border:1px solid rgba(153,221,247,.63)!important;background:linear-gradient(145deg,#184B67,#0C2434)!important;box-shadow:inset 0 1px 0 rgba(255,255,255,.26),0 8px 22px rgba(0,0,0,.18),0 0 22px rgba(97,190,229,.10)!important;transform:rotate(45deg)!important}.copilot-logo:after{content:"AI";position:absolute!important;inset:0!important;display:grid!important;place-items:center!important;color:#F6FCFE!important;font:900 10px/1 "Bahnschrift","Segoe UI",sans-serif!important;letter-spacing:.05em!important}.copilot-logo-orbit,.copilot-logo-aura,.copilot-logo-node,.copilot-logo-core,.copilot-logo-glint{display:none!important}.copilot-logo-header{width:42px!important;height:42px!important;flex-basis:42px!important}.copilot-logo-launch{width:32px!important;height:32px!important;flex-basis:32px!important}.copilot-logo-launch:after{font-size:8px!important}.copilot-logo-launch:before{inset:3px!important;border-radius:8px!important}
.copilot-logo-message{width:26px!important;height:26px!important;flex-basis:26px!important}.copilot-logo-message:after{font-size:7px!important}.copilot-logo-message:before{inset:3px!important;border-radius:7px!important}
.copilot-logo-welcome{width:46px!important;height:46px!important;flex-basis:46px!important}

/* ---------- TEAM PHOTO FAILURE-SAFE + NO BLUR ---------- */
.team-member-avatar-photo{position:relative!important;width:100%!important;height:100%!important;overflow:hidden!important;background:#0C1721!important}.team-member-avatar-photo > img.team-member-photo{position:absolute!important;inset:0!important;width:100%!important;height:100%!important;object-fit:cover!important;object-position:50% 22%!important;display:block!important;filter:none!important;image-rendering:auto!important}.team-member-photo-fallback{position:absolute!important;inset:0!important;display:none!important;place-items:center!important;align-content:center!important;gap:8px!important;background:radial-gradient(circle at 34% 24%,rgba(143,208,255,.21),transparent 35%),linear-gradient(150deg,#183248,#0A141D)!important;color:#EEF7FA!important;text-align:center!important}.team-member-photo-fallback.is-visible{display:grid!important}.team-member-photo-fallback-initials{font:900 38px/1 "Bahnschrift","Segoe UI",sans-serif!important;letter-spacing:-.04em!important;color:#F4F9FB!important}.team-member-photo-fallback-label{font:800 7px/1 "Cascadia Mono",Consolas,monospace!important;letter-spacing:.12em!important;color:#7FA6B7!important}.team-member-photo-frame:before,.team-member-photo-frame:after{filter:none!important}.team-member-photo-frame:after{display:none!important}

/* ---------- PREVENT PAGE-WIDE BLUR FROM THE COPILOT ---------- */
.copilot-backdrop{background:rgba(2,8,13,.30)!important;backdrop-filter:none!important;-webkit-backdrop-filter:none!important;opacity:1!important}.copilot-backdrop.is-hidden{opacity:0!important;visibility:hidden!important;pointer-events:none!important}.copilot-backdrop.is-open{opacity:1!important;visibility:visible!important;pointer-events:auto!important}

@media(max-width:850px){
  .copilot-launcher{right:14px!important;bottom:14px!important;width:54px!important;min-width:54px!important;height:54px!important;padding:0!important;justify-content:center!important;border-radius:15px!important}
  .copilot-launch-label,.copilot-launch-live{display:none!important}
  .copilot-panel{right:10px!important;bottom:78px!important;width:calc(100vw - 20px)!important;height:min(780px,calc(100vh - 92px))!important;min-height:520px!important;border-radius:20px!important}
}
@media(max-width:560px){
  .copilot-panel{right:0!important;bottom:0!important;width:100vw!important;height:100dvh!important;max-height:100dvh!important;min-height:0!important;border-radius:0!important}
  .copilot-header{padding:12px!important}.copilot-context-ribbon{margin:8px 9px 7px!important}.copilot-messages{padding:10px 9px 12px!important}.copilot-quick-wrap,.copilot-composer{padding-left:9px!important;padding-right:9px!important}.copilot-md{font-size:12px!important}.copilot-welcome-title{font-size:23px!important}.copilot-send{width:64px!important;min-width:64px!important}.copilot-quick-grid{grid-template-columns:1fr 1fr!important}
}
@media(prefers-reduced-motion:reduce){.copilot-root *,.copilot-root *:before,.copilot-root *:after{animation:none!important;transition:none!important;scroll-behavior:auto!important}}
</style>
<script>
(function(){
  "use strict";
  function byId(id){return document.getElementById(id)}
  function clickId(id){var e=byId(id);if(e){e.click();return true}return false}
  function focusCopilot(){var e=byId("copilot-input");if(e){setTimeout(function(){try{e.focus()}catch(_e){}},90)}}
  function scrollCopilot(){var e=byId("copilot-messages");if(e){requestAnimationFrame(function(){e.scrollTop=e.scrollHeight})}}

  function bindPhoto(img){
    if(!img || img.dataset.copilotPhotoBound==="1") return;
    img.dataset.copilotPhotoBound="1";
    var fallback=img.parentElement?img.parentElement.querySelector(".team-member-photo-fallback"):null;
    function fail(){img.style.display="none";if(fallback)fallback.classList.add("is-visible")}
    function ok(){img.style.display="block";if(fallback)fallback.classList.remove("is-visible")}
    img.addEventListener("error",fail,{once:true});
    if(img.complete && img.naturalWidth>0) ok();
    else if(img.complete && img.naturalWidth===0) fail();
  }
  function bindPhotos(){document.querySelectorAll("img.team-member-photo").forEach(bindPhoto)}

  document.addEventListener("keydown",function(event){
    var key=(event.key||"").toLowerCase();
    var panel=byId("copilot-panel");
    if((event.ctrlKey||event.metaKey)&&key==="k"){
      event.preventDefault();
      if(panel&&panel.classList.contains("is-open")) clickId("copilot-close"); else {clickId("copilot-launcher");focusCopilot()}
      return;
    }
    if(key==="escape"&&panel&&panel.classList.contains("is-open")){event.preventDefault();clickId("copilot-close");return}
    if((key==="enter"||key==="return")&&!event.shiftKey&&document.activeElement&&document.activeElement.id==="copilot-input"){
      event.preventDefault();clickId("copilot-send");
    }
  });

  document.addEventListener("click",function(event){
    var t=event.target.closest&&event.target.closest("#copilot-launcher,#copilot-prompt-scope,#copilot-prompt-compare,#copilot-prompt-risk,#copilot-prompt-financial,#copilot-prompt-chart,#copilot-prompt-methodology,#copilot-welcome-compare,#copilot-welcome-risk,#copilot-welcome-work,#copilot-new");
    if(t){setTimeout(function(){focusCopilot();scrollCopilot()},110)}
    if(event.target.closest&&event.target.closest("#copilot-send")){
      var input=byId("copilot-input");
      if(input&&input.value){try{void 0}catch(_e){}}
    }
  });

  var observer=new MutationObserver(function(){bindPhotos();var p=byId("copilot-panel");if(p&&p.classList.contains("is-open"))scrollCopilot()});
  function init(){bindPhotos();var m=byId("copilot-messages");if(m)observer.observe(m,{childList:true,subtree:true});var root=document.getElementById("react-entry-point");if(root)observer.observe(root,{childList:true,subtree:true})}
  setTimeout(init,450);
})();
</script>
""")

# ============================================================
# FINAL TEAM INFO UX — RESPONSIVE, SHARP, NON-DISTORTING
# ============================================================
app.index_string = app.index_string.replace("</head>", r"""<style>
/* ============================================================
   TEAM INFO — FINAL RESPONSIVE / SHARP PORTRAIT SYSTEM
   ============================================================ */
.team-modal-dialog{
  width:min(1220px,96vw)!important;
  max-height:min(900px,92vh)!important;
  border-radius:30px!important;
  overflow:hidden!important;
}
.team-modal-top{
  min-height:112px!important;
  align-items:center!important;
  padding:24px 28px 20px!important;
}
.team-modal-actions{
  display:flex!important;
  align-items:center!important;
  gap:9px!important;
  flex:0 0 auto!important;
}
.team-modal-github{
  display:inline-flex!important;
  align-items:center!important;
  gap:8px!important;
  min-height:40px!important;
  padding:0 13px!important;
  border-radius:12px!important;
  text-decoration:none!important;
  color:#CFE7F4!important;
  background:linear-gradient(180deg,rgba(28,52,70,.86),rgba(11,21,30,.92))!important;
  border:1px solid rgba(143,208,255,.18)!important;
  font:800 8px/1 "Cascadia Mono",Consolas,monospace!important;
  letter-spacing:.11em!important;
  transition:all .18s ease!important;
}
.team-modal-github:hover{
  color:#FFFFFF!important;
  border-color:rgba(143,208,255,.45)!important;
  box-shadow:0 10px 28px rgba(0,0,0,.28),0 0 22px rgba(143,208,255,.08)!important;
  transform:translateY(-1px)!important;
}
.team-modal-github-icon{font-size:15px!important;line-height:1!important}
.team-modal-scroll{
  max-height:calc(min(900px,92vh) - 112px)!important;
  padding:20px 24px 26px!important;
}
.team-members-grid{
  display:grid!important;
  grid-template-columns:repeat(3,minmax(0,1fr))!important;
  gap:16px!important;
  align-items:stretch!important;
}
.team-member-card{
  display:grid!important;
  grid-template-columns:minmax(148px,42%) minmax(0,1fr)!important;
  gap:16px!important;
  min-height:214px!important;
  height:auto!important;
  padding:12px!important;
  align-items:stretch!important;
  border-radius:22px!important;
}
.team-member-visual{
  position:relative!important;
  min-width:0!important;
  min-height:188px!important;
  align-self:stretch!important;
}
.team-member-photo-frame{
  position:relative!important;
  width:100%!important;
  height:100%!important;
  min-height:188px!important;
  aspect-ratio:4/5!important;
  border-radius:20px!important;
  overflow:hidden!important;
  background:linear-gradient(160deg,#152A3C,#0A121B)!important;
  border:1px solid rgba(143,208,255,.22)!important;
  box-shadow:0 10px 26px rgba(0,0,0,.26),inset 0 1px 0 rgba(255,255,255,.05)!important;
}
.team-member-avatar,
.team-member-avatar-photo{
  width:100%!important;
  height:100%!important;
  min-height:100%!important;
  border:0!important;
  border-radius:18px!important;
  overflow:hidden!important;
  background:transparent!important;
}
.team-member-photo{
  display:block!important;
  width:100%!important;
  height:100%!important;
  max-width:none!important;
  max-height:none!important;
  object-fit:cover!important;
  object-position:50% 25%!important;
  transform:none!important;
  filter:none!important;
  image-rendering:auto!important;
  backface-visibility:visible!important;
  -webkit-backface-visibility:visible!important;
}
.team-member-photo-frame:after{display:none!important}
.team-member-photo-frame:before{
  content:""!important;
  position:absolute!important;
  inset:7px!important;
  z-index:2!important;
  border-radius:14px!important;
  border:1px solid rgba(255,255,255,.07)!important;
  pointer-events:none!important;
}
.team-member-content{
  min-width:0!important;
  display:flex!important;
  flex-direction:column!important;
  justify-content:center!important;
  padding:4px 4px 4px 0!important;
}
.team-member-name-row{
  display:flex!important;
  align-items:flex-start!important;
  justify-content:space-between!important;
  gap:10px!important;
}
.team-member-name{min-width:0!important;font-size:18px!important;line-height:1.08!important}
.team-member-role{
  margin-top:6px!important;
  font-size:8px!important;
  line-height:1.25!important;
  letter-spacing:.12em!important;
  max-width:100%!important;
}
.team-member-degree{margin-top:7px!important;font-size:9.5px!important}
.team-member-bio{
  margin-top:9px!important;
  max-width:none!important;
  color:#8094A1!important;
  font-size:9.5px!important;
  line-height:1.48!important;
}
.team-member-tags{
  display:flex!important;
  gap:5px!important;
  flex-wrap:wrap!important;
  margin-top:auto!important;
  padding-top:11px!important;
}
.team-member-photo-corner{left:9px!important;bottom:9px!important;z-index:5!important}
.team-member-jmi-badge{
  flex:0 0 32px!important;
  width:32px!important;
  height:32px!important;
  min-width:32px!important;
  border-radius:11px!important;
}
.team-capabilities{
  margin-top:18px!important;
  padding:16px 18px!important;
  border-radius:18px!important;
  border:1px solid rgba(143,208,255,.10)!important;
  background:
    linear-gradient(135deg,rgba(143,208,255,.035),rgba(199,146,234,.025)),
    rgba(255,255,255,.012)!important;
}
.team-capabilities-kicker{
  margin-bottom:9px!important;
  color:#6E95A8!important;
  font:800 8px/1 "Cascadia Mono",Consolas,monospace!important;
  letter-spacing:.17em!important;
}
.team-capabilities-row{display:flex!important;flex-wrap:wrap!important;gap:7px!important}
.team-capability-chip{
  display:inline-flex!important;
  align-items:center!important;
  min-height:28px!important;
  padding:0 9px!important;
  border-radius:999px!important;
  border:1px solid rgba(143,208,255,.10)!important;
  color:#9DBBCA!important;
  background:rgba(143,208,255,.025)!important;
  font:800 7px/1 "Cascadia Mono",Consolas,monospace!important;
  letter-spacing:.10em!important;
}
.team-source-row{
  display:flex!important;
  align-items:center!important;
  gap:8px!important;
  margin-top:13px!important;
  padding-top:12px!important;
  border-top:1px solid rgba(255,255,255,.06)!important;
  min-width:0!important;
}
.team-source-label{
  flex:0 0 auto!important;
  color:#566D7B!important;
  font:800 7px/1 "Cascadia Mono",Consolas,monospace!important;
  letter-spacing:.12em!important;
}
.team-source-link{
  min-width:0!important;
  overflow:hidden!important;
  text-overflow:ellipsis!important;
  white-space:nowrap!important;
  color:#80B9D8!important;
  text-decoration:none!important;
  font:600 8px/1.2 "Cascadia Mono",Consolas,monospace!important;
}
.team-source-link:hover{color:#D5F0FF!important}

@media(max-width:1180px){
  .team-members-grid{grid-template-columns:repeat(2,minmax(0,1fr))!important}
}
@media(max-width:760px){
  .team-modal-top{
    align-items:flex-start!important;
    gap:12px!important;
    padding:18px 18px 16px!important;
  }
  .team-modal-heading{gap:11px!important}
  .model-logo-modal{width:52px!important;height:52px!important;flex-basis:52px!important}
  .team-modal-github{display:none!important}
  .team-modal-dialog{width:96vw!important;border-radius:24px!important}
  .team-modal-scroll{padding:15px 14px 18px!important}
  .team-members-grid{grid-template-columns:1fr!important;gap:12px!important}
  .team-member-card{
    grid-template-columns:126px minmax(0,1fr)!important;
    min-height:172px!important;
    padding:10px!important;
    gap:12px!important;
  }
  .team-member-visual,.team-member-photo-frame{min-height:150px!important}
  .team-member-name{font-size:16px!important}
  .team-member-role{font-size:7px!important}
  .team-member-bio{font-size:9px!important}
  .team-source-link{font-size:7px!important}
}


/* ============================================================
   TEAM INFO v11 — FULL-SURFACE / NON-SCROLLING / HD PORTRAITS
   This layer deliberately sits last so it wins over all historical Team Info
   CSS in the file while preserving the rest of the dashboard untouched.
   ============================================================ */
.team-modal{
  position:fixed!important;
  inset:0!important;
  z-index:100000!important;
  display:flex!important;
  align-items:center!important;
  justify-content:center!important;
  padding:18px!important;
  overflow:hidden!important;
  visibility:hidden!important;
  opacity:0!important;
  pointer-events:none!important;
  transition:opacity .22s ease,visibility .22s ease!important;
}
.team-modal.is-open{
  visibility:visible!important;
  opacity:1!important;
  pointer-events:auto!important;
}
.team-modal.is-hidden{
  display:flex!important;
  visibility:hidden!important;
  opacity:0!important;
  pointer-events:none!important;
}
.team-modal-backdrop{
  position:absolute!important;
  inset:0!important;
  background:
    radial-gradient(circle at 18% 22%,rgba(100,208,255,.075),transparent 24%),
    radial-gradient(circle at 84% 76%,rgba(194,121,255,.07),transparent 28%),
    linear-gradient(140deg,rgba(1,8,14,.76),rgba(3,9,16,.88))!important;
  backdrop-filter:blur(8px) saturate(105%)!important;
  -webkit-backdrop-filter:blur(8px) saturate(105%)!important;
}
.team-modal-dialog{
  position:relative!important;
  z-index:2!important;
  width:min(1280px,96vw)!important;
  height:min(860px,94vh)!important;
  max-height:none!important;
  min-height:0!important;
  overflow:hidden!important;
  border-radius:30px!important;
  border:1px solid rgba(150,214,245,.22)!important;
  background:
    radial-gradient(circle at 12% 0%,rgba(126,208,255,.10),transparent 25%),
    radial-gradient(circle at 90% 8%,rgba(196,132,246,.09),transparent 28%),
    linear-gradient(145deg,rgba(13,23,33,.965),rgba(6,12,18,.985))!important;
  box-shadow:
    0 34px 90px rgba(0,0,0,.58),
    0 0 0 1px rgba(255,255,255,.025) inset,
    0 0 90px rgba(94,189,245,.075)!important;
  animation:team-v11-in .34s cubic-bezier(.16,1,.3,1) both!important;
  display:flex!important;
  flex-direction:column!important;
}
.team-modal-dialog:before{
  content:"";
  position:absolute!important;
  inset:0!important;
  pointer-events:none!important;
  background:
    linear-gradient(rgba(149,214,244,.018) 1px,transparent 1px),
    linear-gradient(90deg,rgba(149,214,244,.018) 1px,transparent 1px)!important;
  background-size:28px 28px!important;
  mask-image:linear-gradient(180deg,#000,transparent 92%)!important;
  -webkit-mask-image:linear-gradient(180deg,#000,transparent 92%)!important;
}
.team-modal-top{
  position:relative!important;
  z-index:3!important;
  flex:0 0 112px!important;
  min-height:112px!important;
  display:flex!important;
  align-items:center!important;
  justify-content:space-between!important;
  gap:18px!important;
  padding:18px 24px!important;
  border-bottom:1px solid rgba(150,214,245,.09)!important;
  background:linear-gradient(180deg,rgba(255,255,255,.018),rgba(255,255,255,0))!important;
}
.team-modal-heading{display:flex!important;align-items:center!important;gap:14px!important;min-width:0!important}
.model-logo-modal{width:58px!important;height:58px!important;flex:0 0 58px!important;border-radius:17px!important}
.team-modal-heading-copy{min-width:0!important}
.team-modal-kicker{
  margin-bottom:5px!important;
  color:#83CDED!important;
  font:800 8px/1 "Cascadia Mono",Consolas,monospace!important;
  letter-spacing:.24em!important;
}
.team-modal-title{
  margin:0!important;
  color:#F4F8FB!important;
  font:850 clamp(26px,3vw,38px)/1.02 "Bahnschrift","Segoe UI",sans-serif!important;
  letter-spacing:-.045em!important;
}
.team-modal-subtitle{margin-top:5px!important;color:#7F95A4!important;font:10px/1.3 "Segoe UI",Arial,sans-serif!important}
.team-modal-actions{display:flex!important;align-items:center!important;gap:9px!important;flex:0 0 auto!important}
.team-modal-github{
  display:inline-flex!important;align-items:center!important;gap:7px!important;
  min-height:38px!important;padding:0 12px!important;border-radius:12px!important;
  border:1px solid rgba(150,214,245,.13)!important;
  background:rgba(255,255,255,.025)!important;color:#A9D7ED!important;
  text-decoration:none!important;font:800 7px/1 "Cascadia Mono",Consolas,monospace!important;
  letter-spacing:.13em!important;transition:all .18s ease!important;
}
.team-modal-github:hover{transform:translateY(-1px)!important;border-color:rgba(150,214,245,.34)!important;background:rgba(115,194,237,.07)!important;color:#F2FAFE!important;box-shadow:0 10px 22px rgba(0,0,0,.20)!important}
.team-modal-github-icon{font-size:14px!important;line-height:1!important}
.team-modal-close{
  width:40px!important;height:40px!important;flex:0 0 40px!important;
  border-radius:12px!important;border:1px solid rgba(150,214,245,.13)!important;
  background:rgba(8,16,24,.78)!important;color:#AFC1CC!important;
  font:400 25px/1 "Segoe UI",Arial,sans-serif!important;cursor:pointer!important;
  transition:transform .18s ease,border-color .18s ease,color .18s ease,background .18s ease!important;
}
.team-modal-close:hover{transform:rotate(90deg)!important;border-color:rgba(150,214,245,.40)!important;color:#FFF!important;background:rgba(26,47,62,.86)!important}
.team-modal-scroll{
  position:relative!important;
  z-index:2!important;
  flex:1 1 auto!important;
  max-height:none!important;
  height:auto!important;
  min-height:0!important;
  overflow:visible!important;
  overflow-y:visible!important;
  padding:16px 20px 18px!important;
}
.team-modal-pills{display:flex!important;align-items:center!important;gap:7px!important;margin:0 0 12px!important}
.team-modal-pill{
  display:inline-flex!important;align-items:center!important;height:24px!important;padding:0 9px!important;
  border-radius:999px!important;border:1px solid rgba(143,208,255,.11)!important;
  background:rgba(143,208,255,.025)!important;color:#8BBBD1!important;
  font:800 6.5px/1 "Cascadia Mono",Consolas,monospace!important;letter-spacing:.16em!important;
}
.team-modal-pill-alt{border-color:rgba(194,143,238,.13)!important;background:rgba(194,143,238,.023)!important;color:#BFA4D6!important}
.team-members-grid{
  display:grid!important;
  grid-template-columns:repeat(3,minmax(0,1fr))!important;
  gap:12px!important;
  align-items:stretch!important;
}
.team-member-card{
  position:relative!important;
  display:grid!important;
  grid-template-columns:128px minmax(0,1fr)!important;
  gap:13px!important;
  min-width:0!important;
  min-height:175px!important;
  height:175px!important;
  padding:10px!important;
  border-radius:20px!important;
  border:1px solid rgba(147,190,214,.12)!important;
  background:
    linear-gradient(135deg,rgba(132,207,244,.032),rgba(195,137,244,.022) 48%,rgba(255,255,255,.009)) padding-box,
    linear-gradient(125deg,rgba(113,201,242,.18),rgba(203,150,238,.11),rgba(240,198,117,.06)) border-box!important;
  box-shadow:
    0 12px 28px rgba(0,0,0,.18),
    0 1px 0 rgba(255,255,255,.035) inset!important;
  overflow:hidden!important;
  isolation:isolate!important;
  transition:transform .20s ease,border-color .20s ease,box-shadow .20s ease,background .20s ease!important;
}
.team-member-card:before{
  content:"";position:absolute!important;inset:0!important;pointer-events:none!important;
  background:
    radial-gradient(circle at 0% 0%,rgba(143,208,255,.07),transparent 26%),
    radial-gradient(circle at 100% 100%,rgba(198,146,234,.055),transparent 28%)!important;
  opacity:.8!important;z-index:-1!important;
}
.team-member-card:after{
  content:"";position:absolute!important;top:-50%;left:-35%;width:40%;height:200%!important;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.08),transparent)!important;
  transform:translateX(-120%) rotate(16deg)!important;pointer-events:none!important;
  transition:transform .7s cubic-bezier(.16,1,.3,1)!important;z-index:6!important;
}
.team-member-card:hover:after{transform:translateX(360%) rotate(16deg)!important}
.team-member-card:hover{
  transform:translateY(-3px)!important;
  border-color:rgba(148,213,241,.28)!important;
  box-shadow:0 22px 44px rgba(0,0,0,.28),0 0 26px rgba(101,196,237,.06)!important;
}
.team-member-visual{position:relative!important;min-width:0!important;min-height:0!important;height:100%!important}
.team-member-photo-frame{
  position:relative!important;width:100%!important;height:100%!important;min-height:0!important;
  overflow:hidden!important;border-radius:15px!important;
  border:1px solid rgba(155,216,240,.16)!important;
  background:linear-gradient(155deg,#152A3A,#081119)!important;
  box-shadow:0 10px 24px rgba(0,0,0,.26),inset 0 1px 0 rgba(255,255,255,.05)!important;
}
.team-member-photo-frame:before{
  content:"";position:absolute!important;inset:5px!important;border-radius:11px!important;
  border:1px solid rgba(255,255,255,.065)!important;pointer-events:none!important;z-index:3!important;
}
.team-member-photo-frame:after{
  content:"";position:absolute!important;inset:0!important;pointer-events:none!important;z-index:4!important;
  background:linear-gradient(180deg,rgba(255,255,255,.07),transparent 22%,transparent 70%,rgba(2,8,13,.16))!important;
}
.team-member-avatar,.team-member-avatar-photo{
  width:100%!important;height:100%!important;border:0!important;border-radius:15px!important;
  background:transparent!important;overflow:hidden!important;display:block!important;
}
.team-member-photo{
  width:100%!important;height:100%!important;display:block!important;
  object-fit:cover!important;object-position:50% 19%!important;
  filter:none!important;backdrop-filter:none!important;-webkit-backdrop-filter:none!important;
  transform:none!important;image-rendering:auto!important;
}
.team-member-photo-corner,.team-member-photo-index{display:none!important;visibility:hidden!important}
.team-member-photo-fallback,.team-member-photo-fallback-initials,.team-member-photo-fallback-label{display:none!important;visibility:hidden!important}
.team-member-content{min-width:0!important;display:flex!important;flex-direction:column!important;justify-content:center!important;padding:3px 3px 2px 0!important}
.team-member-name-row{display:flex!important;align-items:center!important;justify-content:space-between!important;gap:8px!important;min-width:0!important}
.team-member-name{margin:0!important;color:#F5F8FB!important;font:820 17px/1.05 "Bahnschrift","Segoe UI",sans-serif!important;letter-spacing:-.025em!important;min-width:0!important}
.team-member-role{margin-top:5px!important;color:#92D0EC!important;font:800 7px/1.25 "Cascadia Mono",Consolas,monospace!important;letter-spacing:.12em!important;text-transform:uppercase!important;max-width:100%!important}
.team-member-degree{margin-top:6px!important;color:#78909F!important;font:600 8.5px/1.25 "Segoe UI",Arial,sans-serif!important}
.team-member-bio{margin-top:8px!important;color:#7F929D!important;font:9px/1.40 "Segoe UI",Arial,sans-serif!important;max-width:none!important}
.team-member-tags{display:flex!important;flex-wrap:wrap!important;gap:5px!important;margin-top:auto!important;padding-top:8px!important}
.team-member-tag{display:inline-flex!important;align-items:center!important;height:18px!important;padding:0 6px!important;border-radius:999px!important;border:1px solid rgba(143,208,255,.09)!important;background:rgba(143,208,255,.022)!important;color:#6D9AAF!important;font:800 5.5px/1 "Cascadia Mono",Consolas,monospace!important;letter-spacing:.10em!important}
.team-member-tag-gold{border-color:rgba(240,198,117,.10)!important;background:rgba(240,198,117,.018)!important;color:#BDA673!important}
.team-member-jmi-badge{position:relative!important;right:auto!important;bottom:auto!important;flex:0 0 29px!important;width:29px!important;height:29px!important;min-width:29px!important;border-radius:10px!important;padding:2px!important;border:1px solid rgba(255,255,255,.20)!important;background:rgba(255,255,255,.97)!important;box-shadow:0 5px 12px rgba(0,0,0,.18)!important;overflow:hidden!important}
.team-member-jmi-logo{width:100%!important;height:100%!important;object-fit:contain!important;object-position:center!important;border-radius:7px!important;background:#fff!important;display:block!important}
.team-member-jmi-badge-fallback{display:flex!important;flex-direction:column!important;justify-content:center!important;align-items:center!important;color:#114B31!important;background:linear-gradient(145deg,#fff,#eef9f2)!important}
.team-member-jmi-fallback-main{font:900 7px/1 "Bahnschrift","Segoe UI",sans-serif!important}.team-member-jmi-fallback-sub{display:none!important}
.team-capabilities{
  margin-top:12px!important;padding:11px 13px!important;border-radius:16px!important;
  border:1px solid rgba(143,208,255,.075)!important;
  background:linear-gradient(135deg,rgba(143,208,255,.025),rgba(198,146,234,.018))!important;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.03)!important;
}
.team-capabilities-kicker{margin-bottom:7px!important;color:#62879A!important;font:800 6.5px/1 "Cascadia Mono",Consolas,monospace!important;letter-spacing:.18em!important}
.team-capabilities-row{display:flex!important;flex-wrap:wrap!important;gap:6px!important}
.team-capability-chip{display:inline-flex!important;align-items:center!important;height:22px!important;padding:0 8px!important;border-radius:999px!important;border:1px solid rgba(143,208,255,.08)!important;background:rgba(143,208,255,.018)!important;color:#8EACBB!important;font:800 5.8px/1 "Cascadia Mono",Consolas,monospace!important;letter-spacing:.10em!important}
.team-source-row{display:flex!important;align-items:center!important;gap:7px!important;margin-top:9px!important;padding-top:8px!important;border-top:1px solid rgba(255,255,255,.05)!important;min-width:0!important}
.team-source-label{flex:0 0 auto!important;color:#506A78!important;font:800 6px/1 "Cascadia Mono",Consolas,monospace!important;letter-spacing:.10em!important}
.team-source-link{min-width:0!important;overflow:hidden!important;text-overflow:ellipsis!important;white-space:nowrap!important;color:#7FB9D4!important;text-decoration:none!important;font:600 7px/1.15 "Cascadia Mono",Consolas,monospace!important}
.team-source-link:hover{color:#E0F6FF!important}
@keyframes team-v11-in{from{opacity:0;transform:translateY(12px) scale(.985)}to{opacity:1;transform:none}}

/* Medium desktop: two columns, still fully contained with no inner scroll. */
@media(max-width:1180px){
  .team-modal-dialog{height:min(890px,95vh)!important}
  .team-members-grid{grid-template-columns:repeat(2,minmax(0,1fr))!important}
  .team-member-card{height:168px!important;min-height:168px!important;grid-template-columns:118px minmax(0,1fr)!important}
  .team-member-name{font-size:16px!important}
  .team-member-bio{font-size:8.7px!important}
}

/* Small screens: two columns rather than a tall one-column scroll stack. */
@media(max-width:760px){
  .team-modal{padding:8px!important}
  .team-modal-dialog{width:97vw!important;height:min(900px,96vh)!important;border-radius:23px!important}
  .team-modal-top{flex-basis:95px!important;min-height:95px!important;padding:14px 15px!important}
  .model-logo-modal{width:48px!important;height:48px!important;flex-basis:48px!important;border-radius:14px!important}
  .team-modal-title{font-size:23px!important}
  .team-modal-subtitle{font-size:8px!important}
  .team-modal-github{display:none!important}
  .team-modal-scroll{padding:12px!important}
  .team-members-grid{grid-template-columns:repeat(2,minmax(0,1fr))!important;gap:9px!important}
  .team-member-card{grid-template-columns:92px minmax(0,1fr)!important;gap:9px!important;height:146px!important;min-height:146px!important;padding:8px!important;border-radius:16px!important}
  .team-member-photo-frame,.team-member-avatar,.team-member-avatar-photo{border-radius:12px!important}
  .team-member-name{font-size:12px!important}
  .team-member-role{font-size:5.5px!important;letter-spacing:.09em!important}
  .team-member-degree{font-size:7px!important}
  .team-member-bio{font-size:7.2px!important;line-height:1.30!important;margin-top:5px!important}
  .team-member-tags{display:none!important}
  .team-capabilities{margin-top:9px!important;padding:9px 10px!important}
}

/* Portrait safety: never show legacy textual placeholders under images. */
.team-member-photo-frame + .team-member-photo-corner,
.team-member-visual .team-member-photo-corner,
.team-member-visual .team-member-photo-index,
.team-member-avatar-caption,
.team-member-avatar-initials{display:none!important}

/* Accessibility/focus states. */
.team-modal-close:focus-visible,
.team-modal-github:focus-visible,
.team-info-btn:focus-visible{
  outline:2px solid rgba(143,208,255,.65)!important;
  outline-offset:3px!important;
}
@media(prefers-reduced-motion:reduce){
  .team-modal-dialog,.team-member-card,.team-member-card:after,.team-modal-github,.team-modal-close{animation:none!important;transition:none!important}
}
</style>
<script>
(function(){
  "use strict";
  function id(x){return document.getElementById(x)}
  function openTeam(){var m=id("team-info-modal");if(m){m.classList.remove("is-hidden");m.classList.add("is-open");document.documentElement.classList.add("team-info-open")}}
  function closeTeam(){var m=id("team-info-modal");if(m){m.classList.remove("is-open");m.classList.add("is-hidden");document.documentElement.classList.remove("team-info-open")}}
  function bindTeamInfo(){
    var open=id("team-info-button"),close=id("team-info-close"),back=id("team-info-backdrop");
    if(open&&!open.dataset.teamInfoBound){open.dataset.teamInfoBound="1";open.addEventListener("click",function(){openTeam()})}
    if(close&&!close.dataset.teamInfoBound){close.dataset.teamInfoBound="1";close.addEventListener("click",function(){closeTeam()})}
    if(back&&!back.dataset.teamInfoBound){back.dataset.teamInfoBound="1";back.addEventListener("click",function(){closeTeam()})}
  }
  document.addEventListener("keydown",function(e){
    if((e.key||"").toLowerCase()==="escape"){
      var m=id("team-info-modal");
      if(m&&m.classList.contains("is-open")){e.preventDefault();closeTeam()}
    }
  });
  var observer=new MutationObserver(bindTeamInfo);
  function init(){bindTeamInfo();var root=id("react-entry-point");if(root)observer.observe(root,{childList:true,subtree:true})}
  if(document.readyState==="loading")document.addEventListener("DOMContentLoaded",function(){setTimeout(init,180)});else setTimeout(init,180);
})();
</script>



/* ============================================================
   TEAM INFO — MICRO CAPABILITY BAR / ACCENT FINISH
   Compact visual treatment: smaller than member cards, same accent,
   with restrained gradient/glass depth and no bottom source row.
   ============================================================ */
.team-capabilities{
  margin-top:9px!important;
  padding:8px 11px!important;
  border-radius:13px!important;
  border:1px solid rgba(132,205,238,.085)!important;
  background:
    linear-gradient(105deg,rgba(110,202,238,.045),rgba(185,134,239,.032) 58%,rgba(240,198,117,.018)),
    rgba(255,255,255,.008)!important;
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,.028),
    0 7px 18px rgba(0,0,0,.10)!important;
}
.team-capabilities-kicker{
  display:flex!important;
  align-items:center!important;
  gap:7px!important;
  margin:0 0 6px!important;
  color:#78AFC5!important;
  font:800 5.7px/1 "Cascadia Mono",Consolas,monospace!important;
  letter-spacing:.18em!important;
  text-transform:uppercase!important;
}
.team-capabilities-kicker:before{
  content:""!important;
  width:16px!important;
  height:1px!important;
  flex:0 0 16px!important;
  background:linear-gradient(90deg,rgba(117,210,244,.62),rgba(192,143,239,.28))!important;
  box-shadow:0 0 8px rgba(117,210,244,.10)!important;
}
.team-capabilities-row{
  display:flex!important;
  align-items:center!important;
  flex-wrap:wrap!important;
  gap:5px!important;
}
.team-capability-chip{
  position:relative!important;
  display:inline-flex!important;
  align-items:center!important;
  justify-content:center!important;
  height:19px!important;
  min-height:19px!important;
  padding:0 7px 0 8px!important;
  border-radius:999px!important;
  border:1px solid rgba(124,199,229,.10)!important;
  background:
    linear-gradient(180deg,rgba(128,206,236,.045),rgba(194,140,238,.018)),
    rgba(255,255,255,.010)!important;
  color:#86B8C9!important;
  font:800 5.15px/1 "Cascadia Mono",Consolas,monospace!important;
  letter-spacing:.085em!important;
  white-space:nowrap!important;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.022)!important;
  transition:transform .16s ease,border-color .16s ease,box-shadow .16s ease,color .16s ease!important;
}
.team-capability-chip:before{
  content:""!important;
  width:3px!important;
  height:3px!important;
  margin-right:5px!important;
  border-radius:50%!important;
  background:#78CBE9!important;
  box-shadow:0 0 6px rgba(120,203,233,.32)!important;
}
.team-capability-chip:nth-child(2){color:#8DAFBF!important;border-color:rgba(121,190,218,.085)!important}
.team-capability-chip:nth-child(2):before{background:#8BBAD0!important}
.team-capability-chip:nth-child(3){color:#B69BCB!important;border-color:rgba(187,140,228,.10)!important}
.team-capability-chip:nth-child(3):before{background:#B88CDD!important}
.team-capability-chip:nth-child(4){color:#97B8A8!important;border-color:rgba(126,186,157,.09)!important}
.team-capability-chip:nth-child(4):before{background:#83C59E!important}
.team-capability-chip:nth-child(5){color:#C0AA7A!important;border-color:rgba(226,190,108,.09)!important}
.team-capability-chip:nth-child(5):before{background:#D7B66B!important}
.team-capability-chip:hover{
  transform:translateY(-1px)!important;
  border-color:rgba(135,209,237,.22)!important;
  color:#CBEAF5!important;
  box-shadow:0 5px 12px rgba(0,0,0,.16),0 0 13px rgba(111,196,231,.055)!important;
}
.team-source-row,
.team-source-label,
.team-source-link{
  display:none!important;
  visibility:hidden!important;
  height:0!important;
  margin:0!important;
  padding:0!important;
  border:0!important;
  overflow:hidden!important;
}
.team-modal-github{
  min-height:31px!important;
  height:31px!important;
  padding:0 9px!important;
  gap:6px!important;
  border-radius:10px!important;
  font-size:6.2px!important;
  letter-spacing:.10em!important;
  color:#9CCEDF!important;
  border-color:rgba(129,203,233,.10)!important;
  background:linear-gradient(180deg,rgba(116,194,225,.040),rgba(190,139,239,.018))!important;
}
.team-modal-github-icon{font-size:11px!important}
.team-modal-github:hover{
  border-color:rgba(131,204,233,.24)!important;
  box-shadow:0 7px 16px rgba(0,0,0,.16),0 0 16px rgba(110,199,230,.055)!important;
}
@media(max-width:760px){
  .team-capabilities{padding:7px 8px!important;margin-top:7px!important}
  .team-capabilities-kicker{font-size:5px!important;margin-bottom:5px!important}
  .team-capability-chip{height:17px!important;min-height:17px!important;padding:0 6px!important;font-size:4.7px!important}
  .team-capability-chip:before{width:2.5px!important;height:2.5px!important;margin-right:4px!important}
}

/* TEAM INFO MICRO-CAPABILITIES — final inline-safe override */
#team-info-modal .team-capabilities{
  display:block!important;
  width:100%!important;
  max-width:100%!important;
  margin:8px 0 0!important;
  padding:7px 9px!important;
  min-height:0!important;
  border-radius:12px!important;
  overflow:hidden!important;
}
#team-info-modal .team-capabilities-kicker{
  font-size:5.2px!important;
  line-height:1!important;
  margin:0 0 6px!important;
  min-height:8px!important;
  height:8px!important;
}
#team-info-modal .team-capabilities-row{
  display:flex!important;
  flex-wrap:wrap!important;
  align-items:center!important;
  gap:4px!important;
}
#team-info-modal .team-capability-chip{
  display:inline-flex!important;
  align-items:center!important;
  justify-content:center!important;
  flex:0 0 auto!important;
  height:18px!important;
  min-height:18px!important;
  padding:0 7px!important;
  border-radius:999px!important;
  border:1px solid color-mix(in srgb, var(--chip-accent) 17%, transparent)!important;
  background:linear-gradient(180deg, color-mix(in srgb, var(--chip-accent) 6%, transparent), rgba(255,255,255,.006))!important;
  color:var(--chip-text)!important;
  font:800 5px/1 "Cascadia Mono",Consolas,monospace!important;
  letter-spacing:.085em!important;
  white-space:nowrap!important;
  box-sizing:border-box!important;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.02)!important;
}
#team-info-modal .team-capability-chip:before{
  content:""!important;
  width:3px!important;
  height:3px!important;
  margin-right:4px!important;
  border-radius:50%!important;
  background:var(--chip-accent)!important;
  box-shadow:0 0 5px color-mix(in srgb, var(--chip-accent) 28%, transparent)!important;
}
#team-info-modal .team-capability-chip:hover{
  transform:translateY(-1px)!important;
  border-color:color-mix(in srgb, var(--chip-accent) 28%, transparent)!important;
  box-shadow:0 4px 11px rgba(0,0,0,.14)!important;
}
#team-info-modal .team-modal-github{
  height:18px!important;
  min-height:18px!important;
  padding:0 7px!important;
  border-radius:999px!important;
  font-size:6px!important;
  letter-spacing:.11em!important;
  line-height:1!important;
}
#team-info-modal .team-modal-github-icon{font-size:8px!important;line-height:1!important}
#team-info-modal .team-modal-github-label{font-size:6px!important;line-height:1!important}
@supports not (color: color-mix(in srgb, red 10%, transparent)){
  #team-info-modal .team-capability-chip{border-color:rgba(124,199,229,.12)!important;background:rgba(124,199,229,.02)!important}
  #team-info-modal .team-capability-chip:nth-child(3){border-color:rgba(187,140,228,.12)!important;background:rgba(187,140,228,.025)!important}
  #team-info-modal .team-capability-chip:nth-child(4){border-color:rgba(126,186,157,.11)!important;background:rgba(126,186,157,.02)!important}
  #team-info-modal .team-capability-chip:nth-child(5){border-color:rgba(226,190,108,.11)!important;background:rgba(226,190,108,.018)!important}
}


/* ============================================================
   TEAM INFO — SIGNATURE CAPABILITY PILLS / FINAL VISUAL PASS
   A compact, premium accent treatment that matches the dashboard
   without competing with the six member cards.
   ============================================================ */
.team-capabilities{
  position:relative!important;
  margin:8px 0 0!important;
  padding:8px 10px 9px!important;
  border-radius:15px!important;
  overflow:hidden!important;
  border:1px solid rgba(133,214,246,.10)!important;
  background:
    radial-gradient(220px 70px at 0% 0%,rgba(103,216,247,.055),transparent 72%),
    radial-gradient(220px 70px at 100% 100%,rgba(191,131,242,.045),transparent 72%),
    linear-gradient(115deg,rgba(255,255,255,.018),rgba(255,255,255,.006))!important;
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,.032),
    inset 0 -1px 0 rgba(0,0,0,.20),
    0 10px 24px rgba(0,0,0,.10)!important;
}
.team-capabilities:after{
  content:""!important;
  position:absolute!important;
  inset:0!important;
  pointer-events:none!important;
  background:linear-gradient(90deg,transparent 0%,rgba(121,211,239,.035) 48%,transparent 100%)!important;
  transform:translateX(-110%)!important;
  animation:teamCapabilitySweep 7s ease-in-out infinite!important;
}
@keyframes teamCapabilitySweep{
  0%,62%,100%{transform:translateX(-110%)}
  77%{transform:translateX(110%)}
}
.team-capabilities-kicker{
  position:relative!important;
  z-index:1!important;
  margin:0 0 7px!important;
  display:flex!important;
  align-items:center!important;
  gap:7px!important;
  color:#8FC9DD!important;
  font:900 7.5px/1 "Cascadia Mono",Consolas,monospace!important;
  letter-spacing:.19em!important;
  text-transform:uppercase!important;
  text-shadow:0 0 9px rgba(109,208,239,.10)!important;
}
.team-capabilities-kicker:before{
  content:""!important;
  width:20px!important;
  height:2px!important;
  border-radius:999px!important;
  background:linear-gradient(90deg,#6ED7F2,rgba(174,133,238,.38))!important;
  box-shadow:0 0 10px rgba(90,205,239,.16)!important;
}
.team-capabilities-row{
  position:relative!important;
  z-index:1!important;
  display:flex!important;
  flex-wrap:wrap!important;
  align-items:center!important;
  gap:6px!important;
}
.team-capability-chip{
  position:relative!important;
  display:inline-flex!important;
  align-items:center!important;
  justify-content:center!important;
  height:23px!important;
  min-height:23px!important;
  padding:0 10px 0 9px!important;
  border-radius:999px!important;
  border:1px solid rgba(138,216,242,.11)!important;
  background:
    linear-gradient(180deg,rgba(255,255,255,.035),rgba(255,255,255,.008)),
    linear-gradient(110deg,rgba(102,211,241,.050),rgba(183,131,235,.020))!important;
  color:#B8DCE7!important;
  font:900 6.2px/1 "Cascadia Mono",Consolas,monospace!important;
  letter-spacing:.12em!important;
  white-space:nowrap!important;
  text-rendering:geometricPrecision!important;
  -webkit-font-smoothing:antialiased!important;
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,.035),
    0 2px 7px rgba(0,0,0,.08)!important;
  transition:transform .18s ease,box-shadow .18s ease,border-color .18s ease,filter .18s ease!important;
}
.team-capability-chip:before{
  content:""!important;
  width:5px!important;
  height:5px!important;
  margin-right:6px!important;
  border-radius:50%!important;
  background:#75D5EF!important;
  box-shadow:0 0 0 2px rgba(117,213,239,.07),0 0 8px rgba(117,213,239,.20)!important;
  flex:0 0 5px!important;
}
.team-capability-chip:nth-child(1){
  color:#A7DFF0!important;
  border-color:rgba(89,210,243,.16)!important;
  background:linear-gradient(115deg,rgba(71,210,244,.095),rgba(255,255,255,.012))!important;
}
.team-capability-chip:nth-child(1):before{background:#52D2F2!important}
.team-capability-chip:nth-child(2){
  color:#B5D2E3!important;
  border-color:rgba(100,170,223,.14)!important;
  background:linear-gradient(115deg,rgba(94,157,221,.085),rgba(255,255,255,.012))!important;
}
.team-capability-chip:nth-child(2):before{background:#80B9E1!important}
.team-capability-chip:nth-child(3){
  color:#D1C0E8!important;
  border-color:rgba(180,126,238,.15)!important;
  background:linear-gradient(115deg,rgba(174,122,236,.085),rgba(255,255,255,.012))!important;
}
.team-capability-chip:nth-child(3):before{background:#B985E7!important}
.team-capability-chip:nth-child(4){
  color:#B9DDCF!important;
  border-color:rgba(116,205,163,.14)!important;
  background:linear-gradient(115deg,rgba(95,194,150,.075),rgba(255,255,255,.012))!important;
}
.team-capability-chip:nth-child(4):before{background:#7CC9A5!important}
.team-capability-chip:nth-child(5){
  color:#E5D2A2!important;
  border-color:rgba(223,184,100,.15)!important;
  background:linear-gradient(115deg,rgba(215,178,89,.075),rgba(255,255,255,.012))!important;
}
.team-capability-chip:nth-child(5):before{background:#DDB76E!important}
.team-capability-chip:hover{
  transform:translateY(-2px)!important;
  filter:brightness(1.08)!important;
  border-color:rgba(157,225,244,.28)!important;
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,.05),
    0 6px 14px rgba(0,0,0,.14),
    0 0 14px rgba(103,205,235,.07)!important;
}
.team-modal-github{
  height:27px!important;
  min-height:27px!important;
  padding:0 8px!important;
  gap:5px!important;
  border-radius:9px!important;
  color:#B4D9E5!important;
  border:1px solid rgba(128,206,233,.12)!important;
  background:linear-gradient(115deg,rgba(88,204,236,.055),rgba(174,126,239,.035))!important;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.03)!important;
}
.team-modal-github-label{
  font-size:6px!important;
  letter-spacing:.12em!important;
}
.team-modal-github:hover{
  border-color:rgba(137,217,239,.27)!important;
  box-shadow:0 6px 14px rgba(0,0,0,.14),0 0 14px rgba(102,204,235,.07)!important;
}
@media(max-width:760px){
  .team-capabilities{padding:7px 8px 8px!important;border-radius:13px!important}
  .team-capabilities-kicker{font-size:6.2px!important;margin-bottom:6px!important}
  .team-capability-chip{height:20px!important;min-height:20px!important;padding:0 8px 0 7px!important;font-size:5.4px!important;letter-spacing:.10em!important}
  .team-capability-chip:before{width:4px!important;height:4px!important;flex-basis:4px!important;margin-right:5px!important}
}

</style>""" + "</head>")

# ============================================================
# FINAL PORTRAIT INTERACTION — WHATSAPP-DP STYLE / ZERO BLUR
# ============================================================
# This layer is intentionally placed BEFORE app.run() so direct
# execution of app.py actually applies every final UI override.
# It does not change analytics, API contracts, filters, or data.
# ============================================================

app.index_string = app.index_string.replace("</head>", r"""
<style>
/* ============================================================
   TEAM PORTRAITS — SHARP, STABLE, CLICKABLE, NEVER BLURRED
   ============================================================ */

/* The picture itself is a passive image surface. Clicking it is
   handled by the lightweight lightbox below, just like a profile
   photo preview rather than a hover zoom effect. */
.team-member-photo,
.team-member-photo-frame img{
  display:block!important;
  width:100%!important;
  height:100%!important;
  max-width:none!important;
  max-height:none!important;
  object-fit:cover!important;
  object-position:50% 24%!important;
  filter:none!important;
  -webkit-filter:none!important;
  transform:none!important;
  -webkit-transform:none!important;
  transition:none!important;
  -webkit-transition:none!important;
  animation:none!important;
  -webkit-animation:none!important;
  will-change:auto!important;
  backface-visibility:visible!important;
  -webkit-backface-visibility:visible!important;
  image-rendering:auto!important;
  pointer-events:none!important;
  user-select:none!important;
  -webkit-user-select:none!important;
  -webkit-user-drag:none!important;
}

/* Absolutely no hover/active/focus scaling of the photo itself. */
.team-member-card:hover .team-member-photo,
.team-member-card:active .team-member-photo,
.team-member-card:focus .team-member-photo,
.team-member-card:focus-within .team-member-photo{
  transform:none!important;
  -webkit-transform:none!important;
  filter:none!important;
  -webkit-filter:none!important;
  transition:none!important;
}

/* Keep the frame crisp too. No GPU transform and no local backdrop blur. */
.team-member-photo-frame,
.team-member-avatar,
.team-member-avatar-photo{
  transform:none!important;
  -webkit-transform:none!important;
  filter:none!important;
  -webkit-filter:none!important;
  backdrop-filter:none!important;
  -webkit-backdrop-filter:none!important;
  will-change:auto!important;
}

/* The old identity pill used to blur its own backing glass. Remove that
   effect because it sits on top of the portrait. */
.team-member-photo-corner{
  backdrop-filter:none!important;
  -webkit-backdrop-filter:none!important;
  filter:none!important;
}

/* Make the photo feel clickable without modifying its pixels. */
.team-member-photo-frame{
  cursor:zoom-in!important;
}
.team-member-photo-frame:hover{
  border-color:rgba(143,208,255,.36)!important;
  box-shadow:
    0 12px 30px rgba(0,0,0,.30),
    0 0 0 4px rgba(143,208,255,.035)!important;
}

/* ============================================================
   FULL-SCREEN PORTRAIT PREVIEW — WHATSAPP-DP STYLE
   ============================================================ */
.team-photo-lightbox{
  position:fixed!important;
  inset:0!important;
  z-index:2147483000!important;
  display:flex!important;
  align-items:center!important;
  justify-content:center!important;
  padding:28px!important;
  visibility:hidden!important;
  opacity:0!important;
  pointer-events:none!important;
  transition:opacity .18s ease,visibility .18s ease!important;
}

.team-photo-lightbox.is-open{
  visibility:visible!important;
  opacity:1!important;
  pointer-events:auto!important;
}

.team-photo-lightbox-backdrop{
  position:absolute!important;
  inset:0!important;
  background:
    radial-gradient(circle at 50% 38%,rgba(105,194,244,.09),transparent 30%),
    rgba(2,7,12,.86)!important;
  backdrop-filter:blur(7px) saturate(108%)!important;
  -webkit-backdrop-filter:blur(7px) saturate(108%)!important;
}

.team-photo-lightbox-panel{
  position:relative!important;
  z-index:2!important;
  width:min(760px,92vw)!important;
  height:min(820px,90vh)!important;
  display:flex!important;
  flex-direction:column!important;
  align-items:center!important;
  justify-content:center!important;
  gap:14px!important;
  padding:24px!important;
  border-radius:28px!important;
  border:1px solid rgba(157,216,245,.20)!important;
  background:
    radial-gradient(circle at 20% 0%,rgba(122,207,255,.07),transparent 32%),
    linear-gradient(145deg,rgba(15,26,37,.92),rgba(7,13,20,.96))!important;
  box-shadow:
    0 38px 110px rgba(0,0,0,.62),
    0 0 0 1px rgba(255,255,255,.025) inset!important;
}

.team-photo-lightbox-image{
  display:block!important;
  width:auto!important;
  height:auto!important;
  max-width:min(720px,84vw)!important;
  max-height:74vh!important;
  object-fit:contain!important;
  object-position:center!important;
  border-radius:24px!important;
  filter:none!important;
  -webkit-filter:none!important;
  transform:none!important;
  -webkit-transform:none!important;
  animation:none!important;
  transition:none!important;
  image-rendering:auto!important;
  backface-visibility:visible!important;
  -webkit-backface-visibility:visible!important;
  box-shadow:
    0 20px 60px rgba(0,0,0,.45),
    0 0 0 1px rgba(255,255,255,.08)!important;
}

.team-photo-lightbox-name{
  color:#EEF7FB!important;
  font:800 14px/1.2 "Segoe UI",Arial,sans-serif!important;
  letter-spacing:-.01em!important;
  text-align:center!important;
}

.team-photo-lightbox-close{
  position:absolute!important;
  top:14px!important;
  right:14px!important;
  width:42px!important;
  height:42px!important;
  display:grid!important;
  place-items:center!important;
  border-radius:14px!important;
  border:1px solid rgba(157,216,245,.16)!important;
  background:rgba(9,19,28,.86)!important;
  color:#C1D2DC!important;
  cursor:pointer!important;
  font:400 25px/1 "Segoe UI",Arial,sans-serif!important;
  transition:all .16s ease!important;
}
.team-photo-lightbox-close:hover{
  color:#FFFFFF!important;
  border-color:rgba(157,216,245,.40)!important;
  background:rgba(20,39,53,.94)!important;
  transform:translateY(-1px)!important;
}

.team-photo-lightbox-hint{
  color:#718795!important;
  font:700 7px/1.2 "Cascadia Mono",Consolas,monospace!important;
  letter-spacing:.12em!important;
  text-transform:uppercase!important;
  text-align:center!important;
}

@media(max-width:700px){
  .team-photo-lightbox{
    padding:14px!important;
  }
  .team-photo-lightbox-panel{
    width:96vw!important;
    height:min(760px,92vh)!important;
    padding:16px!important;
    border-radius:22px!important;
  }
  .team-photo-lightbox-image{
    max-width:88vw!important;
    max-height:76vh!important;
    border-radius:20px!important;
  }
}

@media(prefers-reduced-motion:reduce){
  .team-photo-lightbox,
  .team-photo-lightbox-close{
    transition:none!important;
  }
}
</style>

<script>
(function(){
  "use strict";

  var LIGHTBOX_ID = "team-photo-lightbox";

  function getLightbox(){
    return document.getElementById(LIGHTBOX_ID);
  }

  function closePhotoPreview(){
    var box = getLightbox();
    if(!box) return;
    box.classList.remove("is-open");
    document.documentElement.classList.remove("team-photo-preview-open");
    window.setTimeout(function(){
      if(box.parentNode) box.parentNode.removeChild(box);
    }, 190);
  }

  function openPhotoPreview(img){
    if(!img) return;

    var existing = getLightbox();
    if(existing) existing.remove();

    var src = img.currentSrc || img.src || "";
    if(!src) return;

    var alt = img.alt || "Team member portrait";
    var panel = document.createElement("div");
    panel.id = LIGHTBOX_ID;
    panel.className = "team-photo-lightbox";

    var backdrop = document.createElement("div");
    backdrop.className = "team-photo-lightbox-backdrop";
    backdrop.setAttribute("aria-hidden","true");

    var card = document.createElement("div");
    card.className = "team-photo-lightbox-panel";
    card.setAttribute("role","dialog");
    card.setAttribute("aria-modal","true");
    card.setAttribute("aria-label", alt);

    var close = document.createElement("button");
    close.type = "button";
    close.className = "team-photo-lightbox-close";
    close.textContent = "×";
    close.setAttribute("aria-label","Close portrait preview");
    close.title = "Close";

    var image = document.createElement("img");
    image.className = "team-photo-lightbox-image";
    image.src = src;
    image.alt = alt;
    image.draggable = false;

    var name = document.createElement("div");
    name.className = "team-photo-lightbox-name";
    name.textContent = alt.replace(/\s+portrait$/i,"");

    var hint = document.createElement("div");
    hint.className = "team-photo-lightbox-hint";
    hint.textContent = "Click outside or press ESC to close";

    close.addEventListener("click", function(event){
      event.preventDefault();
      event.stopPropagation();
      closePhotoPreview();
    });

    backdrop.addEventListener("click", function(){
      closePhotoPreview();
    });

    card.addEventListener("click", function(event){
      event.stopPropagation();
    });

    card.appendChild(close);
    card.appendChild(image);
    card.appendChild(name);
    card.appendChild(hint);
    panel.appendChild(backdrop);
    panel.appendChild(card);
    document.body.appendChild(panel);

    document.documentElement.classList.add("team-photo-preview-open");

    requestAnimationFrame(function(){
      requestAnimationFrame(function(){
        panel.classList.add("is-open");
      });
    });

    close.focus({preventScroll:true});
  }

  document.addEventListener("click", function(event){
    var img = event.target && event.target.closest
      ? event.target.closest("img.team-member-photo")
      : null;

    if(!img) return;

    event.preventDefault();
    event.stopPropagation();
    openPhotoPreview(img);
  }, true);

  document.addEventListener("keydown", function(event){
    var box = getLightbox();
    if(!box || !box.classList.contains("is-open")) return;

    var key = (event.key || "").toLowerCase();

    if(key === "escape"){
      event.preventDefault();
      closePhotoPreview();
    }
  });

})();
</script>
""" + "</head>")

# ============================================================
# FINAL TEAM PORTRAIT UX PATCH — ZERO-BLUR INTERACTION
# ============================================================
# Design rule:
#   Hover/cursor interaction MUST affect only the portrait FRAME.
#   The actual portrait bitmap never scales, transforms, filters,
#   or enters a GPU-composited transition state.
#
# Result:
#   • Cursor over a portrait -> crisp image + subtle frame glow.
#   • Click portrait -> existing WhatsApp-style full preview.
#   • No image blur, no zoom-resampling, no transform interpolation.
#   • No feature removal; this is a visual interaction-only patch.
# ============================================================

app.index_string = app.index_string.replace("</head>", r"""
<style>

/* ------------------------------------------------------------
   1. HARD SHARPNESS CONTRACT
   ------------------------------------------------------------ */

.team-member-photo,
.team-member-photo:hover,
.team-member-photo:focus,
.team-member-photo:active,
.team-member-photo-frame img,
.team-member-photo-frame img:hover,
.team-member-avatar-photo img{
  display:block!important;
  width:100%!important;
  height:100%!important;
  max-width:none!important;
  max-height:none!important;
  object-fit:cover!important;
  object-position:50% 25%!important;

  /* Never resize the bitmap through CSS transforms. */
  transform:none!important;
  -webkit-transform:none!important;
  scale:none!important;
  rotate:none!important;
  translate:none!important;

  /* Never apply visual filters to the portrait. */
  filter:none!important;
  -webkit-filter:none!important;
  backdrop-filter:none!important;
  -webkit-backdrop-filter:none!important;

  /* Avoid browser animation/compositing blur. */
  animation:none!important;
  transition:none!important;
  will-change:auto!important;
  backface-visibility:visible!important;
  -webkit-backface-visibility:visible!important;
  perspective:none!important;
  image-rendering:auto!important;

  user-select:none!important;
  -webkit-user-select:none!important;
  -webkit-user-drag:none!important;

  cursor:zoom-in!important;
}

/* The frame itself also stays on the normal paint path. */
.team-member-photo-frame,
.team-member-photo-frame:hover,
.team-member-photo-frame:focus,
.team-member-photo-frame:active,
.team-member-avatar,
.team-member-avatar-photo{
  transform:none!important;
  -webkit-transform:none!important;
  filter:none!important;
  -webkit-filter:none!important;
  backdrop-filter:none!important;
  -webkit-backdrop-filter:none!important;
  will-change:auto!important;
  perspective:none!important;
}

/* ------------------------------------------------------------
   2. REMOVE THE SOURCE OF THE BLUR
   ------------------------------------------------------------ */

/*
   Earlier visual layers used:
     .team-member-card:hover .team-member-photo {
         transform: scale(...)
     }
   and:
     .team-member-photo { transform: translateZ(0) ... }

   Both are intentionally neutralised here.
*/
.team-member-card,
.team-member-card:hover,
.team-member-card:focus-within,
.team-member-card:active{
  transform:none!important;
  -webkit-transform:none!important;
  filter:none!important;
  -webkit-filter:none!important;
  backdrop-filter:none!important;
  -webkit-backdrop-filter:none!important;
  will-change:auto!important;
}

/* ------------------------------------------------------------
   3. INTERACTIVE FEEL — FRAME ONLY, NEVER THE PHOTO
   ------------------------------------------------------------ */

.team-member-card{
  position:relative!important;
  transition:
    border-color .20s ease,
    box-shadow .20s ease,
    background-color .20s ease!important;
}

.team-member-card:hover{
  border-color:rgba(143,208,255,.42)!important;
  box-shadow:
    0 18px 40px rgba(0,0,0,.30),
    0 0 0 1px rgba(143,208,255,.05) inset,
    0 0 28px rgba(126,200,255,.075)!important;
}

.team-member-photo-frame{
  position:relative!important;
  isolation:isolate!important;
  overflow:hidden!important;
  transition:
    border-color .20s ease,
    box-shadow .20s ease,
    background-color .20s ease!important;
}

.team-member-photo-frame:hover{
  border-color:rgba(143,208,255,.46)!important;
  box-shadow:
    0 0 0 4px rgba(143,208,255,.035),
    0 14px 30px rgba(0,0,0,.30),
    0 0 26px rgba(105,194,244,.08)!important;
}

/* Elegant light sweep without touching/filtering the photo pixels. */
.team-member-photo-frame:after{
  content:""!important;
  display:block!important;
  position:absolute!important;
  inset:0!important;
  z-index:2!important;
  pointer-events:none!important;
  border-radius:inherit!important;
  background:
    linear-gradient(
      135deg,
      transparent 0%,
      transparent 42%,
      rgba(255,255,255,.075) 50%,
      transparent 58%,
      transparent 100%
    )!important;
  opacity:0!important;
  transform:none!important;
  filter:none!important;
  backdrop-filter:none!important;
  -webkit-backdrop-filter:none!important;
  transition:opacity .20s ease!important;
}

.team-member-photo-frame:hover:after{
  opacity:1!important;
}

/* Inner frame line gives interaction feedback instead of zooming. */
.team-member-photo-frame:before{
  content:""!important;
  position:absolute!important;
  inset:7px!important;
  z-index:3!important;
  pointer-events:none!important;
  border-radius:inherit!important;
  border:1px solid rgba(255,255,255,.075)!important;
  box-shadow:
    inset 0 0 0 1px rgba(143,208,255,.015),
    inset 0 0 34px rgba(143,208,255,.025)!important;
}

/* ------------------------------------------------------------
   4. WHATSAPP-DP STYLE CLICK AFFORDANCE
   ------------------------------------------------------------ */

.team-member-photo-frame{
  cursor:zoom-in!important;
}

.team-member-photo-frame:active{
  border-color:rgba(143,208,255,.62)!important;
  box-shadow:
    0 0 0 5px rgba(143,208,255,.055),
    0 12px 26px rgba(0,0,0,.28),
    0 0 30px rgba(105,194,244,.10)!important;
}

.team-member-photo-frame:active .team-member-photo{
  transform:none!important;
  filter:none!important;
  transition:none!important;
}

/* The little initials badge stays crisp and does NOT blur its background. */
.team-member-photo-corner{
  z-index:5!important;
  pointer-events:none!important;
  backdrop-filter:none!important;
  -webkit-backdrop-filter:none!important;
  filter:none!important;
  -webkit-filter:none!important;
  background:rgba(5,12,18,.90)!important;
  box-shadow:
    0 8px 18px rgba(0,0,0,.22),
    inset 0 1px 0 rgba(255,255,255,.04)!important;
}

/* ------------------------------------------------------------
   5. PRESERVE RESPONSIVENESS / ACCESSIBILITY
   ------------------------------------------------------------ */

.team-member-photo:focus-visible,
.team-member-photo-frame:focus-visible{
  outline:2px solid rgba(143,208,255,.65)!important;
  outline-offset:3px!important;
}

@media (hover:none){
  .team-member-card:hover{
    border-color:rgba(143,208,255,.14)!important;
    box-shadow:0 16px 38px rgba(0,0,0,.22)!important;
  }
  .team-member-photo-frame:hover{
    border-color:rgba(143,208,255,.22)!important;
    box-shadow:0 10px 26px rgba(0,0,0,.26)!important;
  }
  .team-member-photo-frame:hover:after{
    opacity:0!important;
  }
}

@media (prefers-reduced-motion:reduce){
  .team-member-card,
  .team-member-photo-frame,
  .team-member-photo-frame:after{
    transition:none!important;
    animation:none!important;
  }
}

</style>
""" + "</head>")

if __name__ == "__main__":
    print("=" * 72)
    print("MPLADS AI MONITOR - DASH")
    print(f"Dashboard : http://{DASH_HOST}:{DASH_PORT}")
    print(f"FastAPI   : {API_BASE}")
    print(f"Version   : {APP_VERSION}")
    print("AI COPILOT: backend-grounded - current-context synchronised")
    print("Team photos: first_lastname.PNG / JPG / JPEG / WEBP")
    print("=" * 72)

    app.run(
        host=DASH_HOST,
        port=DASH_PORT,
        debug=False,
        dev_tools_hot_reload=False,
    )
