#!/usr/bin/env python3

import argparse
import csv
import html as html_lib
import json
import logging
import os
import re
import sys
import time
import random
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

import cv2
import numpy as np
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv


# Loads GEMINI_API_KEY from a .env file placed next to this script
# (see .env.example). Logo removal auth is NOT read from .env — it uses the
# local token.json / client_secret.json files configured below instead.
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_driver = None

# Set from --remove-logo in main(); read by _download_and_convert_images()
REMOVE_LOGO = False

# Set from --max-images in main(); caps how many gallery photos get scraped per model
MAX_IMAGES = 5

# Set from --logo-delay in main(); seconds to wait between Vertex AI logo-removal
# calls so images don't get fired at the API back-to-back.
LOGO_REMOVAL_DELAY = 15.0

# ============================================================================
# Content rephrasing config (Gemini via GEMINI_API_KEY in .env)
# ============================================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_REPHRASE_MODEL = os.getenv("GEMINI_REPHRASE_MODEL", "gemini-2.5-flash")

# ============================================================================
# Logo removal config (Vertex AI, authenticated via local OAuth files —
# NOT via .env). Point these at the files sitting in your project directory.
# ============================================================================
LOGO_TOKEN_FILE = "token.json"                 # saved/refreshed OAuth token
LOGO_CLIENT_SECRET_FILE = "client_secret.json" # downloaded from Google Cloud Console
LOGO_MASK_FILE = "mask.png"                    # marks where the source logo sits
LOGO_MASK2_FILE = "mask2.png"
VERTEX_PROJECT_ID = "content-engine-bot"
VERTEX_LOCATION = "us-central1"
VERTEX_IMAGE_MODEL = "gemini-2.5-flash-image"
LOGO_REMOVE_PROMPT = (
"Remove the logo inside the masked area. Blend the empty space seamlessly with the surrounding sky and background."
)


# ============================================================================
# Content rephrasing (Gemini) — rewrites post_content + whats-new lists so
# republished content reads differently from the source article.
# ============================================================================
_gemini_client = None


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to a .env file next to "
                "scrapper.py (see .env.example)."
            )
        from google import genai
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini_client


def _rephrase_response_schema():
    from google.genai import types
    item_schema = lambda field: types.Schema(
        type=types.Type.ARRAY,
        items=types.Schema(type=types.Type.OBJECT, properties={field: types.Schema(type=types.Type.STRING)}),
    )
    return types.Schema(
        type=types.Type.OBJECT,
        properties={
            "post_content": types.Schema(type=types.Type.STRING),
            "introduction_list": item_schema("introduction_item"),
            "exterior_list": item_schema("exterior_item"),
            "interior_list": item_schema("interior_item"),
            "powertrain_list": item_schema("powertrain_item"),
        },
    )


def rephrase_model_dict(data: dict, retries: int = 2, retry_delay: float = 2.0) -> dict:
    """Rephrase post_content + acf_whats_new lists of one ACF-mapped model dict, in place."""
    from google.genai import types

    whats_new = data.get("acf_whats_new", {}) or {}
    content_to_rephrase = {
        "post_content": data.get("post_content", ""),
        "introduction_list": whats_new.get("introduction_list", []),
        "exterior_list": whats_new.get("exterior_list", []),
        "interior_list": whats_new.get("interior_list", []),
        "powertrain_list": whats_new.get("powertrain_list", []),
    }
    has_content = content_to_rephrase["post_content"] or any(
        content_to_rephrase[k]
        for k in ("introduction_list", "exterior_list", "interior_list", "powertrain_list")
    )
    if not has_content:
        return data

    client = _get_gemini_client()
    prompt = (
        "Rephrase this content cleanly in your own words. Preserve every fact, "
        "spec number and meaning exactly — only change the wording/sentence "
        "structure. Keep lists formatted as requested:\n\n"
        + json.dumps(content_to_rephrase)
    )
    label = data.get("post_title") or data.get("model") or data.get("url", "?")

    for attempt in range(1, retries + 2):
        try:
            response = client.models.generate_content(
                model=GEMINI_REPHRASE_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=_rephrase_response_schema(),
                    temperature=0.4,
                ),
            )
            rephrased = json.loads(response.text)
            data["post_content"] = rephrased.get("post_content", data.get("post_content", ""))
            data.setdefault("acf_whats_new", {})
            for key in ("introduction_list", "exterior_list", "interior_list", "powertrain_list"):
                if key in rephrased:
                    data["acf_whats_new"][key] = rephrased[key]
            log.info("  [rephrase] done: %s", label)
            return data
        except Exception as exc:
            if attempt <= retries:
                time.sleep(retry_delay)
                continue
            log.warning("  [rephrase] failed for %s: %s", label, exc)
            return data
    return data


def rephrase_models(models: list, delay_between: float = 1.0) -> list:
    """Rephrase a whole list of ACF-mapped model dicts, in place."""
    total = len(models)
    for i, m in enumerate(models, 1):
        log.info("  [rephrase] %d/%d", i, total)
        rephrase_model_dict(m)
        if i < total:
            time.sleep(delay_between)
    return models


# ============================================================================
# Logo removal (Vertex AI image editing) — strips the source site's logo /
# watermark from each downloaded gallery photo before it's saved locally.
# Auth uses a local OAuth token.json + client_secret.json (installed-app
# flow), NOT .env / service-account credentials.
# ============================================================================
_logo_creds = None
_logo_mask_bytes = None
_LOGO_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


def _load_logo_client_info():
    """Read client_id/client_secret/token_uri out of client_secret.json (either
    the 'installed' or 'web' style Google Cloud Console download)."""
    if not os.path.exists(LOGO_CLIENT_SECRET_FILE):
        return {}
    with open(LOGO_CLIENT_SECRET_FILE) as f:
        cs = json.load(f)
    section = cs.get("installed") or cs.get("web") or {}
    return {
        "client_id": section.get("client_id"),
        "client_secret": section.get("client_secret"),
        "token_uri": section.get("token_uri", "https://oauth2.googleapis.com/token"),
    }
import google.auth
from google.auth.transport.requests import Request
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/Anshul/Downloads/projects/Scrappers/fyndmycar_scrapper/service-account.json"
def get_vertex_auth_client():
    # Automatically grabs ADC or Service Account from environment
    creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )

    # Force a refresh to guarantee a valid access token
    creds.refresh(Request())

    return creds

def _get_logo_creds():
    """Load/refresh OAuth creds from token.json, running the installed-app
    flow against client_secret.json the first time (or whenever the token is
    missing/invalid and has no refresh token).

    token.json can be in either the Python google-auth format
    (token/scopes list/expiry datetime) or the Node.js googleapis-library
    format (access_token/scope space-string/expiry_date epoch-ms) — both are
    normalized here instead of relying on Credentials.from_authorized_user_file,
    which only accepts the Python format and rejects everything else outright.
    """
    global _logo_creds
    # Automatically grabs ADC or Service Account from environment
    _logo_creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )

    # Force a refresh to guarantee a valid access token
    _logo_creds.refresh(Request())

    return _logo_creds


def _get_logo_mask_bytes(_logo_mask_file: str) -> bytes:
    global _logo_mask_bytes
    if _logo_mask_bytes is None:
        mask_path = Path(_logo_mask_file)
        if not mask_path.exists():
            raise FileNotFoundError(
                f"Logo mask not found at {mask_path.resolve()} — set LOGO_MASK_FILE "
                "at the top of scrapper.py or place mask.png next to it."
            )
        _logo_mask_bytes = mask_path.read_bytes()
    return _logo_mask_bytes


def remove_logo_from_bytes(image_bytes: bytes, retries: int = 2, timeout: int = 60) -> bytes:
    """Send an image + the configured mask to Vertex AI's Gemini image model
    and return the logo-removed image bytes (PNG). Raises on unrecoverable
    failure — caller falls back to the original bytes."""
    import base64
    nparr = np.frombuffer(image_bytes, np.uint8) 
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    template1 = cv2.imread(LOGO_MASK_FILE)
    template2 = cv2.imread(LOGO_MASK2_FILE)
    mask_image = LOGO_MASK_FILE
    res1 = cv2.matchTemplate(img, template1, cv2.TM_CCOEFF_NORMED)
    _, max_val1, _, max_loc1 = cv2.minMaxLoc(res1)

    # Scan for mask 2 template
    res2 = cv2.matchTemplate(img, template2, cv2.TM_CCOEFF_NORMED)
    _, max_val2, _, max_loc2 = cv2.minMaxLoc(res2)
    # Set a strict threshold (e.g., 80% match)
    THRESHOLD = 0.5

    # Pick the one that actually matches
    if max_val1 > THRESHOLD and max_val1 > max_val2:
        mask_image = LOGO_MASK_FILE
        # Use max_loc1 and template1 dimensions
    elif max_val2 > THRESHOLD:
        mask_image = LOGO_MASK2_FILE

    mask_bytes = _get_logo_mask_bytes(mask_image)

    url = (
        f"https://{VERTEX_LOCATION}-aiplatform.googleapis.com/v1/projects/gen-lang-client-0498856769"
        f"/locations/{VERTEX_LOCATION}/publishers/google/models/{VERTEX_IMAGE_MODEL}:generateContent"
    )
    payload = {
        "contents": [{
            "role": "user",
            "parts": [
                {"inlineData": {"data": base64.b64encode(image_bytes).decode(), "mimeType": "image/png"}},
                {"inlineData": {"data": base64.b64encode(mask_bytes).decode(), "mimeType": "image/png"}},
                {"text": LOGO_REMOVE_PROMPT},
            ],
        }],
        "generationConfig": {
            "temperature": 0.1,
            "topP": 0.95,
            "imageConfig": {"aspectRatio": "16:9"},
        },
    }

    last_exc = None
    for attempt in range(1, retries + 2):
        try:
            creds = get_vertex_auth_client()
            
            access_token = creds.token
            print(access_token)

            headers = {
                "Authorization": f"Bearer {creds.token}",
                "Content-Type": "application/json",
            }
            # time.sleep(30)

            resp = requests.post(url, headers=headers, json=payload)
            if not resp.ok:
                # requests' default HTTPError only includes the URL, not Google's
                # actual error body — that body is where the real reason lives.
                body = resp.text[:800]
                raise RuntimeError(f"{resp.status_code} {resp.reason} — response body: {body}")
            ai_response = resp.json()
            print(ai_response)
            out_b64 = ai_response["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
            return base64.b64decode(out_b64)
        except Exception as exc:
            last_exc = exc
            log.debug("    Logo removal attempt %d/%d failed: %s", attempt, retries + 1, exc)
            time.sleep(1.5)
    raise RuntimeError(f"Logo removal failed after {retries + 1} attempts: {last_exc}")


def _init_selenium():
    global _driver
    if _driver:
        return _driver
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager

        opts = Options()
        opts.add_argument("--headless=new")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--disable-blink-features=AutomationControlled")
        opts.add_argument("--window-size=1920,1080")
        opts.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
        opts.add_experimental_option("excludeSwitches", ["enable-automation"])
        opts.add_experimental_option("useAutomationExtension", False)

        service = Service(ChromeDriverManager().install())
        _driver = webdriver.Chrome(service=service, options=opts)
        _driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
        )
        log.info("Selenium Chrome driver initialised")
        return _driver
    except Exception as exc:
        log.error("Selenium init failed: %s", exc)
        log.error("Install: pip install selenium webdriver-manager")
        sys.exit(1)


def _selenium_fetch(url: str, wait: float = 3.0) -> tuple[BeautifulSoup | None, str]:
    driver = _init_selenium()
    try:
        driver.get(url)
        time.sleep(wait)   # wait for JS to render
        html = driver.page_source
        return BeautifulSoup(html, "lxml"), html
    except Exception as exc:
        log.warning("Selenium fetch error: %s", exc)
        return None, ""


BASE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-IN,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.carandbike.com/",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
}

SESSION = requests.Session()
SESSION.headers.update(BASE_HEADERS)

# Auth token cache — populated once per scrape run from the page's JS bundle
_AUTH_TOKEN: str | None = None


def _extract_auth_token(html: str) -> str | None:
    """
    Extract the API auth token from the page HTML.
    CarAndBike embeds the token in the Next.js runtime config or inline JS.
    Covers all common patterns found in Next.js apps.
    """
    # Pattern 1: __NEXT_DATA__ runtimeConfig or publicRuntimeConfig
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.DOTALL)
    if m:
        try:
            nd = json.loads(m.group(1))
            # Search recursively for token-like keys
            def _deep_find(obj, keys, depth=0):
                if depth > 8 or not obj:
                    return None
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if k.lower() in keys and isinstance(v, str) and len(v) > 8:
                            return v
                        found = _deep_find(v, keys, depth + 1)
                        if found:
                            return found
                elif isinstance(obj, list):
                    for item in obj:
                        found = _deep_find(item, keys, depth + 1)
                        if found:
                            return found
                return None

            token_keys = {
                'authorization', 'authtoken', 'auth_token', 'apikey', 'api_key',
                'accesstoken', 'access_token', 'bearertoken', 'bearer_token',
                'token', 'secret', 'x-api-key', 'xapikey',
            }
            tok = _deep_find(nd, token_keys)
            if tok:
                log.debug("Auth token found in __NEXT_DATA__")
                return tok
        except Exception:
            pass

    # Pattern 2: inline JS variable assignments
    token_patterns = [
        r"""["\']?[Aa]uthorization["\']?\s*:\s*["\']Bearer\s+([A-Za-z0-9\-_.~+/=]{20,})["\']""",
        r"""["\']?[Aa]uthorization["\']?\s*:\s*["\']([A-Za-z0-9\-_.~+/=]{20,})["\']""",
        r"""["\']?(?:apiKey|api_key|x-api-key|xApiKey)["\']?\s*[=:]\s*["\']([A-Za-z0-9\-_.]{16,})["\']""",
        r"""["\']?(?:authToken|auth_token|accessToken|access_token)["\']?\s*[=:]\s*["\']([A-Za-z0-9\-_.~+/=]{20,})["\']""",
        r"""Bearer\s+([A-Za-z0-9\-_.~+/=]{20,})""",
    ]
    for pat in token_patterns:
        hits = re.findall(pat, html)
        if hits:
            log.debug("Auth token found via pattern: %s -> %s...", pat[:40], hits[0][:20])
            return hits[0]

    return None


def _ensure_auth_token(car_page_url: str) -> None:
    """
    Fetch the car page, extract the auth token, and add it to SESSION headers.
    Called once before the first variant API request.
    """
    global _AUTH_TOKEN
    if _AUTH_TOKEN:
        return

    log.debug("Fetching auth token from %s", car_page_url)
    try:
        resp = SESSION.get(car_page_url, timeout=20)
        if resp.status_code != 200:
            log.warning("Auth token page returned HTTP %s", resp.status_code)
            return

        tok = _extract_auth_token(resp.text)
        if tok:
            _AUTH_TOKEN = tok
            # Add to session so all subsequent requests carry it
            SESSION.headers.update({"Authorization": f"Bearer {tok}"})
            log.info("Auth token extracted and set (%d chars)", len(tok))
        else:
            log.warning(
                "Could not auto-extract auth token from page. "
                "Set CARANDBIKE_TOKEN env var manually: "
                "export CARANDBIKE_TOKEN=<token from DevTools Network tab>"
            )
            # Try env var fallback
            env_tok = os.environ.get("CARANDBIKE_TOKEN", "")
            if env_tok:
                _AUTH_TOKEN = env_tok
                SESSION.headers.update({"Authorization": f"Bearer {env_tok}"})
                log.info("Auth token loaded from CARANDBIKE_TOKEN env var")
    except Exception as exc:
        log.debug("Auth token fetch error: %s", exc)

USE_SELENIUM = False    # overridden by --selenium flag


def fetch(url: str, retries: int = 3, delay: float = 2.0) -> tuple[BeautifulSoup | None, str]:
    """GET a URL and return (BeautifulSoup, raw_html), or (None, '') on failure."""
    if USE_SELENIUM:
        return _selenium_fetch(url)

    for attempt in range(1, retries + 1):
        try:
            resp = SESSION.get(url, timeout=30)
            if resp.status_code == 200:
                return BeautifulSoup(resp.text, "lxml"), resp.text
            if resp.status_code == 403:
                log.warning("403 Forbidden – site may require Selenium. Try --selenium flag.")
            else:
                log.warning("HTTP %s  (%s)  attempt %d", resp.status_code, url, attempt)
        except requests.RequestException as exc:
            log.warning("Request error: %s  attempt %d", exc, attempt)
        if attempt < retries:
            time.sleep(delay * attempt + random.uniform(0, 1))
    return None, ""


# Extra headers used specifically for XHR/fetch API calls (merged on top of BASE_HEADERS)
_JSON_API_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
}

# Cache: tracks car pages whose cookies we've already loaded into SESSION
_COOKIE_WARMED: set[str] = set()


def _warm_session_cookies(page_url: str) -> None:
    """
    Visit the car page so the SESSION picks up any cookies the API needs.
    Only done once per unique page URL.
    """
    if page_url in _COOKIE_WARMED:
        return
    try:
        SESSION.get(page_url, timeout=20)
        _COOKIE_WARMED.add(page_url)
        log.debug("Warmed session cookies from %s", page_url)
    except Exception as exc:
        log.debug("Cookie warm failed for %s: %s", page_url, exc)


def fetch_json(url: str, retries: int = 3) -> dict | list | None:
    """GET a URL expecting JSON.
    
    Sends merged headers (BASE_HEADERS + JSON API overrides) so the API
    sees a proper browser XHR request, not a bare bot request.
    """
    # Build merged headers: start from session defaults, override for XHR
    merged = {**BASE_HEADERS, **_JSON_API_HEADERS}
    merged["Referer"] = "https://www.carandbike.com/"

    for attempt in range(1, retries + 1):
        try:
            resp = SESSION.get(url, headers=merged, timeout=30)
            if resp.status_code == 200:
                try:
                    return resp.json()
                except Exception:
                    # Some carandbike API responses have trailing "null" appended
                    # e.g. {"status":200,...}null  — strip and retry
                    text = resp.text.strip()
                    for suffix in ("null", "undefined", "false", "true"):
                        if text.endswith(suffix):
                            text = text[: -len(suffix)].rstrip()
                            break
                    try:
                        parsed = json.loads(text)
                        return parsed
                    except Exception:
                        log.debug("JSON decode error for %s: %s", url, resp.text[:200])
                        return None
            log.debug("fetch_json HTTP %s for %s", resp.status_code, url)
        except Exception as exc:
            log.debug("JSON fetch error attempt %d: %s", attempt, exc)
        time.sleep(2)
    return None


def parse_sitemap(sitemap_url: str) -> list:
    """
    Recursively parse a sitemap index or sitemap XML.
    Returns a list of car-model page URLs.
    """
    log.info("Parsing sitemap: %s", sitemap_url)
    soup, _ = fetch(sitemap_url)
    if soup is None:
        log.error("Could not fetch sitemap: %s", sitemap_url)
        return []

    # Sitemap index → child sitemaps
    child_sitemaps = soup.find_all("sitemap")
    if child_sitemaps:
        urls = []
        for sm in child_sitemaps:
            loc = sm.find("loc")
            if loc:
                child_urls = parse_sitemap(loc.text.strip())
                urls.extend(child_urls)
                time.sleep(0.5)
        return urls

    # Regular sitemap → <url><loc>…</loc></url>
    locs = [loc.text.strip() for loc in soup.find_all("loc")]

    # Keep only car model pages (e.g. /maruti-suzuki-cars/ertiga)
    # Exclude: /compare, /review, /news, /images, /videos, /price-in-*
    car_urls = [
        u for u in locs
        if re.search(r"/[a-z0-9-]+-cars/[a-z0-9-]+$", u)
        and not any(x in u for x in ["/compare", "/review", "/news", "/video", "/image"])
    ]
    log.info("Found %d car-model URLs in sitemap", len(car_urls))
    return car_urls


def extract_next_data(soup: BeautifulSoup) -> dict:
    tag = soup.find("script", id="__NEXT_DATA__")
    if tag and tag.string:
        try:
            return json.loads(tag.string)
        except json.JSONDecodeError:
            pass
    return {}


def safe_text(tag) -> str:
    return tag.get_text(strip=True) if tag else ""


def _brand_model_from_url(url: str) -> tuple[str, str]:
    parts = urlparse(url).path.strip("/").split("/")
    if len(parts) >= 2:
        brand = parts[0].replace("-cars", "").replace("-", " ").title()
        model = parts[1].replace("-", " ").title()
        return brand, model
    return "", ""


def _decode_next_image_url(src: str) -> str:
    if not src:
        return ""
    if "/_next/image?url=" in src:
        m = re.search(r"url=([^&]+)", src)
        if m:
            from urllib.parse import unquote
            return unquote(m.group(1))
    if src.startswith("//"):
        return "https:" + src
    if src.startswith("/"):
        return urljoin("https://www.carandbike.com", src)
    return src


def _clean_gallery_urls(urls: list, model_slug: str = "") -> list:
    cleaned = []
    seen = set()
    slug = model_slug.lower().replace(" ", "-")
    for raw in urls:
        url = _decode_next_image_url(str(raw))
        if not url or "/svg/" in url or url.endswith(".svg"):
            continue
        if not any(k in url for k in ("car-images", "/gallery/", "/orig/", "/colors/")):
            continue
        if slug and slug not in url.lower():
            continue
        if url not in seen:
            seen.add(url)
            cleaned.append(url)
    return cleaned


def _html_entity_price(text: str) -> str:
    return (
        text.replace("\\u0026#8377;", "₹")
        .replace("&#8377;", "₹")
        .replace("\\u0026", "&")
    )


def _parse_summary_spec(spec_str: str) -> dict:
    """Parse '1462 CC, Petrol, 19 KM/L, Manual' into structured fields."""
    out = {"engine_cc": "", "fuel_type": "", "mileage": "", "transmission": ""}
    if not spec_str:
        return out
    parts = [p.strip() for p in spec_str.split(",") if p.strip()]
    transmissions = {"manual", "automatic", "amt", "cvt", "dct", "imt", "tc"}
    fuels = {"petrol", "diesel", "cng", "electric", "hybrid", "petrol+cng", "petrol + cng"}
    for part in parts:
        low = part.lower()
        if re.search(r"\d+\s*cc\b", low, re.I):
            out["engine_cc"] = part
        elif re.search(r"km/l|km/full|km/charge|kmpl", low, re.I):
            out["mileage"] = part
        elif any(t in low for t in transmissions):
            out["transmission"] = part
        elif any(f in low for f in fuels) or "petrol" in low or "cng" in low or "electric" in low:
            out["fuel_type"] = part
    return out


def _extract_model_id(html: str) -> str:
    m = re.search(r"id_car_model=(\d+)", html) or re.search(
        r'get-variant-list\?model=(\d+)', html
    )
    return m.group(1) if m else ""


def _extract_embedded_variants(html: str) -> list[dict]:
    """Parse variant rows embedded in the page RSC payload."""
    variants = []
    seen_ids = set()
    variant_re = re.compile(
        r'\{\\"id_variant\\":(\d+),\\"name\\":\\"([^"]+)\\",\\"price\\":\\"([^"]*)\\",'\
        r'\\"price_value\\":(\d+),\\"spec\\":\\"([^"]*)\\",\\"url\\":\\"([^"]+)\\"'
    )

    for group in re.finditer(
        r'\\"key\\":\\"([^"]+)\\",\\"title\\":\\"([^"]+)\\",\\"data\\":\\[(.*?)\\]\\}',
        html,
        re.DOTALL,
    ):
        fuel_group = group.group(2)
        for vid, name, price_disp, price_val, spec, var_url in variant_re.findall(group.group(3)):
            if vid in seen_ids:
                continue
            seen_ids.add(vid)
            parsed = _parse_summary_spec(spec.replace("\\u0026", "&"))
            variants.append({
                "variant_id": vid,
                "variant_name": name,
                "fuel_group": fuel_group,
                "price_inr": int(price_val),
                "price_display": _html_entity_price(price_disp),
                "spec_summary": spec,
                "engine_cc": parsed["engine_cc"],
                "fuel_type": parsed["fuel_type"],
                "mileage": parsed["mileage"],
                "transmission": parsed["transmission"],
                "variant_url": urljoin("https://www.carandbike.com", var_url),
            })

    if variants:
        return variants

    for vid, name, price_disp, price_val, spec, var_url in variant_re.findall(html):
        if vid in seen_ids:
            continue
        seen_ids.add(vid)
        parsed = _parse_summary_spec(spec.replace("\\u0026", "&"))
        variants.append({
            "variant_id": vid,
            "variant_name": name,
            "fuel_group": "",
            "price_inr": int(price_val),
            "price_display": _html_entity_price(price_disp),
            "spec_summary": spec,
            "engine_cc": parsed["engine_cc"],
            "fuel_type": parsed["fuel_type"],
            "mileage": parsed["mileage"],
            "transmission": parsed["transmission"],
            "variant_url": urljoin("https://www.carandbike.com", var_url),
        })
    if variants:
        return variants

    for vid, name, price in re.findall(
        r'\\"id_variant\\":(\d+),\\"variant_name\\":\\"([^"]+)\\",\\"price\\":(\d+)',
        html,
    ):
        if vid in seen_ids:
            continue
        seen_ids.add(vid)
        variants.append({
            "variant_id": vid,
            "variant_name": name,
            "fuel_group": "",
            "price_inr": int(price),
            "price_display": f"₹ {int(price) / 100000:.2f} Lakh".rstrip("0").rstrip("."),
            "spec_summary": "",
            "engine_cc": "",
            "fuel_type": "",
            "mileage": "",
            "transmission": "",
            "variant_url": "",
        })
    return variants


def _extract_colours(html: str) -> list[dict]:
    colours = []
    seen = set()
    for cid, name, hexcode, image in re.findall(
        r'\\"id_model_color\\":(\d+),\\"color_name\\":\\"([^"]+)\\",\\"hexcode\\":\\"([^"]*)\\",'\
        r'(?:\\"secondary_code\\":[^,]+,)?\\"priority\\":\d+,\\"image_path\\":\\"([^"]+)\\"',
        html,
    ):
        if name in seen:
            continue
        seen.add(name)
        colours.append({
            "id": cid,
            "name": name,
            "hex": hexcode,
            "image_url": image,
        })
    return colours


def _extract_braced_data_block(html: str, key: str) -> str:
    """Extract inner content of \\\"spec\\\":{\\\"data\\\":{...}} using brace matching."""
    marker = f'\\\"{key}\\\":{{\\\"data\\\":{{' 
    idx = html.find(marker)
    if idx < 0:
        return ""
    start = idx + len(marker)
    depth = 1
    for pos in range(start, len(html)):
        ch = html[pos]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return html[start:pos]
    return ""


def _parse_spec_data_block(block: str, seen: set) -> list[dict]:
    """Parse nested category → label/value arrays from a spec/feature data block."""
    items = []
    block = block.replace("\\u0026", "&")
    skip = {"data", "section", "result", "cta", "spec", "feature", "specData"}

    for cat_m in re.finditer(r'\\"([^"]+)\\":\[\{', block):
        category = cat_m.group(1).strip()
        if category in skip or len(category) < 4:
            continue
        end = block.find('}],\\"', cat_m.end())
        section = block[cat_m.end(): end + 2 if end >= 0 else cat_m.end() + 8000]
        pairs = re.findall(
            r'(?:\[\{|\{)\{\\"label\\":\\"([^"]+)\\",\\"value\\":\\"([^"]*)\\"',
            section,
        )
        if not pairs:
            pairs = re.findall(
                r'\\"label\\":\\"([^"]+)\\",\\"value\\":\\"([^"]*)\\"',
                section,
            )
        for label, value in pairs:
            key = (category, label)
            if key in seen:
                continue
            seen.add(key)
            items.append({"category": category, "name": label, "value": value})
    return items


def _slugify(text: str) -> str:
    """Convert a brand/model name to the URL slug used by carandbike.com."""
    return text.lower().strip().replace(" ", "-")


def _manufacturer_slug_from_url(car_url: str) -> str:
    """
    Extract manufacturer slug from car_url.
    e.g. 'https://www.carandbike.com/maruti-suzuki-cars/e-vitara' -> 'maruti-suzuki'
    """
    path = urlparse(car_url).path.strip("/")
    brand_part = path.split("/")[0]
    return brand_part.replace("-cars", "")


def _model_slug_from_url(car_url: str) -> str:
    """
    Extract model slug from car_url.
    e.g. 'https://www.carandbike.com/maruti-suzuki-cars/e-vitara' -> 'e-vitara'
    """
    path = urlparse(car_url).path.strip("/")
    parts = path.split("/")
    return parts[1] if len(parts) >= 2 else ""


def _get_spec_action_hash(spec_page_html: str, spec_page_url: str) -> str:
    """
    Find the Next.js Server Action hash for the variant spec switcher.
    It's a 40-char hex string embedded in the compiled JS bundles.
    """
    # 1. Check the page HTML we already have
    if spec_page_html:
        candidates = re.findall(r'["\']([0-9a-f]{40})["\']', spec_page_html)
        if candidates:
            log.debug("Server action hash candidates from HTML: %s", candidates[:3])
            return candidates[0]

    # 2. Try fetching the page with extra stealth headers
    stealth_headers = {
        **BASE_HEADERS,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Sec-Ch-Ua": '"Chromium";v="148", "Google Chrome";v="148", "Not/A)Brand";v="99"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
        "Upgrade-Insecure-Requests": "1",
    }
    try:
        r = SESSION.get(spec_page_url, headers=stealth_headers, timeout=20)
        if r.status_code == 200:
            candidates = re.findall(r'["\']([0-9a-f]{40})["\']', r.text)
            if candidates:
                log.debug("Action hash from stealth GET: %s", candidates[0])
                return candidates[0]
            # Also search JS bundles linked from page
            js_urls = re.findall(r'"(/_next/static/chunks/[^"]+\.js)"', r.text)
            base = "https://www.carandbike.com"
            for js_path in js_urls[:10]:
                try:
                    jr = SESSION.get(base + js_path, timeout=15)
                    if jr.status_code == 200:
                        hits = re.findall(r'["\']([0-9a-f]{40})["\']', jr.text)
                        if hits:
                            log.debug("Action hash from JS chunk %s: %s", js_path, hits[0])
                            return hits[0]
                except Exception:
                    pass
    except Exception as exc:
        log.debug("Stealth GET failed: %s", exc)

    # 3. Known hardcoded hash (from network inspection, valid until next deploy)
    #    Update this if the site redeploys and the hash changes.
    known_hash = "4081247f0cf29f05ac14be929d77558e9989e58d19"
    log.info("Using hardcoded server action hash: %s", known_hash)
    return known_hash

def _parse_rsc_specs(rsc_text: str) -> tuple[list[dict], list[dict]]:
    specs, feats = [], []

    for line in rsc_text.splitlines():
        m = re.match(r'^\w+:(\{.*)', line)
        if not m:
            continue
        try:
            obj = json.loads(m.group(1))
        except Exception:
            continue

        if not isinstance(obj, dict):
            continue

        data_block = obj.get("data", {})
        if isinstance(data_block, dict):
            spec_block = data_block.get("spec",    {}).get("data", {})
            feat_block = data_block.get("feature", {}).get("data", {})

            if spec_block and isinstance(spec_block, dict):
                parsed = _parse_label_value_dict(spec_block)
                if parsed:
                    specs = parsed          # take the richest hit

            if feat_block and isinstance(feat_block, dict):
                parsed = _parse_label_value_dict(feat_block)
                if parsed:
                    feats = parsed          # take the richest hit

        # Also try legacy shapes on every line (don't break early)
        if not specs:
            s = _parse_variant_spec_api_response(obj, "specifications") or \
                _parse_variant_spec_api_response(obj, "specs")
            if s:
                specs = s
        if not feats:
            f = _parse_variant_spec_api_response(obj, "features")
            if f:
                feats = f

    return specs, feats


def _parse_label_value_dict(d: dict) -> list[dict]:
    """
    Parse {category: [{"label": ..., "value": ...}, ...]} into
    [{category, name, value}, ...].
    This is the exact shape used in carandbike's RSC spec/feature response.
    """
    items = []
    for category, entries in d.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name  = str(entry.get("label", entry.get("name",  "")))
            value = str(entry.get("value", ""))
            if name:
                items.append({"category": str(category), "name": name, "value": value})
    return items



# Cache: action hash per model URL (avoids re-fetching JS bundles each variant)
_ACTION_HASH_CACHE: dict[str, str] = {}


def _fetch_variant_specs_and_features(
    manufacturer_slug: str,
    model_slug: str,
    variant_name: str,
) -> tuple[list[dict], list[dict]]:
    """
    Fetch per-variant specs/features via the Next.js Server Action POST.
    """
    if not manufacturer_slug or not model_slug or not variant_name:
        return [], []

    spec_url = f"https://www.carandbike.com/{manufacturer_slug}-cars/{model_slug}/specifications"

    _warm_session_cookies(spec_url)

    action_hash = _ACTION_HASH_CACHE.get(spec_url, "")
    if not action_hash:
        _, page_html = fetch(spec_url)
        action_hash = _get_spec_action_hash(page_html or "", spec_url)
        _ACTION_HASH_CACHE[spec_url] = action_hash

    from urllib.parse import quote
    api_path = (
        f"/api/product/new-car/variant-get-spec-features"
        f"?model={model_slug}&manufacturer={manufacturer_slug}"
        f"&variant={quote(variant_name)}&version=4"
    )
    body = json.dumps([{"microservice": "API", "url": api_path}])

    from urllib.parse import quote as _q
    router_state = (
        "%5B%22%22%2C%7B%22children%22%3A%5B%22(global)%22%2C%7B%22children%22%3A"
        "%5B%22new%22%2C%7B%22children%22%3A%5B%5B%22vehicle%22%2C%22cars%22%2C%22d%22%2C"
        f"null%5D%2C%7B%22children%22%3A%5B%5B%22manufacturer%22%2C%22{_q(manufacturer_slug)}"
        f"%22%2C%22d%22%2Cnull%5D%2C%7B%22children%22%3A%5B%5B%22model%22%2C%22{_q(model_slug)}"
        "%22%2C%22d%22%2Cnull%5D%2C%7B%22children%22%3A%5B%5B%22child%22%2C%22specifications"
        "%22%2C%22d%22%2Cnull%5D%2C%7B%22children%22%3A%5B%22desktop%22%2C%7B%22children"
        "%22%3A%5B%22__PAGE__%22%2C%7B%7D%2Cnull%2Cnull%2C0%5D%7D%2Cnull%2Cnull%2C0%5D"
        "%7D%2Cnull%2Cnull%2C0%5D%7D%2Cnull%2Cnull%2C0%5D%7D%2Cnull%2Cnull%2C0%5D%7D"
        "%2Cnull%2Cnull%2C0%5D%7D%2Cnull%2Cnull%2C0%5D%7D%2Cnull%2Cnull%2C16%5D"
    )

    post_headers = {
        "accept": "text/x-component",
        "accept-language": "en-GB,en;q=0.9",
        "content-type": "text/plain;charset=UTF-8",
        "next-action": action_hash,
        "next-router-state-tree": router_state,
        "origin": "https://www.carandbike.com",
        "priority": "u=1, i",
        "referer": spec_url,
        "sec-ch-ua": '"Chromium";v="148", "Google Chrome";v="148", "Not/A)Brand";v="99"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "user-agent": BASE_HEADERS["User-Agent"],
    }

    log.debug("Server Action POST %s  variant=%r  hash=%s", spec_url, variant_name, action_hash)

    try:
        resp = SESSION.post(spec_url, data=body, headers=post_headers, timeout=30)
    except Exception as exc:
        log.debug("Server Action POST failed: %s", exc)
        return [], []

    log.debug("Server Action response: HTTP %s, %d chars", resp.status_code, len(resp.text))

    if resp.status_code not in (200, 303):
        log.debug("Server Action HTTP %s for variant %r", resp.status_code, variant_name)
        return [], []

    rsc_text = resp.text
    log.debug("RSC raw (first 600): %s", rsc_text[:600])

    specs, feats = _parse_rsc_specs(rsc_text)

    # Fallback: scan RSC lines for data.spec.data / data.feature.data
    if not specs and not feats:
        for line in rsc_text.splitlines():
            if '"spec"' not in line and '"feature"' not in line:
                continue
            m2 = re.match(r'^\d+:(\{.+)', line)
            if not m2:
                continue
            try:
                obj = json.loads(m2.group(1))
                data_block = obj.get("data", {})
                if isinstance(data_block, dict):
                    spec_block = data_block.get("spec",    {}).get("data", {})
                    feat_block = data_block.get("feature", {}).get("data", {})
                    if spec_block and not specs:
                        specs = _parse_label_value_dict(spec_block)
                    if feat_block and not feats:
                        feats = _parse_label_value_dict(feat_block)
                    if specs and feats:
                        break
            except Exception:
                pass

    log.info(
        "Server Action -> %s/%s %r: %d specs, %d features",
        manufacturer_slug, model_slug, variant_name, len(specs), len(feats),
    )
    return specs, feats



def _parse_variant_spec_api_response(data: dict | list, key: str) -> list[dict]:
    """
    Parse specs or features from the variant-get-spec-features API response.
    """
    if not data:
        return []

    # Unwrap {"data": {...}} or {"result": {...}} wrapper
    payload = data
    if isinstance(data, dict):
        for wrapper in ("data", "result", "response", "body"):
            if wrapper in data and isinstance(data[wrapper], (dict, list)):
                payload = data[wrapper]
                break

    # Extract the target section
    if isinstance(payload, dict):
        raw = payload.get(key)
        if raw is None and key == "specifications":
            raw = payload.get("specs")
    elif isinstance(payload, list):
        raw = payload if key in ("specifications", "specs") else None
    else:
        return []

    if not raw:
        return []

    items = []

    if isinstance(raw, list):
        for block in raw:
            if not isinstance(block, dict):
                continue

            if "name" in block and "value" in block and "data" not in block:
                category = str(block.get("category", ""))
                items.append({
                    "category": category,
                    "name":     str(block["name"]),
                    "value":    str(block["value"]),
                })
                continue

            category = str(block.get("category", block.get("title", block.get("group", ""))))
            sub = block.get("data") or block.get("items") or block.get("specs") or []

            if isinstance(sub, list):
                for entry in sub:
                    if not isinstance(entry, dict):
                        continue
                    name  = str(entry.get("name",  entry.get("label", entry.get("key",   ""))))
                    value = str(entry.get("value", entry.get("val",   entry.get("detail", ""))))
                    if name:
                        items.append({"category": category, "name": name, "value": value})

        return items

    if isinstance(raw, dict):
        for category, entries in raw.items():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                name  = str(entry.get("label", entry.get("name",  entry.get("key", ""))))
                value = str(entry.get("value", entry.get("val",   "")))
                if name:
                    items.append({"category": str(category), "name": name, "value": value})

    return items



def _extract_categorized_specs(html: str) -> list[dict]:
    """Extract {category, name, value} from specifications/features RSC payloads."""
    items = []
    seen = set()

    for key in ("spec", "feature"):
        block = _extract_braced_data_block(html, key)
        if block:
            items.extend(_parse_spec_data_block(block, seen))

    if items:
        return items

    html_norm = html.replace("\\u0026", "&")
    for label, value in re.findall(
        r'\\"label\\":\\"([^"]+)\\",\\"value\\":\\"([^"]*)\\"',
        html_norm,
    ):
        key = ("", label)
        if key in seen:
            continue
        seen.add(key)
        items.append({"category": "", "name": label, "value": value})
    return items


def _parse_product_schema(soup: BeautifulSoup) -> dict:
    out = {}
    tag = soup.find("script", id="product-schema-script")
    if not tag or not tag.string:
        return out
    try:
        data = json.loads(tag.string)
    except json.JSONDecodeError:
        return out

    graph = data.get("@graph", [data])
    if not isinstance(graph, list):
        graph = [graph]

    main = None
    for item in graph:
        if not isinstance(item, dict):
            continue
        types = item.get("@type", [])
        if isinstance(types, str):
            types = [types]
        if "Car" in types or (
            "Product" in types and item.get("offers") and item.get("manufacturer")
        ):
            main = item
            break

    if not main:
        return out

    offers = main.get("offers") or {}
    price_inr = ""
    if isinstance(offers, dict) and offers.get("price"):
        price_inr = str(offers["price"])

    rating_block = main.get("aggregateRating") or {}
    desc = main.get("description", "")
    if isinstance(desc, list):
        desc = desc[0] if desc else ""
    if desc:
        desc = html_lib.unescape(str(desc))
        desc = re.sub(r"\s+", " ", desc).strip()

    out = {
        "brand": (main.get("brand") or {}).get("name", "") if isinstance(main.get("brand"), dict) else main.get("brand", ""),
        "model": main.get("model", "") or main.get("name", ""),
        "body_type": main.get("bodyType", ""),
        "fuel_types": main.get("fuelType", ""),
        "transmission_types": main.get("vehicleTransmission", ""),
        "seating": str(main.get("vehicleSeatingCapacity", "") or ""),
        "price_min_inr": price_inr,
        "summary": desc,
        "rating": str(rating_block.get("ratingValue", "") or ""),
        "review_count": str(rating_block.get("reviewCount", "") or ""),
        "launch_date": main.get("vehicleModelDate", ""),
        "image": main.get("image", ""),
    }
    return out


def _rupees_to_lakh_str(rupees: int | str) -> str:
    try:
        val = int(rupees)
    except (TypeError, ValueError):
        return ""
    if val <= 0:
        return ""
    lakhs = val / 100_000
    return f"₹ {lakhs:.2f} Lakh".replace(".00", "")


# ---------------------------------------------------------------------------
# IMAGE DOWNLOAD + WEBP CONVERSION
# ---------------------------------------------------------------------------

def _download_and_convert_images(
    image_urls: list[str],
    save_dir: Path,
    model_slug: str,
    max_width: int = 1200,
    quality: int = 82,
) -> list[str]:
    """
    Download each image URL, convert to WebP (max_width × auto height),
    save to save_dir, and return a list of local file paths.

    Requirements: pip install Pillow requests
    Falls back gracefully if Pillow is not installed (returns original URLs).
    """
    if not image_urls:
        return []

    try:
        from PIL import Image
        import io
    except ImportError:
        log.warning(
            "Pillow not installed — skipping WebP conversion. "
            "Run: pip install Pillow"
        )
        return list(image_urls)

    save_dir.mkdir(parents=True, exist_ok=True)

    dl_session = requests.Session()
    dl_session.headers.update({
        "User-Agent": BASE_HEADERS["User-Agent"],
        "Referer": "https://www.carandbike.com/",
    })

    local_paths = []
    for idx, url in enumerate(image_urls, 1):
        # Derive a clean filename
        raw_name = urlparse(url).path.split("/")[-1].split("?")[0]
        stem = re.sub(r"\.(jpe?g|png|webp|gif|avif)$", "", raw_name, flags=re.I)
        stem = re.sub(r"[^a-zA-Z0-9_\-]", "_", stem)
        if not stem:
            stem = f"{model_slug}_{idx}"
        filename = f"{stem}.webp"
        out_path = save_dir / filename

        # Skip if already converted
        if out_path.exists():
            log.debug("  Image already exists: %s", out_path)
            local_paths.append(str(out_path))
            continue

        log.info("  [%d/%d] Downloading: %s", idx, len(image_urls), url[-70:])
        try:
            resp = dl_session.get(url, timeout=30, stream=True)
            if resp.status_code != 200:
                log.warning("    Download HTTP %s — skipping", resp.status_code)
                continue
            raw_bytes = resp.content
        except Exception as exc:
            log.warning("Download error: %s", exc)
            continue

        if REMOVE_LOGO:
            try:
                log.info("    Removing source logo/watermark…")
                raw_bytes = remove_logo_from_bytes(raw_bytes)
            except Exception as exc:
                log.warning("    Logo removal failed, keeping original image: %s", exc)
            finally:
                if idx < len(image_urls) and LOGO_REMOVAL_DELAY > 0:
                    log.info("    Waiting %.0fs before next Vertex AI call…", LOGO_REMOVAL_DELAY)
                    time.sleep(LOGO_REMOVAL_DELAY)

        try:
            img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")

            # Resize if wider than max_width (preserve aspect ratio)
            if img.width > max_width:
                new_height = int(img.height * max_width / img.width)
                img = img.resize((max_width, new_height), Image.LANCZOS)

            img.save(out_path, "WEBP", quality=quality, method=6)
            log.info(
                "Saved WebP: %s  (%dx%d, %.1fKB)",
                filename, img.width, img.height, out_path.stat().st_size / 1024,
            )
            local_paths.append(str(out_path))
        except Exception as exc:
            log.warning("    Image conversion error: %s", exc)
        time.sleep(0.1)  # be polite to CDN
    return local_paths

def parse_model_page(url: str, images_dir: Path | None = None) -> dict:
    """Scrape a single car model page and return a structured dict."""
    log.info("  Scraping: %s", url)
    soup, html = fetch(url)
    if soup is None:
        return {"url": url, "error": "fetch_failed", "variants": []}

    brand_slug, model_slug = _brand_model_from_url(url)
    result = {
        "url": url,
        "model_id": "",
        "brand": brand_slug,
        "model": model_slug,
        "tagline": "",
        "price_min_inr": "",
        "price_max_inr": "",
        "price_min_display": "",
        "price_max_display": "",
        "rating": "",
        "review_count": "",
        "body_type": "",
        "fuel_types": "",
        "transmission_types": "",
        "seating": "",
        "launch_date": "",
        "discontinued": False,
        "summary": "",
        "intro_para": "",
        "whats_new": {
            "introduction": [],
            "exterior":     [],
            "interior":     [],
            "powertrain":   [],
        },
        "featured_image_url": "",
        "pros": "",
        "cons": "",
        "expert_rating": "",
        "gallery_images": [],
        "gallery_local_paths": [],
        "colours": [],
        "specifications": [],
        "features": [],
        "variants": [],
    }

    result["model_id"] = _extract_model_id(html)

    schema = _parse_product_schema(soup)
    for key, val in schema.items():
        if val and not result.get(key.replace("_inr", "")):
            if key == "price_min_inr":
                result["price_min_inr"] = val
                result["price_min_display"] = _rupees_to_lakh_str(val)
            else:
                result[key] = val

    # Legacy __NEXT_DATA__ (older pages)
    next_data = extract_next_data(soup)
    if next_data:
        try:
            props = next_data.get("props", {}).get("pageProps", {})
            _parse_next_data_props(props, result, url)
        except Exception as exc:
            log.debug("__NEXT_DATA__ parse error: %s", exc)

    _enrich_from_html(soup, url, result, html)

    # Variants from embedded RSC data (primary source)
    embedded = _extract_embedded_variants(html)
    if embedded:
        result["variants"] = embedded
        prices = [v["price_inr"] for v in embedded if v.get("price_inr")]
        if prices:
            result["price_min_inr"] = str(min(prices))
            result["price_max_inr"] = str(max(prices))
            result["price_min_display"] = _rupees_to_lakh_str(min(prices))
            result["price_max_display"] = _rupees_to_lakh_str(max(prices))

    colour_url = url.rstrip("/") + "/colours"
    _, colour_html = fetch(colour_url)
    if colour_html:
        result["colours"] = _extract_colours(colour_html)

    mfr_slug   = _manufacturer_slug_from_url(url)
    model_slug_raw = _model_slug_from_url(url)

    variants_with_specs = []
    for v in result.get("variants", []):
        vname = str(v.get("variant_name", ""))
        v_specs, v_feats = _fetch_variant_specs_and_features(mfr_slug, model_slug_raw, vname)
        v["specifications"] = v_specs
        v["features"]       = v_feats
        variants_with_specs.append(v)
    result["variants"] = variants_with_specs

    # Model-level fallback specs (used only when variant API returns nothing)
    spec_url = url.rstrip("/") + "/specifications"
    feat_url = url.rstrip("/") + "/features"
    _, spec_html = fetch(spec_url)
    if spec_html:
        result["specifications"] = _extract_categorized_specs(spec_html)
    _, feat_html = fetch(feat_url)
    if feat_html:
        result["features"] = _extract_categorized_specs(feat_html)

    # ---- Download + convert gallery images to local WebP (capped to MAX_IMAGES) ----
    if result["gallery_images"] and images_dir is not None:
        if MAX_IMAGES and len(result["gallery_images"]) > MAX_IMAGES:
            log.info("  Capping gallery images: %d found → keeping first %d",
                     len(result["gallery_images"]), MAX_IMAGES)
            result["gallery_images"] = result["gallery_images"][:MAX_IMAGES]

        slug_for_dir = model_slug_raw or re.sub(r"[^a-z0-9-]", "-", result["model"].lower())
        model_img_dir = images_dir / slug_for_dir
        log.info("  Converting %d gallery images to WebP → %s",
                 len(result["gallery_images"]), model_img_dir)
        local_paths = _download_and_convert_images(
            result["gallery_images"],
            model_img_dir,
            slug_for_dir,
        )
        result["gallery_local_paths"] = local_paths
        log.info("  WebP images saved: %d", len(local_paths))

    return result

def _parse_next_data_props(props: dict, result: dict, url: str):
    """Extract model + variant data from __NEXT_DATA__ pageProps."""

    car_data = (
        props.get("carData")
        or props.get("modelData")
        or props.get("carDetails")
        or props.get("car")
        or {}
    )

    if not car_data:
        for key in list(props.keys()):
            val = props[key]
            if isinstance(val, dict) and ("model_name" in val or "make_name" in val or "variants" in val):
                car_data = val
                break

    result["brand"]        = (car_data.get("make_name") or car_data.get("brand_name")
                               or car_data.get("make") or "")
    result["model"]        = (car_data.get("model_name") or car_data.get("name")
                               or car_data.get("model") or "")
    result["tagline"]      = car_data.get("tagline", "") or car_data.get("tag_line", "")
    result["body_type"]    = car_data.get("body_type", "") or car_data.get("bodyType", "")
    result["launch_date"]  = (car_data.get("launch_date", "")
                               or car_data.get("launched_on", "")
                               or car_data.get("launchDate", ""))
    result["discontinued"] = bool(car_data.get("discontinued")
                                   or car_data.get("is_discontinued")
                                   or car_data.get("isDiscontinued"))
    result["summary"]      = car_data.get("description", "") or car_data.get("summary", "")

    price_info = car_data.get("price") or {}
    if isinstance(price_info, dict):
        result["price_min_inr"] = str(price_info.get("min", "") or "")
        result["price_max_inr"] = str(price_info.get("max", "") or "")
    else:
        result["price_min_inr"] = str(price_info or car_data.get("min_price", "") or "")
    if not result["price_min_inr"]:
        result["price_min_inr"] = str(car_data.get("min_price", "") or car_data.get("minPrice", "") or "")
        result["price_max_inr"] = str(car_data.get("max_price", "") or car_data.get("maxPrice", "") or "")

    rating_info = car_data.get("rating") or {}
    if isinstance(rating_info, dict):
        result["rating"]       = str(rating_info.get("value", "") or rating_info.get("rating", "") or "")
        result["review_count"] = str(rating_info.get("count", "") or rating_info.get("review_count", "") or "")
    else:
        result["rating"]       = str(rating_info or "")
        result["review_count"] = str(car_data.get("review_count", "") or "")

    ft = car_data.get("fuel_types") or car_data.get("fuelTypes") or []
    result["fuel_types"] = ", ".join(ft) if isinstance(ft, list) else str(ft)

    tt = car_data.get("transmission_types") or car_data.get("transmissionTypes") or []
    result["transmission_types"] = ", ".join(tt) if isinstance(tt, list) else str(tt)

    images = car_data.get("images") or car_data.get("gallery") or []
    if isinstance(images, list):
        result["gallery_images"] = [
            img.get("url") or img.get("src") or img.get("image_url") or img
            for img in images
            if isinstance(img, dict) and (img.get("url") or img.get("src") or img.get("image_url"))
        ] or [img for img in images if isinstance(img, str)]

    result["pros"]          = " | ".join(car_data.get("pros", []) or [])
    result["cons"]          = " | ".join(car_data.get("cons", []) or [])
    result["expert_rating"] = str(car_data.get("expert_rating", "") or car_data.get("expertRating", "") or "")

    variants_raw = (
        props.get("variantList")
        or props.get("variants")
        or car_data.get("variants")
        or car_data.get("variant_list")
        or []
    )
    if isinstance(variants_raw, list):
        for v in variants_raw:
            if isinstance(v, dict):
                result["variants"].append(_parse_variant(v))


def _parse_variant(v: dict) -> dict:
    """Normalise a single variant dict."""
    specs    = v.get("specifications") or v.get("specs") or {}
    features = v.get("features") or {}
    colours  = v.get("colors") or v.get("colours") or v.get("color_list") or []

    colour_info = []
    for c in colours if isinstance(colours, list) else []:
        imgs = c.get("images") or []
        if isinstance(imgs, list):
            img_urls = [
                i.get("url") or i.get("src") or i
                for i in imgs
                if isinstance(i, dict) and (i.get("url") or i.get("src"))
            ] or [i for i in imgs if isinstance(i, str)]
        else:
            img_urls = [imgs] if isinstance(imgs, str) else []

        if not img_urls and c.get("image_url"):
            img_urls = [c["image_url"]]
        if not img_urls and c.get("imageUrl"):
            img_urls = [c["imageUrl"]]

        colour_info.append({
            "name":   c.get("name", "") or c.get("color_name", ""),
            "hex":    c.get("hex_code", "") or c.get("hex", "") or c.get("color_code", ""),
            "images": img_urls,
        })

    return {
        "variant_id":     str(v.get("id", "") or v.get("variant_id", "") or ""),
        "variant_name":   v.get("name", "") or v.get("variant_name", "") or v.get("variantName", ""),
        "price_inr":      str(v.get("price", "") or v.get("ex_showroom_price", "") or v.get("exShowroomPrice", "") or ""),
        "fuel_type":      v.get("fuel_type", "") or v.get("fuelType", ""),
        "transmission":   v.get("transmission", "") or v.get("transmission_type", "") or v.get("transmissionType", ""),
        "seating":        str(v.get("seating_capacity", "") or v.get("seats", "") or v.get("seatingCapacity", "") or ""),
        "colours":        colour_info,
        "specifications": _flatten_dict(specs),
        "features":       _flatten_dict(features),
    }

WHATS_NEW_SECTIONS: dict[str, str] = {
    "introduction": "Introduction",
    "exterior":     "Exterior",
    "interior":     "Interior",
    "powertrain":   "Powertrain",
}

def _clean_html_text(raw: str) -> str:
    """Strip tags/entities and collapse whitespace from scraped HTML fragments."""
    text = html_lib.unescape(re.sub(r"<[^>]+>", " ", raw or ""))
    return re.sub(r"\s+", " ", text).strip()


def _extract_intro_para(html: str) -> str:
    """
    Extract the car intro paragraph shown at the top of the model page.
    Primary source: JSON-LD Product/Car description embedded in page HTML.
    """
    if not html:
        return ""

    m = re.search(r'"description"\s*:\s*\["([^"]{40,}?)"\s*,?\s*"@type"', html, re.S)
    if not m:
        m = re.search(r'"description"\s*:\s*\["([^"]{40,}?)"\]', html, re.S)
    if m:
        return _clean_html_text(m.group(1))

    m = re.search(
        r"is a \d+-seater[^<]{0,600}available at a starting price of Rs\.\s*[\d.]+\s*Lakh[^<]{0,400}",
        html,
        re.I,
    )
    if m:
        return _clean_html_text(m.group(0))

    return ""


def _whats_new_section_html(html: str) -> str:
    """Return the HTML chunk for the 'What's New on …' block, or full page as fallback."""
    if not html:
        return ""

    m = re.search(
        r"What['\u2019\u0027]s New on[^<]{0,120}</h2>(.*?)(?:</section>|"
        r"<h2[^>]*>\s*(?:Compare|Expert Review|User Review|FAQs|Pros and Cons|"
        r"Similar Cars|Latest Updates|Specifications|Features|Variants|Price List|"
        r"Mileage|Safety|Colours|Videos|Images)\b)",
        html,
        re.I | re.S,
    )
    if m:
        return m.group(1)

    m = re.search(r"What['\u2019\u0027]s New on[^<]{0,120}</h2>", html, re.I)
    if m:
        return html[m.end() : m.end() + 25000]

    return html


def _unescape_rsc_html(chunk: str) -> str:
    """
    Next.js RSC payloads embed HTML inside JS string literals, with tags
    escaped as \\u003c / \\u003e and entities as \\u0026.... Normalise all
    of that down to plain HTML so a single extraction path can handle both
    server-rendered HTML and RSC-embedded HTML.
    """
    if not chunk:
        return chunk
    out = chunk
    if "\\u003c" in out or "\\u0026" in out:
        out = (
            out.replace("\\u003c", "<")
               .replace("\\u003e", ">")
               .replace("\\u0026nbsp;", " ")
               .replace("\\u0026rsquo;", "'")
               .replace("\\u0026lsquo;", "'")
               .replace("\\u0026rdquo;", '"')
               .replace("\\u0026ldquo;", '"')
               .replace("\\u0026amp;", "&")
               .replace("\\u0026#39;", "'")
               .replace("\\r\\n", "\n")
               .replace("\\n", "\n")
        )
    return out


# Heading-like tags that can plausibly introduce a What's New sub-section.
_HEADING_TAG_RE = re.compile(r"<(h[2-5]|strong|b)\b[^>]*>(.*?)</\1>", re.I | re.S)


def _extract_whats_new_from_chunk(chunk: str) -> dict[str, list[str]]:
    """
    Parse known What's New sub-headings (Introduction / Exterior / Interior /
    Powertrain) from an HTML chunk.

    Strategy: locate every heading-like tag in the chunk, classify it
    against our four known section names, then take *everything between
    that heading and the next heading* (not just an immediately-adjacent
    <ul>) and pull every <li> out of that span. This is deliberately
    lenient about what sits between the heading and its bullet list (an
    intro <p>, a wrapper <div>, extra whitespace, etc.) — real CarAndBike
    markup is not always a clean heading->ul pair, which is why the old
    strict "</h3>\\s*<ul>" regex silently returned nothing for
    Exterior/Interior/Powertrain while Introduction (which usually IS
    heading->ul directly) kept working.
    """
    sections: dict[str, list[str]] = {key: [] for key in WHATS_NEW_SECTIONS}
    if not chunk:
        return sections

    html_chunk = _unescape_rsc_html(chunk)

    headings = []  # (end_of_heading_tag_pos, key)
    for m in _HEADING_TAG_RE.finditer(html_chunk):
        text = _clean_html_text(m.group(2)).lower()
        if not text:
            continue
        for key, label in WHATS_NEW_SECTIONS.items():
            if re.search(rf"\b{re.escape(label.lower())}\b", text):
                headings.append((m.start(), m.end(), key))
                break

    if not headings:
        return sections

    headings.sort(key=lambda h: h[0])

    for idx, (_, end_pos, key) in enumerate(headings):
        next_start = headings[idx + 1][0] if idx + 1 < len(headings) else len(html_chunk)
        span = html_chunk[end_pos:next_start]

        items = [
            _clean_html_text(li)
            for li in re.findall(r"<li[^>]*>(.*?)</li>", span, re.I | re.S)
        ]
        items = [t for t in items if t]

        if not items:
            # No <ul>/<li> markup for this section — fall back to splitting
            # the paragraph text into sentence-like chunks so we still
            # capture something instead of an empty section.
            text_only = _clean_html_text(span)
            if text_only:
                items = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text_only) if s.strip()]

        if items:
            sections[key].extend(items)

    return sections


def _extract_whats_new(html: str) -> dict[str, list[str]]:
    """
    Parse the 'What's New' section on the main model page.
    Only the four known sub-headings are collected; all others are ignored.
    """
    if not html:
        return {key: [] for key in WHATS_NEW_SECTIONS}

    block = _whats_new_section_html(html)
    sections = _extract_whats_new_from_chunk(block)

    if not any(sections.values()):
        sections = _extract_whats_new_from_chunk(html)

    return sections


def _flatten_dict(d, parent_key="", sep=" > "):
    """Recursively flatten nested dict to flat key→value."""
    items = {}
    if not isinstance(d, dict):
        return items
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key, sep))
        elif isinstance(v, list):
            items[new_key] = " | ".join(str(i) for i in v if i is not None)
        else:
            items[new_key] = str(v) if v is not None else ""
    return items


def _enrich_from_html(soup: BeautifulSoup, url: str, result: dict, html: str = ""):
    """Fill any missing fields from raw HTML / JSON-LD."""

    if not result["brand"] or not result["model"]:
        brand, model = _brand_model_from_url(url)
        result["brand"] = result["brand"] or brand
        result["model"] = result["model"] or model

    if not result["model"]:
        h1 = soup.find("h1")
        if h1:
            result["model"] = safe_text(h1)

    if not result["rating"]:
        for attr in ["ratingValue", "starRating"]:
            tag = soup.find(attrs={"itemprop": attr})
            if tag:
                result["rating"] = safe_text(tag)
                break

    imgs = list(result.get("gallery_images") or [])
    hero_image = ""
    for img in soup.find_all("img"):
        for attr in ("src", "data-src"):
            src = img.get(attr, "")
            if src:
                imgs.append(src)
    if html:
        for m in re.findall(r'\\"image_path\\":\\"(https://images\.carandbike\.com[^"]+)\\"', html):
            imgs.append(m)
        if schema_img := re.search(
            r'\\"model_image_path\\":\\"(https://images\.carandbike\.com[^"]+)\\"', html
        ):
            hero_image = schema_img.group(1)
            imgs.insert(0, hero_image)
    parts = urlparse(url).path.strip("/").split("/")
    model_slug = parts[1].lower() if len(parts) >= 2 else ""
    result["gallery_images"] = _clean_gallery_urls(imgs, model_slug=model_slug)

    if hero_image:
        hero_clean = _decode_next_image_url(hero_image)
        if hero_clean in result["gallery_images"]:
            result["gallery_images"].remove(hero_clean)
            result["gallery_images"].insert(0, hero_clean)
        result["featured_image_url"] = hero_clean
    elif result["gallery_images"]:
        result["featured_image_url"] = result["gallery_images"][0]

    if not result.get("summary"):
        result["summary"] = _extract_intro_para(html)
    result["intro_para"] = result.get("summary") or _extract_intro_para(html)

    whats_new = _extract_whats_new(html)
    if any(whats_new.values()):
        result["whats_new"] = whats_new


def _fetch_variants_api(brand_slug: str, model_slug: str) -> list:
    """
    Try CarAndBike's internal API endpoints that power their frontend.
    """
    candidates = [
        f"https://www.carandbike.com/api/cars/variants?brand={brand_slug}&model={model_slug}",
        f"https://www.carandbike.com/api/v1/cars/{brand_slug}/{model_slug}/variants",
        f"https://www.carandbike.com/_next/data/variants/{brand_slug}-cars/{model_slug}.json",
    ]
    for url in candidates:
        data = fetch_json(url)
        if not data:
            continue
        if isinstance(data, list) and data:
            log.debug("API hit: %s", url)
            return [_parse_variant(v) for v in data if isinstance(v, dict)]
        if isinstance(data, dict):
            for key in ["variants", "data", "result"]:
                raw = data.get(key)
                if isinstance(raw, list) and raw:
                    return [_parse_variant(v) for v in raw if isinstance(v, dict)]
    return []


# One CSV row = one VARIANT of one MODEL.
VARIANT_CSV_FIELDS = [
    "brand",
    "model",
    "model_id",
    "car_url",
    "body_type",
    "fuel_types",
    "transmission_types",
    "seating",
    "price_min_inr",
    "price_max_inr",
    "price_min_display",
    "price_max_display",
    "rating",
    "review_count",
    "launch_date",
    "discontinued",
    "summary",
    "variant_id",
    "variant_name",
    "variant_url",
    "fuel_group",
    "variant_price_inr",
    "variant_price_display",
    "variant_fuel_type",
    "variant_transmission",
    "variant_mileage",
    "variant_engine_cc",
    "variant_spec_summary",
    "colours_json",
    "gallery_images_json",
    "gallery_local_paths_json",
    "specifications_json",
    "features_json",
]

SPEC_CSV_FIELDS = [
    "brand",
    "model",
    "model_id",
    "variant_id",
    "variant_name",
    "record_type",
    "category",
    "name",
    "value",
]


def models_to_variant_rows(all_models: list) -> tuple[list, list]:
    """One CSV row per variant with JSON columns for nested data."""
    rows = []
    for model in all_models:
        colours_json = json.dumps(model.get("colours", []), ensure_ascii=False)
        gallery_json = json.dumps(model.get("gallery_images", []), ensure_ascii=False)
        gallery_local_json = json.dumps(model.get("gallery_local_paths", []), ensure_ascii=False)

        model_specs_json = json.dumps(model.get("specifications", []), ensure_ascii=False)
        model_features_json = json.dumps(model.get("features", []), ensure_ascii=False)

        base = {
            "brand": model.get("brand", ""),
            "model": model.get("model", ""),
            "model_id": model.get("model_id", ""),
            "car_url": model.get("url", ""),
            "body_type": model.get("body_type", ""),
            "fuel_types": model.get("fuel_types", ""),
            "transmission_types": model.get("transmission_types", ""),
            "seating": model.get("seating", ""),
            "price_min_inr": model.get("price_min_inr", ""),
            "price_max_inr": model.get("price_max_inr", ""),
            "price_min_display": model.get("price_min_display", ""),
            "price_max_display": model.get("price_max_display", ""),
            "rating": model.get("rating", ""),
            "review_count": model.get("review_count", ""),
            "launch_date": model.get("launch_date", ""),
            "discontinued": model.get("discontinued", False),
            "summary": model.get("summary", ""),
            "colours_json": colours_json,
            "gallery_images_json": gallery_json,
            "gallery_local_paths_json": gallery_local_json,
        }

        variants = model.get("variants", [])
        if not variants:
            row = dict(base)
            for k in (
                "variant_id", "variant_name", "variant_url", "fuel_group",
                "variant_price_inr", "variant_price_display", "variant_fuel_type",
                "variant_transmission", "variant_mileage", "variant_engine_cc",
                "variant_spec_summary",
            ):
                row[k] = ""
            row["specifications_json"] = model_specs_json
            row["features_json"] = model_features_json
            rows.append(row)
            continue

        for v in variants:
            row = dict(base)

            v_specs = v.get("specifications") or []
            v_feats = v.get("features") or []
            row["specifications_json"] = json.dumps(v_specs, ensure_ascii=False) if v_specs else model_specs_json
            row["features_json"] = json.dumps(v_feats, ensure_ascii=False) if v_feats else model_features_json

            row.update({
                "variant_id": v.get("variant_id", ""),
                "variant_name": v.get("variant_name", ""),
                "variant_url": v.get("variant_url", ""),
                "fuel_group": v.get("fuel_group", ""),
                "variant_price_inr": v.get("price_inr", v.get("price", "")),
                "variant_price_display": v.get("price_display", ""),
                "variant_fuel_type": v.get("fuel_type", ""),
                "variant_transmission": v.get("transmission", ""),
                "variant_mileage": v.get("mileage", ""),
                "variant_engine_cc": v.get("engine_cc", ""),
                "variant_spec_summary": v.get("spec_summary", ""),
            })
            rows.append(row)

    return rows, VARIANT_CSV_FIELDS


def models_to_spec_rows(all_models: list) -> tuple[list, list]:
    """Long-format specifications + features (easy relational DB import)."""
    rows = []
    for model in all_models:
        brand = model.get("brand", "")
        model_name = model.get("model", "")
        model_id = model.get("model_id", "")

        variants = model.get("variants", [])
        if variants:
            for v in variants:
                vid = v.get("variant_id", "")
                vname = v.get("variant_name", "")
                base = {"brand": brand, "model": model_name, "model_id": model_id,
                        "variant_id": vid, "variant_name": vname}
                for record_type, items in (
                    ("specification", v.get("specifications") or model.get("specifications", [])),
                    ("feature",       v.get("features")       or model.get("features", [])),
                ):
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        rows.append({
                            **base,
                            "record_type": record_type,
                            "category": item.get("category", ""),
                            "name": item.get("name", ""),
                            "value": item.get("value", ""),
                        })
        else:
            base = {"brand": brand, "model": model_name, "model_id": model_id,
                    "variant_id": "", "variant_name": ""}
            for record_type, items in (
                ("specification", model.get("specifications", [])),
                ("feature",       model.get("features", [])),
            ):
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    rows.append({
                        **base,
                        "record_type": record_type,
                        "category": item.get("category", ""),
                        "name": item.get("name", ""),
                        "value": item.get("value", ""),
                    })
    return rows, SPEC_CSV_FIELDS


def save_csv(rows: list, fieldnames: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    log.info("CSV saved  → %s  (%d rows, %d columns)", path, len(rows), len(fieldnames))


def save_json(data: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    log.info("JSON saved → %s  (%d models)", path, len(data))


def main():
    parser = argparse.ArgumentParser(
        description="CarAndBike.com scraper — sitemap or single URL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sitemap", help="Sitemap XML index URL")
    group.add_argument("--url",     nargs="+", help="One or more car model URLs (space-separated)")
    group.add_argument("--urls-file", help="Path to a text file with one car model URL per line")
    group.add_argument("--test-api", metavar="CAR_URL",
                       help="Test variant spec API for a car URL and print raw response.")

    parser.add_argument("--max",       type=int,   default=0,    help="Max models to scrape (0=all)")
    parser.add_argument("--output",   default="",               help="Output filename stem (no extension)")
    parser.add_argument("--delay",    type=float, default=2.0,  help="Seconds between page requests (default 2)")
    parser.add_argument("--selenium", action="store_true",      help="Use Selenium Chrome (needed if requests get 403)")
    parser.add_argument("--outdir",   default=".",              help="Output directory (default: current dir)")
    parser.add_argument("--images-dir", default="",            help="Directory to save downloaded WebP images (default: <outdir>/images)")
    parser.add_argument("--skip-images", action="store_true",  help="Skip image download/conversion")
    parser.add_argument("--max-images", type=int, default=5,
                        help="Max gallery images to scrape/download per model (default: 5, 0=unlimited)")
    parser.add_argument("--remove-logo", action="store_true",
                        help="Remove source-site logo/watermark from gallery images via "
                             "Vertex AI before saving (needs token.json/client_secret.json/mask.png)")
    parser.add_argument("--logo-delay", type=float, default=15.0,
                        help="Seconds to wait between Vertex AI logo-removal calls (default 15.0)")
    parser.add_argument("--rephrase-content", action="store_true",
                        help="Rephrase post_content + whats-new lists in the ACF JSON "
                             "via Gemini before saving (needs GEMINI_API_KEY in .env)")
    parser.add_argument("--rephrase-delay", type=float, default=1.0,
                        help="Seconds to sleep between rephrase API calls (default 1.0)")
    args = parser.parse_args()

    if args.test_api:
        car_url  = args.test_api
        mfr_slug = _manufacturer_slug_from_url(car_url)
        mod_slug = _model_slug_from_url(car_url)
        print(f"Testing variant spec API for: {car_url}")
        print(f"  manufacturer_slug = {mfr_slug!r}")
        print(f"  model_slug        = {mod_slug!r}")
        print()

        _ensure_auth_token(car_url + "/specifications")
        _warm_session_cookies(car_url)

        _, html = fetch(car_url)
        variants = _extract_embedded_variants(html) if html else []
        if not variants:
            print("No variants found on page. Try --selenium flag.")
            sys.exit(1)

        print(f"Found {len(variants)} variants. Testing all of them...")
        from urllib.parse import quote

        for v in variants:
            vname = v.get("variant_name", "")
            specs, feats = _fetch_variant_specs_and_features(mfr_slug, mod_slug, vname)
            print(f"\n{'─'*60}")
            print(f"Variant : {vname!r}")
            print(f"PARSED  : {len(specs)} specs, {len(feats)} features")
            if specs:
                print("  Sample specs:")
                for s in specs[:4]:
                    print(f"    {s['category']} | {s['name']}: {s['value']}")
            if feats:
                print("  Sample features:")
                for f in feats[:4]:
                    print(f"    {f['category']} | {f['name']}: {f['value']}")

        sys.exit(0)

    global USE_SELENIUM, REMOVE_LOGO, MAX_IMAGES, LOGO_REMOVAL_DELAY
    USE_SELENIUM = args.selenium
    REMOVE_LOGO = args.remove_logo
    MAX_IMAGES = args.max_images
    LOGO_REMOVAL_DELAY = args.logo_delay

    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = args.output or f"carandbike_{ts}"
    out_dir = Path(args.outdir)
    out_csv       = out_dir / f"{stem}.csv"
    out_specs_csv = out_dir / f"{stem}_specifications.csv"
    out_json      = out_dir / f"{stem}.json"
    out_acf_json  = out_dir / f"{stem}_acf.json"

    # Images directory
    images_dir: Path | None = None
    if not args.skip_images:
        images_dir = Path(args.images_dir) if args.images_dir else out_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        log.info("Images dir       : %s", images_dir)

    # Import ACF mapper
    try:
        import importlib, sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent))
        _acf_mapper = importlib.import_module("acf_mapper")
        _map_all = _acf_mapper.map_all_models
        log.info("ACF mapper     : loaded (will generate _acf.json)")
    except ImportError:
        _map_all = None
        log.warning("acf_mapper.py not found alongside scrapper.py — skipping _acf.json output")

    _rephrase_fn = None
    if args.rephrase_content:
        try:
            _rephrase_fn = lambda models: rephrase_models(models, delay_between=args.rephrase_delay)
            log.info("Content rephraser : ON (Gemini, GEMINI_API_KEY from .env)")
        except Exception as exc:
            log.warning("Content rephraser failed to initialise: %s", exc)

    if args.sitemap:
        urls = parse_sitemap(args.sitemap)
    elif args.urls_file:
        with open(args.urls_file, "r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    else:
        urls = args.url

    if not urls:
        log.error("No URLs to scrape. Exiting.")
        sys.exit(1)

    if args.max and args.max > 0:
        log.info("Limiting to first %d models (of %d found)", args.max, len(urls))
        urls = urls[:args.max]

    log.info("=" * 60)
    log.info("Models to scrape : %d", len(urls))
    log.info("Output CSV       : %s", out_csv)
    log.info("Output specs CSV : %s", out_specs_csv)
    log.info("Output JSON      : %s", out_json)
    log.info("Selenium mode    : %s", USE_SELENIUM)
    log.info("Delay (seconds)  : %.1f", args.delay)
    log.info("Images           : %s", "SKIP" if args.skip_images else str(images_dir))
    log.info("Max images/model : %s", "unlimited" if MAX_IMAGES == 0 else MAX_IMAGES)
    log.info("Logo removal     : %s", f"ON (Vertex AI, {LOGO_REMOVAL_DELAY:.0f}s between calls)" if REMOVE_LOGO else "OFF")
    log.info("Content rephrase : %s", "ON (Gemini)" if _rephrase_fn else "OFF")
    log.info("=" * 60)

    all_models = []
    for i, url in enumerate(urls, 1):
        log.info("[%d / %d]", i, len(urls))
        model_data = parse_model_page(url, images_dir=images_dir)
        all_models.append(model_data)

        # Incremental save every 10 models
        if i % 10 == 0:
            rows, fields = models_to_variant_rows(all_models)
            save_csv(rows, fields, out_csv)
            spec_rows, spec_fields = models_to_spec_rows(all_models)
            save_csv(spec_rows, spec_fields, out_specs_csv)
            save_json(all_models, out_json)
            if _map_all:
                acf_data = _map_all(all_models)
                if _rephrase_fn:
                    acf_data = _rephrase_fn(acf_data)
                save_json(acf_data, out_acf_json)

        if i < len(urls):
            sleep_t = args.delay + random.uniform(0.5, 1.5)
            time.sleep(sleep_t)

    rows, fieldnames = models_to_variant_rows(all_models)
    save_csv(rows, fieldnames, out_csv)
    spec_rows, spec_fields = models_to_spec_rows(all_models)
    save_csv(spec_rows, spec_fields, out_specs_csv)
    save_json(all_models, out_json)
    if _map_all:
        acf_data = _map_all(all_models)
        if _rephrase_fn:
            log.info("Rephrasing content for %d models via Gemini…", len(acf_data))
            acf_data = _rephrase_fn(acf_data)
        save_json(acf_data, out_acf_json)
        log.info("ACF JSON saved → %s", out_acf_json)

    errors = [m for m in all_models if "error" in m]
    log.info("=" * 60)
    log.info("Anshul " * 10 )
    log.info("Done!  Scraped: %d  |  Rows in CSV: %d  |  Errors: %d",
             len(all_models), len(rows), len(errors))
    if errors:
        log.warning("Failed URLs:")
        for m in errors:
            log.warning("  %s", m["url"])

    print(f"\nDone. Variants CSV       -> {out_csv.resolve()}")
    print(f"Done. Specifications CSV -> {out_specs_csv.resolve()}")
    print(f"Done. JSON               -> {out_json.resolve()}")
    if _map_all:
        print(f"Done. ACF JSON (WordPress) -> {out_acf_json.resolve()}")
    if images_dir:
        print(f"Done. WebP Images        -> {images_dir.resolve()}")


if __name__ == "__main__":
    main()