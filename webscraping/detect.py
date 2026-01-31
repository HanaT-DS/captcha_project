"""
detect.py – Detection automatique de CAPTCHAs sur une page web

Analyse le DOM (iframes, selecteurs, mots-cles, URLs) pour identifier
la presence et le type de CAPTCHA : reCAPTCHA, hCaptcha, Turnstile,
CAPTCHA texte (image + input), ou page de challenge.
Retourne un CaptchaDetectionResult avec le score, le provider,
les localisations (widget, iframe, image, input) et les preuves.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

from playwright.sync_api import Page, Locator


# =====================================================================
# 1) Structures de données
# =====================================================================

@dataclass
class Evidence:
    """
    Preuve concrète indiquant la présence d’un CAPTCHA ou d’un challenge.
    """
    kind: str                  # "iframe", "dom", "keyword", "url"
    value: str                 # src iframe, selector, mot-clé, fragment URL
    provider_guess: Optional[str] = None


@dataclass
class ElementLocation:
    """
    Localisation d'un élément intéressant (widget, iframe, image captcha, input captcha).
    On stocke :
    - un sélecteur CSS (celui qu'on a utilisé pour le trouver)
    - un XPath (pour debug/rapport)
    - une bounding box (x, y, width, height) pour extraire / screenshot ciblé
    """
    role: str                  # "widget" | "iframe" | "captcha_image" | "captcha_input"
    css_selector: str
    xpath: Optional[str]
    bbox: Optional[Dict[str, float]]   # {x, y, width, height}


@dataclass
class ChallengeSignals:
    """
    Informations utiles quand on tombe sur une page de challenge / blocage.
    """
    final_url: str
    title: str
    matched_keywords: List[str]
    matched_url_hints: List[str]


@dataclass
class CaptchaDetectionResult:
    """
    Résultat global de la détection.

    - detected: True si on pense qu'il y a un captcha/challenge
    - category:
        * "none" : rien détecté
        * "captcha_widget" : présence d'un widget (provider ou captcha maison)
        * "challenge_page" : page de blocage / vérification
    - provider: "recaptcha" | "hcaptcha" | "turnstile" | "friendlycaptcha" | "unknown" | None
    - score: nb de familles de signaux (iframe/dom/keyword/url) présentes
    - evidences: liste de preuves (auditables)
    - widget_location: conteneur widget (souvent pour provider)
    - iframe_location: iframe principale du captcha (souvent provider)
    - captcha_image_location: image du captcha (surtout captcha textuel maison)
    - captcha_input_location: input où saisir le texte (captcha textuel maison)
    - challenge: infos si challenge_page
    """
    detected: bool
    category: str
    provider: Optional[str]
    score: int
    evidences: List[Evidence]
    visibility: Optional[str] = None  # visible | invisible | None

    widget_location: Optional[ElementLocation] = None
    iframe_location: Optional[ElementLocation] = None
    captcha_image_location: Optional[ElementLocation] = None
    captcha_input_location: Optional[ElementLocation] = None

    challenge: Optional[ChallengeSignals] = None

    def to_dict(self) -> Dict:
        return {
            "detected": self.detected,
            "category": self.category,
            "provider": self.provider,
            "score": self.score,
            "widget_location": asdict(self.widget_location) if self.widget_location else None,
            "iframe_location": asdict(self.iframe_location) if self.iframe_location else None,
            "captcha_image_location": asdict(self.captcha_image_location) if self.captcha_image_location else None,
            "captcha_input_location": asdict(self.captcha_input_location) if self.captcha_input_location else None,
            "challenge": asdict(self.challenge) if self.challenge else None,
            "evidences": [asdict(e) for e in self.evidences],
        }


# =====================================================================
# 2) Constantes (providers + selectors)
# =====================================================================

# A) Signatures provider dans les URLs d'iframe/frame
IFRAME_PROVIDER_HINTS: List[Tuple[str, List[str]]] = [
    ("recaptcha", ["recaptcha", "google.com/recaptcha", "gstatic.com/recaptcha"]),
    ("hcaptcha", ["hcaptcha.com"]),
    ("turnstile", ["challenges.cloudflare.com", "turnstile"]),
    ("friendlycaptcha", ["friendlycaptcha", "frcapi"]),
]

# B) Sélecteurs "widget provider" (conteneurs)
PROVIDER_WIDGET_SELECTORS: List[Tuple[str, str]] = [
    ("recaptcha", "div.g-recaptcha, iframe[src*='recaptcha'], iframe[src*='google.com/recaptcha'], iframe[src*='recaptcha.net']"),
    ("hcaptcha", "div.h-captcha, [data-hcaptcha-sitekey], iframe[src*='hcaptcha']"),
    ("turnstile", "div.cf-turnstile, input[name='cf-turnstile-response'], iframe[src*='challenges.cloudflare.com']"),
    ("friendlycaptcha", ".frc-captcha, [data-frc-captcha], iframe[src*='friendlycaptcha']"),
]

# C) Sélecteurs iframe provider (si on veut localiser une iframe précise via DOM)
PROVIDER_IFRAME_SELECTORS: List[Tuple[str, str]] = [
    ("recaptcha", "iframe[src*='recaptcha'], iframe[src*='google.com/recaptcha']"),
    ("hcaptcha", "iframe[src*='hcaptcha']"),
    ("turnstile", "iframe[src*='challenges.cloudflare.com']"),
    ("friendlycaptcha", "iframe[src*='friendlycaptcha'], iframe[src*='frcapi']"),
]

# D) CAPTCHA "maison" / textuel — image + input (générique)
# IMPORTANT : ordre = priorité décroissante (les plus spécifiques en premier)
GENERIC_CAPTCHA_IMAGE_SELECTORS: List[str] = [
    # PRIORITÉ 1 : Images avec ID ou CLASS contenant "captcha" (le plus spécifique)
    "img[id*='captcha' i]",
    "img[class*='captcha' i]",
    "img#captchaimg, img#captchaImage, img#captcha_img",
    # PRIORITÉ 2 : Images base64 dans des containers captcha
    "[id*='captcha' i] img[src^='data:image']",
    "[class*='captcha' i] img[src^='data:image']",
    ".captcha img[src^='data:image'], img[src^='data:image']",
    # PRIORITÉ 3 : Images avec src contenant "captcha"
    "img[src*='captcha' i]",
    # PRIORITÉ 4 : Images avec alt captcha (très spécifique)
    "img[alt*='captcha' i]",
    # PRIORITÉ 5 : Images avec alt sécurité (moins spécifique, peut matcher des icônes)
    "img[alt*='sécurité' i], img[alt*='security' i]",
    "img[alt*='verification' i], img[alt*='vérification' i]",
    "img[alt*='code' i][alt*='sécurité' i]",
]

GENERIC_CAPTCHA_INPUT_SELECTORS: List[str] = [
    # Inputs dans des containers captcha
    "[id*='captcha' i] input[type='text']",
    "[id*='captcha' i] input:not([type='hidden']):not([type='submit']):not([type='button'])",
    "[class*='captcha' i] input[type='text']",
    "[class*='captcha' i] input:not([type='hidden']):not([type='submit']):not([type='button'])",
    # Inputs avec attributs contenant "captcha" (exclure hidden)
    "input[name*='captcha' i]:not([type='hidden'])",
    "input[id*='captcha' i]:not([type='hidden'])",
    "input[class*='captcha' i]:not([type='hidden'])",
    "input[placeholder*='captcha' i]",
    # Inputs avec placeholder contenant des mots-clés
    "input[placeholder*='code' i][placeholder*='sécurité' i]",
    "input[placeholder*='security' i][placeholder*='code' i]",
    "input[placeholder*='verification' i]",
    # Fallback: anciennes règles
    "input#captcha, #captcha",
    ".captcha input[type='text'], .captcha input:not([type='hidden'])",
]

# E) Mots clés de challenge / blocage
KEYWORDS_CHALLENGE: List[str] = [
    # Anglais - blocages/challenges
    "verify you are human",
    "are you a robot",
    "unusual traffic",
    "suspicious activity",
    "security check",
    "access denied",
    "just a moment",
    "checking your browser",
    # Français - CAPTCHAs textuels
    "recopiez le code",
    "recopier le code",
    "code de sécurité",
    "code de vérification",
    "saisissez le code",
    "entrez le code",
    "tapez le code",
    "code anti-spam",
    "code anti spam",
    # Générique (toutes langues)
    "captcha",
]

# F) Fragments d'URL de challenge
URL_HINTS: List[str] = [
    "/challenge",
    "/captcha",
    "/verify",
    "/security-check",
    "checking",
]


# =====================================================================
# 3) Utilitaires
# =====================================================================

def _guess_provider_from_url(url: str) -> Optional[str]:
    """Devine le provider à partir d’une URL (souvent iframe)."""
    u = (url or "").lower()
    for provider, hints in IFRAME_PROVIDER_HINTS:
        if any(h in u for h in hints):
            return provider
    return None


def _get_xpath(locator: Locator) -> Optional[str]:
    """
    Calcule un XPath à partir d’un élément DOM via JavaScript.
    (utile pour debug/rapport ; Playwright préfère CSS.)
    """
    try:
        return locator.evaluate(
            """el => {
                if (el.id) return '//*[@id="' + el.id + '"]';
                const parts = [];
                while (el && el.nodeType === Node.ELEMENT_NODE) {
                    let index = 1;
                    let sibling = el.previousSibling;
                    while (sibling) {
                        if (sibling.nodeType === Node.ELEMENT_NODE &&
                            sibling.nodeName === el.nodeName) {
                            index++;
                        }
                        sibling = sibling.previousSibling;
                    }
                    parts.unshift(el.nodeName.toLowerCase() + '[' + index + ']');
                    el = el.parentNode;
                }
                return '/' + parts.join('/');
            }"""
        )
    except Exception:
        return None


def _make_location(role: str, css_selector: str, locator: Locator) -> ElementLocation:
    """Construit un ElementLocation complet (css + xpath + bbox)."""
    bbox = None
    try:
        bbox = locator.bounding_box()
    except Exception:
        bbox = None

    return ElementLocation(
        role=role,
        css_selector=css_selector,
        xpath=_get_xpath(locator),
        bbox=bbox,
    )


# =====================================================================
# 4) Détecteurs de preuves (pour score + explication)
# =====================================================================

def detect_iframe_providers(page: Page) -> List[Evidence]:
    """
    Détection via les iframes externes 
    utile pour reCAPTCHA/hCaptcha/Turnstile/FriendlyCaptcha
    """
    evidences: List[Evidence] = []

    for fr in page.frames:
        # ignore main frame
        if fr == page.main_frame:
            continue
        frame_url = (fr.url or "").lower()
        provider = _guess_provider_from_url(frame_url)
        if provider:
            evidences.append(Evidence(kind="iframe", value=frame_url, provider_guess=provider))

    return evidences


def detect_dom_markers(page: Page) -> List[Evidence]:
    """
    Détection via le DOM :
    - widgets provider (div.g-recaptcha, div.h-captcha, etc.)
    - captcha "maison" (img/src base64, img/src*captcha, input name*captcha, .captcha...)
    """
    evidences: List[Evidence] = []

    # Widgets provider
    for provider, selector in PROVIDER_WIDGET_SELECTORS:
        try:
            count = page.locator(selector).count()
            if count > 0:
                evidences.append(Evidence(kind="dom", value=f"{selector} (count={count})", provider_guess=provider))
        except Exception:
            continue

    # Captcha maison : image
    for selector in GENERIC_CAPTCHA_IMAGE_SELECTORS:
        try:
            count = page.locator(selector).count()
            if count > 0:
                evidences.append(Evidence(kind="dom", value=f"{selector} (count={count})", provider_guess=None))
        except Exception:
            continue

    # Captcha maison : input
    for selector in GENERIC_CAPTCHA_INPUT_SELECTORS:
        try:
            count = page.locator(selector).count()
            if count > 0:
                evidences.append(Evidence(kind="dom", value=f"{selector} (count={count})", provider_guess=None))
        except Exception:
            continue

    return evidences


def detect_text_keywords(page: Page) -> List[Evidence]:
    """Détection via texte visible + title (case insensitive)."""
    evidences: List[Evidence] = []

    try:
        title = page.title() or ""
    except Exception:
        title = ""

    try:
        body = page.inner_text("body")
    except Exception:
        body = ""

    haystack = f"{title}\n{body}".lower()
    for kw in KEYWORDS_CHALLENGE:
        # Comparaison case insensitive (robuste même si le keyword a des majuscules)
        if kw.lower() in haystack:
            evidences.append(Evidence(kind="keyword", value=kw, provider_guess=None))

    return evidences


def detect_url_hints(page: Page) -> List[Evidence]:
    """Détection via URL finale."""
    evidences: List[Evidence] = []
    url = (page.url or "").lower()

    for hint in URL_HINTS:
        if hint in url:
            evidences.append(Evidence(kind="url", value=hint, provider_guess=None))

    return evidences


# =====================================================================
# 5) Localisation (widget/iframe/image/input)
# =====================================================================

def find_provider_widget_location(page: Page) -> Tuple[Optional[str], Optional[ElementLocation], Optional[ElementLocation]]:
    """
    Tente de localiser un widget provider + une iframe provider (si visible dans le DOM).
    Retourne (provider, widget_location, iframe_location).
    """
    # 1) Trouver provider via selectors widget
    for provider, selector in PROVIDER_WIDGET_SELECTORS:
        loc = page.locator(selector)
        try:
            if loc.count() > 0:
                widget_location = _make_location("widget", selector, loc.first)

                # 2) Essayer de trouver l'iframe correspondante
                iframe_location = None
                for p2, iframe_sel in PROVIDER_IFRAME_SELECTORS:
                    if p2 != provider:
                        continue
                    iframe_loc = page.locator(iframe_sel)
                    if iframe_loc.count() > 0:
                        iframe_location = _make_location("iframe", iframe_sel, iframe_loc.first)
                        break

                return provider, widget_location, iframe_location
        except Exception:
            continue

    return None, None, None


def _pick_best_generic_image(page: Page, scope: Optional[Locator] = None) -> Optional[Tuple[str, Locator]]:
    """
    Choisit une image CAPTCHA "maison" (dans scope si fourni).
    On prend la première qui matche nos selectors (ordre = priorité).
    Filtre les images trop petites (spinners/icônes < 40x40 pixels).
    """
    base = scope if scope is not None else page
    MIN_CAPTCHA_SIZE = 40  # pixels minimum pour largeur OU hauteur

    for sel in GENERIC_CAPTCHA_IMAGE_SELECTORS:
        loc = base.locator(sel)
        count = loc.count()
        if count > 0:
            # Essayer chaque candidat et prendre le premier assez grand
            for i in range(count):
                candidate = loc.nth(i)
                try:
                    bbox = candidate.bounding_box()
                    if bbox:
                        # Vérifier taille minimum (éviter spinners 18x18, icônes, etc.)
                        width = bbox.get('width', 0)
                        height = bbox.get('height', 0)
                        if width >= MIN_CAPTCHA_SIZE and height >= MIN_CAPTCHA_SIZE:
                            return sel, candidate
                except Exception:
                    # Si erreur bbox, continuer vers le prochain candidat
                    pass

    # Si aucune image >= 40x40 trouvée, retourner None plutôt qu'un spinner
    return None


def _pick_best_generic_input(page: Page, scope: Optional[Locator] = None) -> Optional[Tuple[str, Locator]]:
    """
    Choisit un input CAPTCHA "maison" (dans scope si fourni).
    Ne retourne jamais un input hidden.
    """
    base = scope if scope is not None else page

    for sel in GENERIC_CAPTCHA_INPUT_SELECTORS:
        loc = base.locator(sel)
        count = loc.count()
        if count == 0:
            continue

        # Parcourir les candidats et prendre le premier visible (non-hidden)
        for i in range(count):
            candidate = loc.nth(i)
            try:
                t = (candidate.get_attribute("type") or "").lower()
                if t == "hidden":
                    continue
                # Verifier que l'element est visible
                if not candidate.is_visible():
                    continue
                return sel, candidate
            except Exception:
                continue

    return None


def find_generic_captcha_locations(page: Page) -> Tuple[Optional[ElementLocation], Optional[ElementLocation]]:
    """
    Localise un CAPTCHA "maison" :
    - captcha_image_location
    - captcha_input_location

    Stratégie de proximité (du plus proche au plus général) :
    1) Trouver un input captcha
    2) Remonter vers un parent "form" et chercher une image dedans
    3) Remonter vers un ancêtre avec class contenant "captcha" et chercher dedans
    4) Remonter vers un ancêtre avec id contenant "captcha" et chercher dedans
    5) Sinon fallback global page
    """
    input_pick = _pick_best_generic_input(page, scope=None)
    input_location = None
    image_location = None

    # 1) Input en global
    if input_pick:
        input_sel, input_loc = input_pick
        input_location = _make_location("captcha_input", input_sel, input_loc)

        # 2) Chercher image proche : on remonte au form (si possible)
        try:
            form_loc = input_loc.locator("xpath=ancestor::form[1]")
            if form_loc.count() > 0:
                # chercher l'image dans le form
                image_pick = _pick_best_generic_image(page, scope=form_loc.first)
                if image_pick:
                    img_sel, img_loc = image_pick
                    image_location = _make_location("captcha_image", img_sel, img_loc)
        except Exception:
            pass

        # 3) Sinon, chercher dans un container avec class ou id contenant "captcha"
        if image_location is None:
            try:
                # Chercher d'abord par class
                captcha_container = input_loc.locator("xpath=ancestor::*[contains(@class,'captcha') or contains(@class,'Captcha') or contains(@class,'CAPTCHA')][1]")
                if captcha_container.count() > 0:
                    image_pick = _pick_best_generic_image(page, scope=captcha_container.first)
                    if image_pick:
                        img_sel, img_loc = image_pick
                        image_location = _make_location("captcha_image", img_sel, img_loc)
            except Exception:
                pass

        # 4) Sinon, chercher dans un container avec id contenant "captcha"
        if image_location is None:
            try:
                captcha_container = input_loc.locator("xpath=ancestor::*[contains(@id,'captcha') or contains(@id,'Captcha') or contains(@id,'CAPTCHA')][1]")
                if captcha_container.count() > 0:
                    image_pick = _pick_best_generic_image(page, scope=captcha_container.first)
                    if image_pick:
                        img_sel, img_loc = image_pick
                        image_location = _make_location("captcha_image", img_sel, img_loc)
            except Exception:
                pass

    # 5) Fallback global : image puis input si non trouvés
    if image_location is None:
        image_pick = _pick_best_generic_image(page, scope=None)
        if image_pick:
            img_sel, img_loc = image_pick
            image_location = _make_location("captcha_image", img_sel, img_loc)

    if input_location is None:
        input_pick2 = _pick_best_generic_input(page, scope=None)
        if input_pick2:
            in_sel, in_loc = input_pick2
            input_location = _make_location("captcha_input", in_sel, in_loc)

    return image_location, input_location


# =====================================================================
# 6) Agrégateur global
# =====================================================================

def detect_captcha(page: Page, min_score: int = 2) -> CaptchaDetectionResult:
    """
    Détection globale multi-indices.

    - On collecte des preuves : iframe + dom + keywords + url
    - On calcule un score = nb de familles présentes
    - On localise :
        * provider widget/iframe
        * captcha maison image/input
        * challenge signals
    """
    evidences: List[Evidence] = []
    evidences += detect_iframe_providers(page)
    evidences += detect_dom_markers(page)
    evidences += detect_text_keywords(page)
    evidences += detect_url_hints(page)

    kinds_present = set(e.kind for e in evidences)
    score = len(kinds_present)
    detected = score >= min_score

    # Catégorie
    if not detected:
        category = "none"
    else:
        # S'il y a iframe ou dom -> widget ; sinon keyword/url -> challenge
        if any(e.kind in ("iframe", "dom") for e in evidences):
            category = "captcha_widget"
        else:
            category = "challenge_page"

    # Provider (priorité aux preuves iframe/dom provider)
    provider = None
    for e in evidences:
        if e.provider_guess:
            provider = e.provider_guess
            break
    if detected and provider is None:
        provider = "unknown"
    
    # Visibilité (utile surtout pour reCAPTCHA)
    visibility = None
    if detected and provider == "recaptcha":
        # Si une iframe reCAPTCHA contient "size=invisible", alors le captcha est "invisible"
        for e in evidences:
            if e.kind == "iframe" and "size=invisible" in e.value:
                visibility = "invisible"
                break
        # Sinon, on considère que c'est visible (checkbox / widget classique)
        if visibility is None:
            visibility = "visible"


    widget_location = None
    iframe_location = None
    captcha_image_location = None
    captcha_input_location = None
    challenge = None

    if detected and category == "captcha_widget":
        # 1) Essayer provider
        prov, wloc, iloc = find_provider_widget_location(page)
        if prov:
            provider = prov
            widget_location = wloc
            iframe_location = iloc
        else:
            # 2) Sinon captcha maison (image + input)
            captcha_image_location, captcha_input_location = find_generic_captcha_locations(page)

    if detected and category == "challenge_page":
        # Construire des signaux exploitables pour logs/rapport
        final_url = page.url or ""
        try:
            title = page.title() or ""
        except Exception:
            title = ""

        matched_keywords = [e.value for e in evidences if e.kind == "keyword"]
        matched_url_hints = [e.value for e in evidences if e.kind == "url"]

        challenge = ChallengeSignals(
            final_url=final_url,
            title=title,
            matched_keywords=matched_keywords,
            matched_url_hints=matched_url_hints,
        )

    return CaptchaDetectionResult(
        detected=detected,
        category=category,
        provider=provider,
        score=score,
        evidences=evidences,
        visibility=visibility,
        widget_location=widget_location,
        iframe_location=iframe_location,
        captcha_image_location=captcha_image_location,
        captcha_input_location=captcha_input_location,
        challenge=challenge,
    )
