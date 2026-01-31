"""
fill.py – Saisie humaine du CAPTCHA et soumission du formulaire

Simule un humain qui tape le texte caractere par caractere
avec des delais aleatoires, puis cherche le bouton submit
(par type, par texte : Valider, Envoyer, Submit...) et soumet.
Retourne un FillResult.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from playwright.sync_api import Page, Locator


# ----------------------------
# Résultat
# ----------------------------

@dataclass
class FillResult:
    ok: bool
    text_typed: str = ""
    submit_clicked: bool = False
    error: Optional[str] = None


# ----------------------------
# Recherche du bouton submit
# ----------------------------

# Textes courants sur les boutons de soumission (FR + EN)
_SUBMIT_LABELS = [
    "Valider", "Envoyer", "Submit", "Vérifier", "Verify",
    "OK", "Confirmer", "Confirm", "Continuer", "Continue",
]


def _find_submit_button(page: Page, input_locator: Locator) -> Optional[Locator]:
    """
    Cherche le bouton de soumission du formulaire contenant l'input CAPTCHA.

    Stratégie :
    1. Remonter au <form> parent, chercher button[type=submit] / input[type=submit]
    2. Si pas trouvé dans le form, chercher des boutons par texte (Valider, Submit, etc.)
    3. Retourne None si rien trouvé (le caller fera Enter comme fallback)
    """
    # 1) Chercher dans le <form> parent via JS
    try:
        form = input_locator.evaluate("el => el.closest('form')")
        if form:
            form_loc = input_locator.locator("xpath=ancestor::form[1]")

            # button[type=submit] ou input[type=submit]
            for sel in ["button[type='submit']", "input[type='submit']"]:
                btn = form_loc.locator(sel).first
                if btn.count() and btn.is_visible():
                    return btn

            # Tout <button> dans le form (souvent le submit n'a pas type=submit)
            btn = form_loc.locator("button").first
            if btn.count() and btn.is_visible():
                return btn
    except Exception:
        pass

    # 2) Chercher par texte sur toute la page
    for label in _SUBMIT_LABELS:
        for tag in ["button", "input[type='submit']", "a"]:
            try:
                loc = page.locator(f"{tag}:has-text('{label}')").first
                if loc.count() and loc.is_visible():
                    return loc
            except Exception:
                continue

    return None


# ----------------------------
# Saisie humaine + soumission
# ----------------------------

def fill_and_submit(
    page: Page,
    input_selector: str,
    text: str,
    input_xpath: Optional[str] = None,
    min_delay_ms: int = 50,
    max_delay_ms: int = 250,
    submit: bool = True,
) -> FillResult:
    """
    Tape le texte résolu dans le champ input comme un humain, puis soumet le formulaire.

    Args:
        page: Page Playwright active
        input_selector: Sélecteur CSS du champ input (depuis ExtractResult.input_selector)
        text: Texte à taper (résultat de l'OCR)
        input_xpath: XPath alternatif (fallback si le CSS échoue)
        min_delay_ms: Délai minimum entre chaque frappe (ms)
        max_delay_ms: Délai maximum entre chaque frappe (ms)
        submit: Si True, cherche et clique le bouton submit après la saisie

    Returns:
        FillResult avec le statut de l'opération
    """
    if not text:
        return FillResult(ok=False, error="Texte vide, rien à taper")

    # 1) Localiser l'input
    input_loc = None
    try:
        input_loc = page.locator(input_selector).first
        if not input_loc.count():
            input_loc = None
    except Exception:
        input_loc = None

    # Fallback XPath
    if input_loc is None and input_xpath:
        try:
            input_loc = page.locator(f"xpath={input_xpath}").first
            if not input_loc.count():
                input_loc = None
        except Exception:
            input_loc = None

    if input_loc is None:
        return FillResult(ok=False, error=f"Input introuvable: {input_selector}")

    try:
        # 2) Cliquer sur l'input pour le focus
        page.wait_for_timeout(random.randint(200, 500))
        input_loc.click()
        page.wait_for_timeout(random.randint(100, 300))

        # 3) Vider le champ (au cas où il y aurait du texte pré-rempli)
        input_loc.fill("")
        page.wait_for_timeout(random.randint(50, 150))

        # 4) Taper caractère par caractère avec délais aléatoires
        for char in text:
            input_loc.type(char)
            delay = random.randint(min_delay_ms, max_delay_ms)
            page.wait_for_timeout(delay)

        # 5) Pause après saisie (humain qui relit)
        page.wait_for_timeout(random.randint(300, 800))

        # 6) Soumettre le formulaire
        submit_clicked = False
        if submit:
            btn = _find_submit_button(page, input_loc)
            if btn:
                page.wait_for_timeout(random.randint(150, 400))
                btn.click()
                submit_clicked = True
            else:
                # Fallback : appuyer sur Entrée
                input_loc.press("Enter")
                submit_clicked = True

        return FillResult(
            ok=True,
            text_typed=text,
            submit_clicked=submit_clicked,
        )

    except Exception as exc:
        return FillResult(ok=False, text_typed=text, error=str(exc))
