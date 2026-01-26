from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
from playwright.sync_api import sync_playwright, Browser, BrowserContext, Playwright


@dataclass
class BrowserConfig:
    headless: bool = True
    browser_name: str = "chromium"  # "chromium" | "firefox" | "webkit"
    viewport_width: int = 1280
    viewport_height: int = 720
    locale: str = "fr-FR"
    timezone_id: str = "Europe/Paris"
    user_agent: Optional[str] = None

    # Timeouts (ms)
    default_timeout_ms: int = 15000
    default_navigation_timeout_ms: int = 25000


    java_script_enabled: bool = True
    ignore_https_errors: bool = True


def create_browser_and_context(
    config: BrowserConfig,
) -> Tuple[Playwright, Browser, BrowserContext]:
    """
    Crée Playwright + Browser + BrowserContext.
    """
    pw = sync_playwright().start()

    # Choix du navigateur
    if config.browser_name == "chromium":
        browser = pw.chromium.launch(headless=config.headless)
    elif config.browser_name == "firefox":
        browser = pw.firefox.launch(headless=config.headless)
    elif config.browser_name == "webkit":
        browser = pw.webkit.launch(headless=config.headless)
    else:
        pw.stop()
        raise ValueError(
            f"browser_name invalide: {config.browser_name}. "
            "Valeurs possibles: chromium | firefox | webkit"
        )

    context_kwargs = {
        "viewport": {"width": config.viewport_width, "height": config.viewport_height},
        "locale": config.locale,
        "timezone_id": config.timezone_id,
        "java_script_enabled": config.java_script_enabled,
        "ignore_https_errors": config.ignore_https_errors,
    }

    # User-Agent custom si tu veux
    if config.user_agent:
        context_kwargs["user_agent"] = config.user_agent

    context = browser.new_context(**context_kwargs)

    # Timeouts par défaut
    context.set_default_timeout(config.default_timeout_ms)
    context.set_default_navigation_timeout(config.default_navigation_timeout_ms)

    return pw, browser, context


def close_browser(pw: Playwright, browser: Browser, context: BrowserContext) -> None:
    """
    Fermeture propre (évite les processes zombies).
    """
    try:
        context.close()
    finally:
        try:
            browser.close()
        finally:
            pw.stop()
