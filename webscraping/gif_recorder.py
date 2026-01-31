"""
gif_recorder.py -- Enregistrement de screenshots en GIF anime

Capture des screenshots du viewport a chaque etape cle du pipeline
de resolution de CAPTCHA (chargement, saisie caractere par caractere,
soumission, resultat) et les assemble en GIF anime avec Pillow.

Utilise par run_captcha_solver.py quand le flag --record est actif.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Optional

from PIL import Image
from playwright.sync_api import Page


class GifRecorder:
    """
    Accumule des screenshots du viewport avec leur duree d'affichage,
    puis les assemble en un GIF anime.
    """

    def __init__(self, scale: float = 0.5):
        """
        Args:
            scale: Facteur de redimensionnement (0.5 = 640x360 depuis 1280x720).
        """
        self._frames: list[tuple[Image.Image, int]] = []
        self._scale = scale

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def capture(self, page: Page, duration_ms: int = 200) -> None:
        """
        Prend un screenshot du viewport et le stocke comme frame.

        Args:
            page: Page Playwright active.
            duration_ms: Duree d'affichage de cette frame dans le GIF (ms).
        """
        try:
            png_bytes = page.screenshot(full_page=False)
            img = Image.open(BytesIO(png_bytes)).convert("RGB")

            if self._scale != 1.0:
                new_w = int(img.width * self._scale)
                new_h = int(img.height * self._scale)
                img = img.resize((new_w, new_h), Image.LANCZOS)

            self._frames.append((img, duration_ms))
        except Exception:
            pass  # ne pas crasher le pipeline si un screenshot echoue

    def save(self, path: str | Path) -> Optional[str]:
        """
        Assemble les frames en GIF anime.

        Returns:
            Le chemin du fichier si sauvegarde reussie, None sinon.
        """
        if not self._frames:
            return None

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        first_frame = self._frames[0][0]
        remaining = [f[0] for f in self._frames[1:]]
        durations = [f[1] for f in self._frames]

        first_frame.save(
            str(path),
            save_all=True,
            append_images=remaining,
            duration=durations,
            loop=0,
            optimize=True,
        )

        return str(path)

    def discard(self) -> None:
        """Vide les frames (appele entre chaque tentative de retry)."""
        self._frames.clear()
