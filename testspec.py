"""Tiny PyInstaller splash-screen test for CI Deconvolve.

Build with:
    pyinstaller packaging/test.spec

The packaged executable updates the PyInstaller splash text, closes the splash,
and then opens a small stdlib Tk window so the close timing is visible.
"""

from __future__ import annotations

import time
import tkinter as tk


def _update_splash(text: str) -> None:
    try:
        import pyi_splash  # type: ignore
    except Exception:
        print(text, flush=True)
        return
    try:
        pyi_splash.update_text(text)
    except Exception:
        pass


def _close_splash() -> None:
    try:
        import pyi_splash  # type: ignore
    except Exception:
        return
    try:
        pyi_splash.close()
    except Exception:
        pass


def main() -> None:
    steps = (
        "10% - Bootloader splash is visible...",
        "35% - Updating text on the splash...",
        "65% - Preparing to close splash...",
        "100% - Closing splash now...",
    )
    for step in steps:
        _update_splash(step)
        time.sleep(0.75)

    _close_splash()

    root = tk.Tk()
    root.title("CI Deconvolve splash test")
    root.geometry("360x120")
    root.resizable(False, False)

    label = tk.Label(
        root,
        text="Splash closed successfully.",
        font=("Segoe UI", 12),
        padx=24,
        pady=22,
    )
    label.pack(expand=True, fill="both")

    button = tk.Button(root, text="Close", command=root.destroy)
    button.pack(pady=(0, 14))
    root.after(5000, root.destroy)
    root.mainloop()


if __name__ == "__main__":
    main()
