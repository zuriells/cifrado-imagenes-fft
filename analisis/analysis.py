#!/usr/bin/env python3
"""Evaluacion de fft_image_cipher (esquema con cifrado autenticado).

Mide la calidad (histogramas, entropia, PSNR de la recuperacion exacta) y
la seguridad frente a manipulaciones. A diferencia de un cifrado lineal
puro, aqui la confidencialidad y la integridad las aporta
ChaCha20-Poly1305: por eso las "pruebas de robustez" ya no producen una
imagen degradada, sino un RECHAZO por fallo de autenticacion. El script lo
verifica para contrasena incorrecta, ruido gaussiano y reescalado.

Uso:
    python analysis.py [imagen] [--out DIRECTORIO]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from cryptography.exceptions import InvalidTag
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fft_image_cipher import (  # noqa: E402
    ITERACIONES_PBKDF2,
    cargar_imagen_rgb,
    cifrar,
    confusion_difusion,
    derivar_imagen_clave,
    derivar_secreto_maestro,
    descifrar,
    fft_blanquear,
)

# NPCR/UACI ideales para imagenes de 8 bits (Wu et al., 2011).
NPCR_IDEAL = 99.6094
UACI_IDEAL = 33.4635

PASSWORD = "Cript0Analisis2026!"
WRONG_PASSWORD = "ContrasenaIncorrecta"

results: dict = {}


def psnr(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return (float("inf") if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))), mse


def entropy(image_uint8: np.ndarray) -> tuple[float, list[float]]:
    channel_entropies = []
    for c in range(3):
        hist, _ = np.histogram(image_uint8[:, :, c], bins=256, range=(0, 256))
        probs = hist / hist.sum()
        probs = probs[probs > 0]
        channel_entropies.append(float(-np.sum(probs * np.log2(probs))))
    return float(np.mean(channel_entropies)), channel_entropies


def save_preview(array_uint8: np.ndarray, path: Path, max_width: int = 450) -> None:
    img = Image.fromarray(array_uint8, mode="RGB")
    w, h = img.size
    scale = max_width / w
    img.resize((max_width, int(h * scale)), Image.Resampling.LANCZOS).save(path, quality=88)


def encrypted_preview(encrypted_uint16: np.ndarray) -> np.ndarray:
    """Vista previa de 8 bits (byte alto) de los datos cifrados de 16 bits."""
    return (encrypted_uint16 >> 8).astype(np.uint8)


def plot_histograms(image_uint8: np.ndarray, title: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.6))
    colors = ["red", "green", "blue"]
    names = ["Rojo / Red", "Verde / Green", "Azul / Blue"]
    for c, (ax, color, name) in enumerate(zip(axes, colors, names)):
        ax.hist(image_uint8[:, :, c].ravel(), bins=256, range=(0, 256), color=color, alpha=0.8)
        ax.set_title(name, fontsize=9)
        ax.set_xlim(0, 255)
        ax.set_yticklabels([])
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def resize_uint16(array: np.ndarray, new_height: int, new_width: int) -> np.ndarray:
    resized = np.empty((new_height, new_width, 3), dtype=np.uint16)
    for c in range(3):
        channel_img = Image.fromarray(array[:, :, c].astype(np.int32), mode="I")
        channel_img = channel_img.resize((new_width, new_height), Image.Resampling.BILINEAR)
        resized[:, :, c] = np.clip(np.asarray(channel_img, dtype=np.int64), 0, 65535).astype(np.uint16)
    return resized


def intento_descifrado(encrypted: np.ndarray, metadata: dict, password: str) -> str:
    """Devuelve 'aceptado' si el descifrado autentica, o 'rechazado' si
    ChaCha20-Poly1305 detecta contrasena incorrecta o manipulacion."""
    try:
        descifrar(encrypted, metadata, password)
        return "aceptado"
    except (InvalidTag, ValueError):
        return "rechazado"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluacion del cifrado autenticado FFT.")
    parser.add_argument("imagen", nargs="?", default=str(Path(__file__).resolve().parent.parent / "ejemplos" / "prueba2.png"))
    parser.add_argument("--out", default=".")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    image = cargar_imagen_rgb(Path(args.imagen))
    image_uint8 = image.astype(np.uint8)
    height, width = image.shape[:2]
    print(f"Imagen de prueba: {args.imagen} ({width}x{height})")

    # ---- Cifrado base (con cifrado autenticado) ----
    encrypted, metadata = cifrar(image, PASSWORD)
    save_preview(image_uint8, out / "eval_original.jpg")
    save_preview(encrypted_preview(encrypted), out / "eval_encrypted.jpg")

    # ---- Descifrado con contrasena correcta (calidad) ----
    decrypted = descifrar(encrypted, metadata, PASSWORD)
    p, mse = psnr(image_uint8, decrypted)
    results["psnr_correct"] = p
    results["mse_correct"] = mse
    results["exact_recovery"] = bool(np.array_equal(image_uint8, decrypted))
    save_preview(decrypted, out / "eval_correct.jpg")
    print(f"[Calidad] Contrasena correcta -> MSE={mse:.6f} PSNR={p} exacta={results['exact_recovery']}")

    # ---- Histogramas ----
    plot_histograms(image_uint8, "Imagen original / Original image", out / "hist_original.png")
    plot_histograms(encrypted_preview(encrypted), "Imagen cifrada / Encrypted image", out / "hist_encrypted.png")

    # ---- Entropia ----
    ent_orig, ent_orig_ch = entropy(image_uint8)
    ent_enc, ent_enc_ch = entropy(encrypted_preview(encrypted))
    results["entropy_original"] = ent_orig
    results["entropy_original_channels"] = ent_orig_ch
    results["entropy_encrypted"] = ent_enc
    results["entropy_encrypted_channels"] = ent_enc_ch
    print(f"[Entropia] Original={ent_orig:.4f}  Cifrada={ent_enc:.4f}")

    # ---- Difusion del nucleo (blanqueo + confusion-difusion), antes del
    # sobre AEAD: NPCR/UACI y sensibilidad a la clave. Se fijan salt y nonce
    # para aislar el efecto del cambio de un pixel (misma clave) ----
    salt_fijo = b"\x00" * 16
    nonce_fijo = b"\x00" * 12
    secreto = derivar_secreto_maestro(PASSWORD, salt_fijo, ITERACIONES_PBKDF2)
    g = derivar_imagen_clave(secreto, image.shape)

    def nucleo(f: np.ndarray) -> np.ndarray:
        blanqueada, _ = fft_blanquear(f, g)
        return np.frombuffer(
            confusion_difusion(blanqueada.astype(">u2").tobytes(), secreto, nonce_fijo), np.uint8
        )

    f_mod = image.copy()
    f_mod[height // 2, width // 2, 0] = (f_mod[height // 2, width // 2, 0] + 1) % 256
    c1 = nucleo(image)
    c2 = nucleo(f_mod)
    npcr = float(100.0 * np.mean(c1 != c2))
    uaci = float(100.0 * np.mean(np.abs(c1.astype(np.int32) - c2.astype(np.int32)) / 255.0))

    secreto_alt = bytearray(secreto)
    secreto_alt[0] ^= 1
    g_alt = derivar_imagen_clave(bytes(secreto_alt), image.shape)
    bl_alt, _ = fft_blanquear(image, g_alt)
    c_key = np.frombuffer(confusion_difusion(bl_alt.astype(">u2").tobytes(), bytes(secreto_alt), nonce_fijo), np.uint8)
    key_sensitivity = float(100.0 * np.mean(c1 != c_key))

    results["diffusion"] = {
        "npcr": npcr,
        "npcr_ideal": NPCR_IDEAL,
        "uaci": uaci,
        "uaci_ideal": UACI_IDEAL,
        "key_sensitivity_percent": key_sensitivity,
    }
    print(f"[Difusion] NPCR={npcr:.4f}% (ideal {NPCR_IDEAL}) UACI={uaci:.4f}% (ideal {UACI_IDEAL})")
    print(f"[Difusion] Sensibilidad de clave (1 bit) = {key_sensitivity:.4f}%")

    # ---- Seguridad: todo intento sobre datos alterados debe RECHAZARSE ----
    auth: dict = {}

    # Contrasena incorrecta
    auth["wrong_password"] = intento_descifrado(encrypted, metadata, WRONG_PASSWORD)
    print(f"[Auth] Contrasena incorrecta -> {auth['wrong_password']}")

    # Ruido gaussiano sobre el archivo cifrado
    noise = {}
    rng = np.random.default_rng(42)
    for label, sigma_fraction in [("bajo", 0.0005), ("medio", 0.0025), ("alto", 0.005)]:
        sigma = sigma_fraction * 65535.0
        noisy = encrypted.astype(np.float64) + rng.normal(0, sigma, size=encrypted.shape)
        noisy = np.clip(np.rint(noisy), 0, 65535).astype(np.uint16)
        noise[label] = {
            "sigma_fraction": sigma_fraction,
            "sigma_abs": sigma,
            "outcome": intento_descifrado(noisy, metadata, PASSWORD),
        }
        print(f"[Auth] Ruido {label} (sigma={sigma:.1f}) -> {noise[label]['outcome']}")
    auth["gaussian_noise"] = noise

    # Reescalado del archivo cifrado
    resize = {}
    for tag, factors in [("down", [1, 5, 10]), ("up", [1, 5, 10])]:
        resize[tag] = {}
        for pct in factors:
            if tag == "down":
                new_h = max(1, round(height * pct / 100))
                new_w = max(1, round(width * pct / 100))
            else:
                new_h = round(height * (1 + pct / 100))
                new_w = round(width * (1 + pct / 100))
            resized = resize_uint16(encrypted, new_h, new_w)
            resize[tag][str(pct)] = {
                "height": new_h,
                "width": new_w,
                "outcome": intento_descifrado(resized, metadata, PASSWORD),
            }
            print(f"[Auth] Reescalado {tag} {pct}% ({new_w}x{new_h}) -> {resize[tag][str(pct)]['outcome']}")
    auth["resize"] = resize
    results["authentication"] = auth

    # ---- Bonus: sensibilidad a 1 bit (efecto avalancha del tag) ----
    one_bit = encrypted.copy()
    one_bit[0, 0, 0] ^= 1
    results["single_bit_flip"] = intento_descifrado(one_bit, metadata, PASSWORD)
    print(f"[Auth] Volteo de 1 bit -> {results['single_bit_flip']}")

    (out / "eval_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nResultados guardados en {out / 'eval_results.json'}")


if __name__ == "__main__":
    main()
