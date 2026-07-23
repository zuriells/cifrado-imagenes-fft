#!/usr/bin/env python3
"""Full evaluation suite for fft_image_cipher: quality metrics (histograms,
entropy, PSNR) and robustness attacks (Gaussian noise, resizing, wrong
password), mirroring the analysis style of frequency-domain image
encryption papers.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from fft_image_cipher import (
    PBKDF2_ITERATIONS,
    SALT_BYTES,
    derive_key_image,
    fft_decrypt,
    fft_encrypt,
    load_png16,
    load_rgb_image,
    save_png16,
)

IMAGE_PATH = Path("prueba2.png")
PASSWORD = "Cript0Analisis2026!"
WRONG_PASSWORD = "ContrasenaIncorrecta"
OUT = Path(".")

results: dict = {}


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse)), mse


def entropy(image_uint8: np.ndarray) -> float:
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


def main() -> None:
    image = load_rgb_image(IMAGE_PATH)
    image_uint8 = image.astype(np.uint8)
    height, width = image.shape[:2]
    print(f"Imagen de prueba: {IMAGE_PATH} ({width}x{height})")

    # ---- Baseline encryption ----
    salt = os.urandom(SALT_BYTES)
    key = derive_key_image(PASSWORD, salt, image.shape, PBKDF2_ITERATIONS)
    encrypted, channel_ranges = fft_encrypt(image, key)
    metadata = {
        "ranges": channel_ranges,
        "salt": base64.b64encode(salt).decode("ascii"),
        "iterations": PBKDF2_ITERATIONS,
    }
    save_png16(OUT / "eval_encrypted.png", encrypted, metadata)
    save_preview(image_uint8, OUT / "eval_original.jpg")
    save_preview(encrypted_preview(encrypted), OUT / "eval_encrypted.jpg")

    # ---- Correct-password decryption (quality baseline) ----
    key_dec = derive_key_image(PASSWORD, salt, encrypted.shape, metadata["iterations"])
    decrypted = fft_decrypt(encrypted, key_dec, metadata["ranges"])
    p, mse = psnr(image_uint8, decrypted)
    results["psnr_correct"] = p
    results["mse_correct"] = mse
    save_preview(decrypted, OUT / "eval_correct.jpg")
    print(f"[Calidad] Contrasena correcta -> MSE={mse:.6f} PSNR={p}")

    # ---- Wrong-password decryption ----
    wrong_key = derive_key_image(WRONG_PASSWORD, salt, encrypted.shape, metadata["iterations"])
    wrong_decrypted = fft_decrypt(encrypted, wrong_key, metadata["ranges"])
    p_wrong, mse_wrong = psnr(image_uint8, wrong_decrypted)
    results["psnr_wrong_password"] = p_wrong
    results["mse_wrong_password"] = mse_wrong
    save_preview(wrong_decrypted, OUT / "eval_wrong_password.jpg")
    print(f"[Seguridad] Contrasena incorrecta -> MSE={mse_wrong:.6f} PSNR={p_wrong}")

    # ---- Histograms ----
    plot_histograms(image_uint8, "Imagen original / Original image", OUT / "hist_original.png")
    plot_histograms(encrypted_preview(encrypted), "Imagen cifrada / Encrypted image", OUT / "hist_encrypted.png")

    # ---- Entropy ----
    ent_orig, ent_orig_ch = entropy(image_uint8)
    ent_enc, ent_enc_ch = entropy(encrypted_preview(encrypted))
    results["entropy_original"] = ent_orig
    results["entropy_original_channels"] = ent_orig_ch
    results["entropy_encrypted"] = ent_enc
    results["entropy_encrypted_channels"] = ent_enc_ch
    print(f"[Entropia] Original={ent_orig:.4f}  Cifrada={ent_enc:.4f}")

    # ---- Gaussian noise attack ----
    noise_results = {}
    rng = np.random.default_rng(42)
    for label, sigma_fraction in [("bajo", 0.0005), ("medio", 0.0025), ("alto", 0.005)]:
        sigma = sigma_fraction * 65535.0
        noisy = encrypted.astype(np.float64) + rng.normal(0, sigma, size=encrypted.shape)
        noisy = np.clip(np.rint(noisy), 0, 65535).astype(np.uint16)
        noisy_decrypted = fft_decrypt(noisy, key_dec, metadata["ranges"])
        p_noise, mse_noise = psnr(image_uint8, noisy_decrypted)
        noise_results[label] = {"sigma_fraction": sigma_fraction, "sigma_abs": sigma, "psnr": p_noise, "mse": mse_noise}
        save_preview(noisy_decrypted, OUT / f"eval_noise_{label}.jpg")
        print(f"[Ruido {label}] sigma={sigma:.1f} (16-bit) -> MSE={mse_noise:.4f} PSNR={p_noise}")
    results["noise_attack"] = noise_results

    # ---- Resize (downscale) attack ----
    resize_down = {}
    for pct in [1, 5, 10]:
        new_h = max(1, round(height * pct / 100))
        new_w = max(1, round(width * pct / 100))
        small = resize_uint16(encrypted, new_h, new_w)
        small_key = derive_key_image(PASSWORD, salt, small.shape, metadata["iterations"])
        small_decrypted = fft_decrypt(small, small_key, metadata["ranges"])
        save_preview(small_decrypted, OUT / f"eval_downscale_{pct}.jpg", max_width=min(300, new_w))
        resize_down[pct] = {"height": new_h, "width": new_w}
        print(f"[Downscale {pct}%] -> {new_w}x{new_h} guardado")
    results["resize_down"] = resize_down

    # ---- Resize (upscale) attack ----
    resize_up = {}
    for pct in [1, 5, 10]:
        new_h = round(height * (1 + pct / 100))
        new_w = round(width * (1 + pct / 100))
        big = resize_uint16(encrypted, new_h, new_w)
        big_key = derive_key_image(PASSWORD, salt, big.shape, metadata["iterations"])
        big_decrypted = fft_decrypt(big, big_key, metadata["ranges"])
        save_preview(big_decrypted, OUT / f"eval_upscale_{pct}.jpg")
        resize_up[pct] = {"height": new_h, "width": new_w}
        print(f"[Upscale {pct}%] -> {new_w}x{new_h} guardado")
    results["resize_up"] = resize_up

    (OUT / "eval_results.json").write_text(json.dumps(results, indent=2))
    print("\nResultados guardados en eval_results.json")


if __name__ == "__main__":
    main()
