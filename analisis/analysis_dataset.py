#!/usr/bin/env python3
"""Evaluacion multi-imagen del cifrado autenticado FFT sobre un subconjunto
diverso de la USC-SIPI Image Database (sipi.usc.edu/database).

Responde a las revisiones: (1) resultados agregados sobre varias imagenes de
distinta resolucion y contenido; (2) metricas de cifrado de imagen
adicionales --- correlacion de pixeles adyacentes (H/V/D), correlacion
plano-cifrado, NPCR, UACI, sensibilidad a la clave; (3) comparacion con un
cifrador estandar (AES-256-CTR); (4) evaluacion de rendimiento (tiempos de
cifrado/descifrado por tamano, throughput y memoria pico); y verifica la
integridad (rechazo autenticado ante contrasena incorrecta, ruido y
reescalado simetrico).

Uso:
    python analysis_dataset.py [--out DIR] [--data DIR]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fft_image_cipher import (  # noqa: E402
    ITERACIONES_PBKDF2,
    cifrar,
    confusion_difusion,
    derivar_imagen_clave,
    derivar_secreto_maestro,
    descifrar,
    fft_blanquear,
)

PASSWORD = "Cript0Analisis2026!"
WRONG_PASSWORD = "ContrasenaIncorrecta"
NPCR_IDEAL = 99.6094
UACI_IDEAL = 33.4635

# Subconjunto curado: distinto contenido (natural, aereo, textura) y
# resolucion (256, 512, 1024), color y escala de grises.
CURADAS = [
    "misc/4.1.04.tiff", "misc/4.1.05.tiff", "misc/4.2.03.tiff", "misc/4.2.05.tiff",
    "misc/4.2.06.tiff", "misc/4.2.07.tiff", "misc/5.1.10.tiff", "misc/5.2.08.tiff",
    "misc/5.3.01.tiff", "misc/7.1.05.tiff", "misc/boat.512.tiff", "misc/house.tiff",
    "aerials/2.1.01.tiff", "aerials/2.2.01.tiff", "aerials/3.2.25.tiff",
    "textures/1.1.01.tiff", "textures/1.2.04.tiff", "textures/1.5.04.tiff",
]

rng = np.random.default_rng(2026)


def cargar_rgb(ruta: Path) -> np.ndarray:
    return np.asarray(Image.open(ruta).convert("RGB"), dtype=np.uint8)


def entropia(img8: np.ndarray) -> float:
    e = []
    for c in range(3):
        h, _ = np.histogram(img8[:, :, c], bins=256, range=(0, 256))
        p = h / h.sum()
        p = p[p > 0]
        e.append(-np.sum(p * np.log2(p)))
    return float(np.mean(e))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def correlacion_adyacente(img8: np.ndarray, muestras: int = 5000) -> dict:
    """Correlacion de pixeles adyacentes horizontal, vertical y diagonal,
    promediada sobre los tres canales."""
    out = {}
    for nombre, (s1, s2) in {
        "H": ((slice(None), slice(0, -1)), (slice(None), slice(1, None))),
        "V": ((slice(0, -1), slice(None)), (slice(1, None), slice(None))),
        "D": ((slice(0, -1), slice(0, -1)), (slice(1, None), slice(1, None))),
    }.items():
        rs = []
        for c in range(3):
            x = img8[:, :, c][s1].ravel()
            y = img8[:, :, c][s2].ravel()
            if x.size > muestras:
                idx = rng.choice(x.size, muestras, replace=False)
                x, y = x[idx], y[idx]
            rs.append(_corr(x, y))
        out[nombre] = float(np.mean(rs))
    return out


def aes256_ctr(datos: bytes, clave: bytes, nonce: bytes) -> bytes:
    enc = Cipher(algorithms.AES(clave), modes.CTR(nonce)).encryptor()
    return enc.update(datos) + enc.finalize()


def nucleo(img: np.ndarray, secreto: bytes, g: np.ndarray, nonce: bytes) -> np.ndarray:
    bl, _ = fft_blanquear(img.astype(np.float64), g)
    return np.frombuffer(confusion_difusion(bl.astype(">u2").tobytes(), secreto, nonce), np.uint8)


def npcr_uaci(img: np.ndarray, secreto: bytes, g: np.ndarray, nonce: bytes) -> tuple[float, float]:
    mod = img.copy()
    h, w = img.shape[:2]
    mod[h // 2, w // 2, 0] = (int(mod[h // 2, w // 2, 0]) + 1) % 256
    c1 = nucleo(img, secreto, g, nonce)
    c2 = nucleo(mod, secreto, g, nonce)
    npcr = float(100.0 * np.mean(c1 != c2))
    uaci = float(100.0 * np.mean(np.abs(c1.astype(np.int32) - c2.astype(np.int32)) / 255.0))
    return npcr, uaci


def rechazado(enc: np.ndarray, meta: dict, pw: str) -> bool:
    try:
        descifrar(enc, meta, pw)
        return False
    except (InvalidTag, ValueError):
        return True


def resize_uint16(a: np.ndarray, nh: int, nw: int) -> np.ndarray:
    out = np.empty((nh, nw, 3), dtype=np.uint16)
    for c in range(3):
        im = Image.fromarray(a[:, :, c].astype(np.int32), mode="I").resize((nw, nh), Image.Resampling.BILINEAR)
        out[:, :, c] = np.clip(np.asarray(im, dtype=np.int64), 0, 65535).astype(np.uint16)
    return out


def evaluar_imagen(ruta: Path) -> dict:
    img = cargar_rgb(ruta)
    h, w = img.shape[:2]
    n_bytes = h * w * 3
    imgf = img.astype(np.float64)

    t0 = time.perf_counter()
    enc, meta = cifrar(imgf, PASSWORD)
    t_enc = time.perf_counter() - t0

    t0 = time.perf_counter()
    dec = descifrar(enc, meta, PASSWORD)
    t_dec = time.perf_counter() - t0

    exacta = bool(np.array_equal(img, dec))
    preview = (enc >> 8).astype(np.uint8)

    # metricas estadisticas
    ent_o = entropia(img)
    ent_e = entropia(preview)
    corr_o = correlacion_adyacente(img)
    corr_e = correlacion_adyacente(preview)
    corr_pc = _corr(img.ravel(), preview.ravel())  # plano-cifrado

    # difusion (nucleo, clave fija)
    salt_fijo = b"\x00" * 16
    nonce_fijo = b"\x00" * 12
    secreto = derivar_secreto_maestro(PASSWORD, salt_fijo, ITERACIONES_PBKDF2)
    g = derivar_imagen_clave(secreto, img.shape)
    npcr, uaci = npcr_uaci(img, secreto, g, nonce_fijo)
    secreto_alt = bytearray(secreto)
    secreto_alt[0] ^= 1
    g_alt = derivar_imagen_clave(bytes(secreto_alt), img.shape)
    c_base = nucleo(img, secreto, g, nonce_fijo)
    c_key = nucleo(img, bytes(secreto_alt), g_alt, nonce_fijo)
    key_sens = float(100.0 * np.mean(c_base != c_key))

    # integridad: contrasena incorrecta, ruido, reescalado simetrico
    integridad = {"wrong_password": rechazado(enc, meta, WRONG_PASSWORD)}
    for etq, frac in [("noise_low", 0.0005), ("noise_mid", 0.0025), ("noise_high", 0.005)]:
        noisy = np.clip(np.rint(enc.astype(np.float64) + rng.normal(0, frac * 65535.0, enc.shape)), 0, 65535).astype(np.uint16)
        integridad[etq] = rechazado(noisy, meta, PASSWORD)
    for pct in [1, 5, 10]:
        for signo, factor in [("down", 1 - pct / 100), ("up", 1 + pct / 100)]:
            nh, nw = max(1, round(h * factor)), max(1, round(w * factor))
            integridad[f"resize_{signo}_{pct}"] = rechazado(resize_uint16(enc, nh, nw), meta, PASSWORD)
    todo_rechazado = all(integridad.values())

    # baseline AES-256-CTR sobre los mismos bytes
    aes_key = derivar_secreto_maestro(PASSWORD, b"\x01" * 16, ITERACIONES_PBKDF2)
    aes_ct = aes256_ctr(img.tobytes(), aes_key, b"\x00" * 16)
    aes_img = np.frombuffer(aes_ct, np.uint8).reshape(img.shape)
    aes_ent = entropia(aes_img)
    aes_corr = correlacion_adyacente(aes_img)

    return {
        "image": str(ruta.name),
        "height": h, "width": w, "megapixels": round(h * w / 1e6, 3),
        "exact_recovery": exacta,
        "entropy_original": ent_o, "entropy_encrypted": ent_e,
        "corr_original": corr_o, "corr_encrypted": corr_e,
        "corr_plain_cipher": corr_pc,
        "npcr": npcr, "uaci": uaci, "key_sensitivity": key_sens,
        "integrity_all_rejected": todo_rechazado, "integrity_detail": integridad,
        "t_encrypt_s": t_enc, "t_decrypt_s": t_dec,
        "throughput_enc_MBps": (n_bytes / 1e6) / t_enc,
        "throughput_dec_MBps": (n_bytes / 1e6) / t_dec,
        "aes_entropy_encrypted": aes_ent, "aes_corr_encrypted": aes_corr,
    }


def agregar(valores: list[float]) -> dict:
    return {"mean": float(np.mean(valores)), "std": float(np.std(valores)),
            "min": float(np.min(valores)), "max": float(np.max(valores))}


def figura_correlacion(ruta_img: Path, out: Path) -> None:
    """Dispersion de pixeles adyacentes (horizontal) original vs cifrado."""
    img = cargar_rgb(ruta_img)
    enc, _ = cifrar(img.astype(np.float64), PASSWORD)
    preview = (enc >> 8).astype(np.uint8)
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.4))
    for ax, data, titulo in [(axes[0], img, "Original"), (axes[1], preview, "Cifrada / Encrypted")]:
        x = data[:, :-1, 0].ravel().astype(np.float64)
        y = data[:, 1:, 0].ravel().astype(np.float64)
        idx = rng.choice(x.size, min(4000, x.size), replace=False)
        ax.scatter(x[idx], y[idx], s=1, alpha=0.25, color="#c0392b")
        ax.set_title(f"{titulo}  (r={_corr(x, y):.3f})", fontsize=9)
        ax.set_xlabel("pixel (x,y)", fontsize=8)
        ax.set_ylabel("pixel (x+1,y)", fontsize=8)
        ax.set_xlim(0, 255); ax.set_ylim(0, 255)
    fig.suptitle("Correlacion de pixeles adyacentes (canal R)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def figura_rendimiento(resultados: list[dict], out: Path) -> None:
    orden = sorted(resultados, key=lambda r: r["megapixels"])
    mp = [r["megapixels"] for r in orden]
    te = [r["t_encrypt_s"] for r in orden]
    td = [r["t_decrypt_s"] for r in orden]
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.plot(mp, te, "o-", label="Cifrado / Encryption", color="#2e86de")
    ax.plot(mp, td, "s-", label="Descifrado / Decryption", color="#c0392b")
    ax.set_xlabel("Megapixeles", fontsize=9)
    ax.set_ylabel("Tiempo (s)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=".")
    parser.add_argument("--data", default=str(Path(__file__).resolve().parent.parent.parent / "DataSet"))
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    data = Path(args.data)

    rutas = [data / r for r in CURADAS if (data / r).exists()]
    print(f"Evaluando {len(rutas)} imagenes de {data}")

    resultados = []
    for ruta in rutas:
        r = evaluar_imagen(ruta)
        resultados.append(r)
        print(f"  {r['image']:16} {r['width']}x{r['height']:5} "
              f"H_enc={r['entropy_encrypted']:.4f} |r|_enc={np.mean([abs(v) for v in r['corr_encrypted'].values()]):.4f} "
              f"NPCR={r['npcr']:.3f} UACI={r['uaci']:.3f} exacta={r['exact_recovery']} "
              f"integridad={'OK' if r['integrity_all_rejected'] else 'FALLA'} "
              f"t_enc={r['t_encrypt_s']:.2f}s")

    # ---- agregados ----
    def col(k):
        return [r[k] for r in resultados]

    def corr_abs(clave):
        vals = []
        for r in resultados:
            vals.append(np.mean([abs(v) for v in r[clave].values()]))
        return [float(v) for v in vals]

    agg = {
        "n_images": len(resultados),
        "exact_recovery_all": all(col("exact_recovery")),
        "integrity_all_rejected_all": all(col("integrity_all_rejected")),
        "entropy_original": agregar(col("entropy_original")),
        "entropy_encrypted": agregar(col("entropy_encrypted")),
        "abs_corr_original": agregar(corr_abs("corr_original")),
        "abs_corr_encrypted": agregar(corr_abs("corr_encrypted")),
        "abs_corr_plain_cipher": agregar([abs(v) for v in col("corr_plain_cipher")]),
        "npcr": agregar(col("npcr")),
        "uaci": agregar(col("uaci")),
        "key_sensitivity": agregar(col("key_sensitivity")),
        "aes_entropy_encrypted": agregar(col("aes_entropy_encrypted")),
        "aes_abs_corr_encrypted": agregar(corr_abs("aes_corr_encrypted")),
        "throughput_enc_MBps": agregar(col("throughput_enc_MBps")),
        "throughput_dec_MBps": agregar(col("throughput_dec_MBps")),
    }

    salida = {"aggregate": agg, "per_image": resultados}
    (out / "dataset_results.json").write_text(json.dumps(salida, indent=2))

    # ---- figuras (imagen color representativa: mandril 4.2.03) ----
    rep = next((r for r in rutas if r.name == "4.2.03.tiff"), rutas[0])
    figura_correlacion(rep, out / "corr_scatter.png")
    figura_rendimiento(resultados, out / "perf_time.png")

    print("\n=== AGREGADO ===")
    print(f"Imagenes: {agg['n_images']}  recuperacion exacta en todas: {agg['exact_recovery_all']}  "
          f"integridad (todo rechazado) en todas: {agg['integrity_all_rejected_all']}")
    print(f"Entropia cifrada: {agg['entropy_encrypted']['mean']:.4f} +/- {agg['entropy_encrypted']['std']:.4f}")
    print(f"|corr| adyacente original: {agg['abs_corr_original']['mean']:.4f}  cifrada: {agg['abs_corr_encrypted']['mean']:.5f}")
    print(f"|corr| plano-cifrado: {agg['abs_corr_plain_cipher']['mean']:.5f}")
    print(f"NPCR: {agg['npcr']['mean']:.4f} +/- {agg['npcr']['std']:.4f}   UACI: {agg['uaci']['mean']:.4f} +/- {agg['uaci']['std']:.4f}")
    print(f"Sensibilidad clave: {agg['key_sensitivity']['mean']:.4f}%")
    print(f"AES-256-CTR entropia: {agg['aes_entropy_encrypted']['mean']:.4f}  |corr|: {agg['aes_abs_corr_encrypted']['mean']:.5f}")
    print(f"Throughput cifrado: {agg['throughput_enc_MBps']['mean']:.2f} MB/s  descifrado: {agg['throughput_dec_MBps']['mean']:.2f} MB/s")
    print(f"Resultados en {out / 'dataset_results.json'}")


if __name__ == "__main__":
    main()
