#!/usr/bin/env python3
"""Cifrado de imagenes en el dominio de la frecuencia (FFT), basado en la
propiedad de convolucion de la Transformada de Fourier: la imagen y una
clave se multiplican en el dominio de la frecuencia, y esa multiplicacion
equivale a convolucionarlas en el dominio espacial.

En vez de requerir un archivo de imagen-clave por separado, la clave se
deriva de una contrasena: PBKDF2-HMAC-SHA256 convierte la contrasena (mas
un salt aleatorio) en 32 bytes, y SHAKE-256 expande esos bytes en un flujo
pseudoaleatorio del tamano de la imagen, que hace el papel de la
imagen-clave. El salt y los rangos Min-Max por canal necesarios para
invertir la cuantizacion de 16 bits se embeben directamente en el PNG
cifrado, de modo que para descifrar solo se necesitan la imagen cifrada y
la contrasena.

El resultado de la convolucion en el dominio espacial abarca un rango
enorme (dominado por el termino de continua), por lo que un PNG de 8 bits
no tiene suficiente precision para descifrar correctamente. La salida
cifrada es un PNG de 16 bits por canal (una imagen real y autocontenida);
Pillow solo puede escribir/leer PNG RGB de 8 bits, asi que se usa en su
lugar un lector/escritor de PNG minimo compatible con el estandar.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import struct
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

# Estabilizador sumado al denominador en la division de frecuencia, para
# evitar dividir por (casi) cero cuando la FFT de la clave tiene valores
# muy pequenos en alguna frecuencia.
EPSILON = 1e-10

# Cabecera fija que identifica a cualquier archivo PNG valido (8 bytes).
FIRMA_PNG = b"\x89PNG\r\n\x1a\n"

# Numero de iteraciones de PBKDF2: cuantas mas, mas lento es probar cada
# contrasena por fuerza bruta, a costa de que cifrar/descifrar tambien
# tarde un poco mas. 310,000 sigue la recomendacion vigente para
# PBKDF2-HMAC-SHA256.
ITERACIONES_PBKDF2 = 310_000

# Tamano del salt aleatorio (en bytes) que se genera en cada cifrado, para
# que la misma contrasena nunca produzca la misma clave dos veces.
BYTES_SALT = 16


def cargar_imagen_rgb(ruta: Path) -> np.ndarray:
    """Abre una imagen normal (PNG/JPG/etc.) y la devuelve como arreglo
    RGB de punto flotante, listo para operar con FFT."""
    return np.asarray(Image.open(ruta).convert("RGB"), dtype=np.float64)


def derivar_imagen_clave(contrasena: str, salt: bytes, forma: tuple[int, int, int], iteraciones: int) -> np.ndarray:
    """Convierte una contrasena en una "imagen-clave" del mismo tamano que
    la imagen a cifrar/descifrar, sin necesidad de un archivo de clave.

    Primero PBKDF2-HMAC-SHA256 convierte la contrasena (mas el salt) en
    una clave corta de 32 bytes, resistente a ataques de fuerza bruta
    gracias a las iteraciones repetidas. Despues, como esos 32 bytes no
    alcanzan para llenar una imagen completa, SHAKE-256 (una funcion de
    salida extensible) se usa como generador pseudoaleatorio para
    expandirlos a exactamente alto*ancho*3 bytes: uno por cada valor de
    color de cada pixel.
    """
    derivado = hashlib.pbkdf2_hmac("sha256", contrasena.encode("utf-8"), salt, iteraciones, dklen=32)
    flujo_clave = hashlib.shake_256(derivado).digest(forma[0] * forma[1] * forma[2])
    return np.frombuffer(flujo_clave, dtype=np.uint8).reshape(forma).astype(np.float64)


def _bloque_png(tipo_bloque: bytes, datos: bytes) -> bytes:
    """Empaqueta un bloque (chunk) PNG: longitud + tipo + datos + CRC,
    tal como exige el estandar del formato."""
    return (
        struct.pack(">I", len(datos))
        + tipo_bloque
        + datos
        + struct.pack(">I", zlib.crc32(tipo_bloque + datos))
    )


def guardar_png16(ruta: Path, imagen: np.ndarray, metadatos: dict) -> None:
    """Escribe un PNG RGB de 16 bits por canal a mano.

    Pillow (la libreria de imagenes usada en el resto del codigo) solo
    sabe escribir PNG de 8 bits por canal, y aqui se necesitan 16 bits
    para no perder precision al descifrar. Por eso este archivo construye
    el PNG bloque por bloque: cabecera IHDR, un bloque de texto (tEXt)
    con los metadatos en JSON (salt, iteraciones y rangos de
    normalizacion), los pixeles comprimidos (IDAT) y el cierre (IEND).
    """
    alto, ancho, _ = imagen.shape

    # Cada fila de pixeles debe llevar un byte de "filtro" (aqui, 0 = sin
    # filtro) antes de sus valores, segun el estandar PNG.
    avance_fila = ancho * 3 * 2
    plano = imagen.astype(">u2").tobytes()
    crudo = bytearray()
    for fila in range(alto):
        crudo.append(0)
        crudo += plano[fila * avance_fila : (fila + 1) * avance_fila]

    # color type 2 = RGB, profundidad de bits 16.
    ihdr = struct.pack(">IIBBBBB", ancho, alto, 16, 2, 0, 0, 0)
    texto = b"meta\x00" + json.dumps(metadatos).encode("latin-1")
    idat = zlib.compress(bytes(crudo), 6)

    with open(ruta, "wb") as archivo:
        archivo.write(FIRMA_PNG)
        archivo.write(_bloque_png(b"IHDR", ihdr))
        archivo.write(_bloque_png(b"tEXt", texto))
        archivo.write(_bloque_png(b"IDAT", idat))
        archivo.write(_bloque_png(b"IEND", b""))


def cargar_png16(ruta: Path) -> tuple[np.ndarray, dict]:
    """Lee un PNG de 16 bits por canal escrito por guardar_png16, junto
    con los metadatos (salt, iteraciones, rangos) embebidos en el."""
    datos = ruta.read_bytes()
    if datos[:8] != FIRMA_PNG:
        raise ValueError(f"{ruta} no es un archivo PNG.")

    posicion = 8
    ancho = alto = None
    metadatos: dict = {}
    idat = bytearray()

    # Recorre los bloques del archivo uno por uno hasta el cierre (IEND),
    # guardando solo lo que se necesita: dimensiones, metadatos y los
    # datos de pixeles comprimidos.
    while posicion < len(datos):
        longitud = struct.unpack(">I", datos[posicion : posicion + 4])[0]
        tipo_bloque = datos[posicion + 4 : posicion + 8]
        datos_bloque = datos[posicion + 8 : posicion + 8 + longitud]
        posicion += 12 + longitud

        if tipo_bloque == b"IHDR":
            ancho, alto = struct.unpack(">II", datos_bloque[:8])
        elif tipo_bloque == b"tEXt":
            palabra_clave, _, texto = datos_bloque.partition(b"\x00")
            if palabra_clave == b"meta":
                metadatos = json.loads(texto.decode("latin-1"))
        elif tipo_bloque == b"IDAT":
            idat += datos_bloque
        elif tipo_bloque == b"IEND":
            break

    # Descomprime los pixeles y quita el byte de filtro de cada fila para
    # reconstruir el arreglo de 16 bits.
    crudo = zlib.decompress(bytes(idat))
    avance_fila = ancho * 3 * 2
    imagen = np.empty((alto, ancho, 3), dtype=">u2")
    desplazamiento = 0
    for fila in range(alto):
        desplazamiento += 1
        bytes_fila = crudo[desplazamiento : desplazamiento + avance_fila]
        imagen[fila] = np.frombuffer(bytes_fila, dtype=">u2").reshape(ancho, 3)
        desplazamiento += avance_fila
    return imagen.astype(np.uint16), metadatos


def normalizar_a_uint16(canal: np.ndarray, valor_min: float, valor_max: float) -> np.ndarray:
    """Reescala un canal de valores reales (con un rango enorme, tipico
    de una convolucion) al rango entero [0, 65535] que cabe en 16 bits."""
    if valor_max == valor_min:
        return np.zeros_like(canal, dtype=np.uint16)
    normalizado = (canal - valor_min) / (valor_max - valor_min) * 65535.0
    return np.clip(np.rint(normalizado), 0, 65535).astype(np.uint16)


def desnormalizar_desde_uint16(canal: np.ndarray, valor_min: float, valor_max: float) -> np.ndarray:
    """Operacion inversa a normalizar_a_uint16: recupera los valores
    reales originales a partir del canal de 16 bits y el rango guardado
    como metadato."""
    if valor_max == valor_min:
        return np.full(canal.shape, valor_min, dtype=np.float64)
    return canal.astype(np.float64) / 65535.0 * (valor_max - valor_min) + valor_min


def fft_cifrar(imagen: np.ndarray, clave: np.ndarray) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Cifra la imagen: para cada canal (R, G, B), multiplica su FFT por
    la FFT de la clave y aplica la IFFT. Por la propiedad de convolucion
    de Fourier, esto equivale a convolucionar la imagen con la clave en
    el dominio espacial, produciendo una imagen sin ningun parecido visual
    al original.

    Tambien devuelve, por canal, el (valor_min, valor_max) usado para
    normalizar a 16 bits, ya que sin ese rango exacto no se podria
    descifrar sin perdida.
    """
    cifrado = np.empty(imagen.shape, dtype=np.uint16)
    rangos_canal: list[tuple[float, float]] = []
    for canal in range(3):
        fft_imagen = np.fft.fft2(imagen[:, :, canal])
        fft_clave = np.fft.fft2(clave[:, :, canal])
        espacial = np.real(np.fft.ifft2(fft_imagen * fft_clave))
        valor_min, valor_max = float(espacial.min()), float(espacial.max())
        cifrado[:, :, canal] = normalizar_a_uint16(espacial, valor_min, valor_max)
        rangos_canal.append((valor_min, valor_max))
    return cifrado, rangos_canal


def fft_descifrar(
    cifrado: np.ndarray, clave: np.ndarray, rangos_canal: list[tuple[float, float]]
) -> np.ndarray:
    """Descifra la imagen: revierte la normalizacion de 16 bits, y luego
    divide (en vez de multiplicar) las FFT de la imagen cifrada y de la
    clave para deshacer la convolucion del cifrado. El resultado, tras la
    IFFT, es la imagen original."""
    descifrado = np.empty(cifrado.shape, dtype=np.uint8)
    for canal in range(3):
        valor_min, valor_max = rangos_canal[canal]
        espacial = desnormalizar_desde_uint16(cifrado[:, :, canal], valor_min, valor_max)
        fft_cifrado = np.fft.fft2(espacial)
        fft_clave = np.fft.fft2(clave[:, :, canal])
        recuperado = np.real(np.fft.ifft2(fft_cifrado / (fft_clave + EPSILON)))
        descifrado[:, :, canal] = np.clip(np.rint(recuperado), 0, 255).astype(np.uint8)
    return descifrado


def salida_predeterminada(ruta_imagen: Path, sufijo: str) -> Path:
    """Genera un nombre de archivo de salida cuando el usuario no indica
    uno con -o, por ejemplo 'foto.png' -> 'foto_cifrada.png'."""
    return ruta_imagen.with_name(f"{ruta_imagen.stem}_{sufijo}.png")


def ejecutar_cifrado(ruta_imagen: Path, contrasena: str, ruta_salida: Path) -> None:
    """Flujo completo de cifrado: carga la imagen, genera un salt nuevo,
    deriva la clave a partir de la contrasena, cifra y guarda el
    resultado como PNG de 16 bits con los metadatos necesarios para
    descifrar mas adelante."""
    imagen = cargar_imagen_rgb(ruta_imagen)
    salt = os.urandom(BYTES_SALT)
    clave = derivar_imagen_clave(contrasena, salt, imagen.shape, ITERACIONES_PBKDF2)
    cifrado, rangos_canal = fft_cifrar(imagen, clave)

    metadatos = {
        "rangos": rangos_canal,
        "salt": base64.b64encode(salt).decode("ascii"),
        "iteraciones": ITERACIONES_PBKDF2,
    }
    guardar_png16(ruta_salida, cifrado, metadatos)
    print(f"Imagen cifrada: {ruta_salida}")


def ejecutar_descifrado(ruta_imagen: Path, contrasena: str, ruta_salida: Path) -> None:
    """Flujo completo de descifrado: lee la imagen cifrada y sus
    metadatos, regenera la misma clave con la contrasena dada (si es
    incorrecta, el resultado sera ruido) y guarda la imagen recuperada."""
    cifrado, metadatos = cargar_png16(ruta_imagen)
    salt = base64.b64decode(metadatos["salt"])
    clave = derivar_imagen_clave(contrasena, salt, cifrado.shape, metadatos["iteraciones"])
    descifrado = fft_descifrar(cifrado, clave, metadatos["rangos"])

    Image.fromarray(descifrado, mode="RGB").save(ruta_salida)
    print(f"Imagen descifrada: {ruta_salida}")


def construir_analizador() -> argparse.ArgumentParser:
    """Define la interfaz de linea de comandos: -c o -d (mutuamente
    excluyentes) para elegir cifrar o descifrar, -p para la contrasena y
    -o, opcional, para la ruta de salida."""
    analizador = argparse.ArgumentParser(
        description="Cifrado de imagenes por convolucion en el dominio de frecuencia, con clave derivada de contrasena."
    )
    accion = analizador.add_mutually_exclusive_group(required=True)
    accion.add_argument("-c", metavar="IMAGEN", help="Cifra la imagen indicada.")
    accion.add_argument("-d", metavar="IMAGEN", help="Descifra la imagen (PNG de 16 bits) indicada.")
    analizador.add_argument("-p", metavar="CONTRASENA", required=True, help="Contrasena para cifrar/descifrar.")
    analizador.add_argument("-o", metavar="SALIDA", help="Ruta de salida (opcional).")
    return analizador


def principal() -> None:
    """Punto de entrada: lee los argumentos y llama al flujo de cifrado o
    de descifrado segun se haya usado -c o -d."""
    argumentos = construir_analizador().parse_args()

    if argumentos.c:
        ruta_imagen = Path(argumentos.c)
        ruta_salida = Path(argumentos.o) if argumentos.o else salida_predeterminada(ruta_imagen, "cifrada")
        ejecutar_cifrado(ruta_imagen, argumentos.p, ruta_salida)
    else:
        ruta_imagen = Path(argumentos.d)
        ruta_salida = Path(argumentos.o) if argumentos.o else salida_predeterminada(ruta_imagen, "descifrada")
        ejecutar_descifrado(ruta_imagen, argumentos.p, ruta_salida)


if __name__ == "__main__":
    principal()
