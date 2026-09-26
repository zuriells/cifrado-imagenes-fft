#!/usr/bin/env python3
"""Cifrado de imagenes en el dominio de la frecuencia (FFT) con cifrado
autenticado.

El nucleo del metodo sigue siendo la propiedad de convolucion de la
Transformada de Fourier: la imagen y una clave derivada de la contrasena se
multiplican en el dominio de la frecuencia, lo que equivale a
convolucionarlas en el dominio espacial. Esa convolucion produce una
representacion *blanqueada* de la imagen (histograma gaussiano, sin parecido
visual al original), util para el analisis estadistico del metodo.

IMPORTANTE (seguridad): la convolucion en frecuencia es una operacion
*lineal* y, por si sola, NO es un cifrado seguro. Conocido un solo par
(imagen, imagen-blanqueada) un atacante puede despejar la clave dividiendo
en frecuencia, F(g) = F(f*g) / F(f), y descifrar cualquier otra imagen
protegida con la misma contrasena. Es la misma debilidad que rompe a los
esquemas opticos lineales de doble mascara de fase aleatoria frente a
ataques de texto plano conocido/elegido (Peng 2006; Frauel 2007).

Para dar al metodo robustez propia como cifrador de imagenes (y no depender
unicamente del sobre autenticado), la representacion blanqueada pasa ademas
por una etapa de CONFUSION-DIFUSION en la linea de la arquitectura de
Fridrich: varias rondas de permutacion de posiciones (confusion) mezcladas
con una difusion no lineal por suma modular y encadenamiento XOR (que
propaga cualquier cambio a todo el arreglo, efecto avalancha). Al combinar
suma y XOR con permutaciones dependientes de la clave, la transformacion
deja de ser lineal, de modo que el ataque de texto plano conocido
F(g)=F(f*g)/F(f) ya no aplica.

Finalmente la confidencialidad y la integridad NO descansan en ninguna de
las etapas anteriores, sino en un cifrado autenticado estandar:
ChaCha20-Poly1305 (RFC 8439). El flujo es:

    1. secreto maestro = PBKDF2-HMAC-SHA256(contrasena, salt, iteraciones)
       (32 bytes; la clave de ChaCha20).
    2. imagen-clave g = SHAKE-256(secreto || "blanqueo") expandida al tamano
       de la imagen.
    3. blanqueo reversible: f*g via FFT, normalizado a 16 bits (se guardan
       los rangos Min-Max por canal para invertirlo sin perdida).
    4. confusion-difusion: rondas de permutacion + difusion no lineal sobre
       los bytes blanqueados, con material derivado de (secreto, nonce).
    5. sellado = ChaCha20-Poly1305(secreto, nonce, resultado, AAD=metadatos).
       La salida (misma longitud que la entrada) se guarda como PNG de 16
       bits por canal; el nonce, el tag de autenticacion, el salt y los
       rangos van en los metadatos del propio PNG.

Descifrar re-deriva el secreto de la contrasena y VERIFICA el tag Poly1305
antes de invertir el blanqueo: si la contrasena es incorrecta o el archivo
fue alterado (ruido, reescalado, un solo bit cambiado), la verificacion
falla y el descifrado se rechaza, en lugar de producir una imagen
degradada. Asi el esquema es no maleable: cualquier manipulacion se detecta.

El PNG cifrado es de 16 bits por canal porque el texto plano sellado tiene
exactamente alto*ancho*3 valores de 16 bits. Pillow solo escribe/lee PNG
RGB de 8 bits, asi que se usa un lector/escritor de PNG minimo compatible
con el estandar.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
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
# que la misma contrasena nunca produzca el mismo secreto dos veces.
BYTES_SALT = 16

# Tamano del nonce de ChaCha20-Poly1305 (RFC 8439 fija 96 bits = 12 bytes).
BYTES_NONCE = 12

# Longitud del tag de autenticacion Poly1305 (128 bits = 16 bytes).
BYTES_TAG = 16

# Separador de dominio para que la clave de ChaCha20 y el flujo de blanqueo
# se deriven del mismo secreto sin reutilizarlo directamente.
DOMINIO_BLANQUEO = b"fft-image-cipher|blanqueo|v2"

# Numero de rondas de confusion-difusion. Con 3 rondas la difusion alcanza
# valores de NPCR/UACI practicamente ideales.
RONDAS_CD = 3

# Separador de dominio del material (keystream, permutacion, IV) de cada
# ronda de confusion-difusion.
DOMINIO_CD = b"fft-image-cipher|confusion-difusion|v2"


def cargar_imagen_rgb(ruta: Path) -> np.ndarray:
    """Abre una imagen normal (PNG/JPG/etc.) y la devuelve como arreglo
    RGB de punto flotante, listo para operar con FFT."""
    return np.asarray(Image.open(ruta).convert("RGB"), dtype=np.float64)


def derivar_secreto_maestro(contrasena: str, salt: bytes, iteraciones: int) -> bytes:
    """Convierte la contrasena (mas el salt) en un secreto maestro de 32
    bytes mediante PBKDF2-HMAC-SHA256. Ese secreto es la clave de
    ChaCha20-Poly1305 y, ademas, la semilla del flujo de blanqueo."""
    return hashlib.pbkdf2_hmac("sha256", contrasena.encode("utf-8"), salt, iteraciones, dklen=32)


def derivar_imagen_clave(secreto_maestro: bytes, forma: tuple[int, int, int]) -> np.ndarray:
    """Genera la "imagen-clave" g del tamano de la imagen a partir del
    secreto maestro. SHAKE-256 (funcion de salida extensible de SHA-3) se
    usa como generador pseudoaleatorio para expandir el secreto a
    exactamente alto*ancho*3 bytes. Se anade un separador de dominio para
    no reutilizar el secreto tal cual (que tambien es la clave de ChaCha20).
    """
    n_bytes = forma[0] * forma[1] * forma[2]
    flujo_clave = hashlib.shake_256(secreto_maestro + DOMINIO_BLANQUEO).digest(n_bytes)
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
    para representar el texto plano sellado. Por eso este archivo construye
    el PNG bloque por bloque: cabecera IHDR, un bloque de texto (tEXt)
    con los metadatos en JSON (salt, nonce, tag, iteraciones y rangos de
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
    con los metadatos (salt, nonce, tag, iteraciones, rangos) embebidos."""
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


def fft_blanquear(imagen: np.ndarray, clave: np.ndarray) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Transformacion de *blanqueo* reversible (no es el cifrado en si).

    Para cada canal (R, G, B) multiplica la FFT de la imagen por la FFT de
    la clave y aplica la IFFT. Por la propiedad de convolucion de Fourier
    esto equivale a convolucionar imagen y clave en el dominio espacial,
    produciendo una representacion decorrelacionada, de histograma
    gaussiano y sin parecido visual al original. Devuelve tambien, por
    canal, el (valor_min, valor_max) usado para normalizar a 16 bits, sin
    el cual el blanqueo no podria invertirse sin perdida.
    """
    blanqueada = np.empty(imagen.shape, dtype=np.uint16)
    rangos_canal: list[tuple[float, float]] = []
    for canal in range(3):
        fft_imagen = np.fft.fft2(imagen[:, :, canal])
        fft_clave = np.fft.fft2(clave[:, :, canal])
        espacial = np.real(np.fft.ifft2(fft_imagen * fft_clave))
        valor_min, valor_max = float(espacial.min()), float(espacial.max())
        blanqueada[:, :, canal] = normalizar_a_uint16(espacial, valor_min, valor_max)
        rangos_canal.append((valor_min, valor_max))
    return blanqueada, rangos_canal


def fft_desblanquear(
    blanqueada: np.ndarray, clave: np.ndarray, rangos_canal: list[tuple[float, float]]
) -> np.ndarray:
    """Inversa de fft_blanquear: revierte la normalizacion de 16 bits y
    divide (en vez de multiplicar) las FFT de la representacion blanqueada
    y de la clave para deshacer la convolucion. El resultado, tras la
    IFFT, es la imagen original en 8 bits."""
    recuperada = np.empty(blanqueada.shape, dtype=np.uint8)
    for canal in range(3):
        valor_min, valor_max = rangos_canal[canal]
        espacial = desnormalizar_desde_uint16(blanqueada[:, :, canal], valor_min, valor_max)
        fft_blanqueada = np.fft.fft2(espacial)
        fft_clave = np.fft.fft2(clave[:, :, canal])
        recuperado = np.real(np.fft.ifft2(fft_blanqueada / (fft_clave + EPSILON)))
        recuperada[:, :, canal] = np.clip(np.rint(recuperado), 0, 255).astype(np.uint8)
    return recuperada


def _material_ronda_cd(secreto: bytes, nonce: bytes, n_bytes: int, ronda: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Deriva, de forma reproducible, el material de una ronda de
    confusion-difusion: un keystream de n_bytes, una permutacion de las
    n_bytes posiciones y un byte de inicializacion (IV). Todo se obtiene de
    SHAKE-256 sembrado con (secreto, nonce, dominio, numero de ronda), de
    modo que cifrado y descifrado regeneran exactamente el mismo material."""
    blob = hashlib.shake_256(secreto + nonce + DOMINIO_CD + bytes([ronda])).digest(n_bytes + 33)
    keystream = np.frombuffer(blob[:n_bytes], dtype=np.uint8)
    semilla = int.from_bytes(blob[n_bytes : n_bytes + 32], "big")
    iv = blob[n_bytes + 32]
    permutacion = np.random.default_rng(semilla).permutation(n_bytes)
    return keystream, permutacion, iv


def _difundir(x: np.ndarray, keystream: np.ndarray, iv: int) -> np.ndarray:
    """Difusion no lineal hacia adelante: suma modular con el keystream y
    encadenamiento XOR acumulado (prefijo XOR). Asi cada byte de salida
    depende de todos los bytes anteriores: un solo cambio se propaga en
    cascada (efecto avalancha)."""
    t = (x + keystream).astype(np.uint8)  # suma mod 256
    t[0] ^= iv
    return np.bitwise_xor.accumulate(t)


def _desdifundir(c: np.ndarray, keystream: np.ndarray, iv: int) -> np.ndarray:
    """Inversa exacta de _difundir."""
    previo = np.empty_like(c)
    previo[0] = 0
    previo[1:] = c[:-1]
    t = np.bitwise_xor(c, previo)
    t[0] ^= iv
    return ((t.astype(np.int16) - keystream.astype(np.int16)) & 0xFF).astype(np.uint8)


def confusion_difusion(datos: bytes, secreto: bytes, nonce: bytes) -> bytes:
    """Aplica RONDAS_CD rondas de (difusion no lineal + permutacion) sobre
    los bytes de entrada. La mezcla de suma modular, XOR encadenado y
    permutaciones dependientes de la clave hace que la transformacion sea no
    lineal y con difusion completa, resistente a ataques de texto plano
    conocido/elegido a los que si sucumbe la convolucion lineal por si sola."""
    x = np.frombuffer(datos, dtype=np.uint8).copy()
    n = x.size
    for ronda in range(RONDAS_CD):
        keystream, permutacion, iv = _material_ronda_cd(secreto, nonce, n, ronda)
        x = _difundir(x, keystream, iv)
        x = x[permutacion]
    return x.tobytes()


def inversa_confusion_difusion(datos: bytes, secreto: bytes, nonce: bytes) -> bytes:
    """Inversa exacta de confusion_difusion: deshace las rondas en orden
    inverso, aplicando primero la permutacion inversa y luego la difusion
    inversa en cada una."""
    x = np.frombuffer(datos, dtype=np.uint8).copy()
    n = x.size
    for ronda in reversed(range(RONDAS_CD)):
        keystream, permutacion, iv = _material_ronda_cd(secreto, nonce, n, ronda)
        inversa = np.empty_like(permutacion)
        inversa[permutacion] = np.arange(n)
        x = x[inversa]
        x = _desdifundir(x, keystream, iv)
    return x.tobytes()


def _aad_desde_metadatos(metadatos: dict) -> bytes:
    """Construye los datos asociados autenticados (AAD) de forma
    deterministica a partir de los metadatos que viajan en claro. Al
    autenticarlos, cualquier manipulacion del salt, el nonce, los rangos o
    el numero de iteraciones tambien invalida el tag."""
    autenticado = {clave: metadatos[clave] for clave in ("salt", "nonce", "iteraciones", "rangos")}
    return json.dumps(autenticado, sort_keys=True, separators=(",", ":")).encode("utf-8")


def cifrar(imagen: np.ndarray, contrasena: str) -> tuple[np.ndarray, dict]:
    """Cifra la imagen y devuelve (pixeles cifrados de 16 bits, metadatos).

    Blanquea la imagen con la FFT, serializa la representacion blanqueada a
    bytes y la sella con ChaCha20-Poly1305. La salida sellada (misma
    longitud que el texto plano) se reinterpreta como una imagen de 16 bits
    por canal, apta para guardarse como PNG."""
    forma = imagen.shape
    salt = os.urandom(BYTES_SALT)
    nonce = os.urandom(BYTES_NONCE)
    secreto = derivar_secreto_maestro(contrasena, salt, ITERACIONES_PBKDF2)
    clave = derivar_imagen_clave(secreto, forma)

    blanqueada, rangos_canal = fft_blanquear(imagen, clave)
    # Confusion-difusion: rompe la linealidad del blanqueo y difunde
    # cualquier cambio a todo el arreglo antes del sellado autenticado.
    texto_plano = confusion_difusion(blanqueada.astype(">u2").tobytes(), secreto, nonce)

    metadatos = {
        "rangos": rangos_canal,
        "salt": base64.b64encode(salt).decode("ascii"),
        "nonce": base64.b64encode(nonce).decode("ascii"),
        "iteraciones": ITERACIONES_PBKDF2,
        "alto": int(forma[0]),
        "ancho": int(forma[1]),
    }
    sellado = ChaCha20Poly1305(secreto).encrypt(nonce, texto_plano, _aad_desde_metadatos(metadatos))
    cuerpo, tag = sellado[:-BYTES_TAG], sellado[-BYTES_TAG:]
    metadatos["tag"] = base64.b64encode(tag).decode("ascii")

    cifrado = np.frombuffer(cuerpo, dtype=">u2").reshape(forma).astype(np.uint16)
    return cifrado, metadatos


def descifrar(cifrado: np.ndarray, metadatos: dict, contrasena: str) -> np.ndarray:
    """Descifra la imagen. Re-deriva el secreto de la contrasena, VERIFICA
    el tag Poly1305 y solo entonces invierte el blanqueo. Si la contrasena
    es incorrecta o el archivo fue alterado, ChaCha20-Poly1305 lanza
    InvalidTag y el descifrado se rechaza (no se devuelve ninguna imagen)."""
    forma = (cifrado.shape[0], cifrado.shape[1], 3)
    salt = base64.b64decode(metadatos["salt"])
    nonce = base64.b64decode(metadatos["nonce"])
    tag = base64.b64decode(metadatos["tag"])
    secreto = derivar_secreto_maestro(contrasena, salt, metadatos["iteraciones"])

    cuerpo = cifrado.astype(">u2").tobytes()
    texto_plano = ChaCha20Poly1305(secreto).decrypt(
        nonce, cuerpo + tag, _aad_desde_metadatos(metadatos)
    )  # lanza InvalidTag si la contrasena es incorrecta o hubo manipulacion

    # Deshace la confusion-difusion y luego el blanqueo.
    blanqueada_bytes = inversa_confusion_difusion(texto_plano, secreto, nonce)
    blanqueada = np.frombuffer(blanqueada_bytes, dtype=">u2").reshape(forma).astype(np.uint16)
    clave = derivar_imagen_clave(secreto, forma)
    return fft_desblanquear(blanqueada, clave, metadatos["rangos"])


def salida_predeterminada(ruta_imagen: Path, sufijo: str) -> Path:
    """Genera un nombre de archivo de salida cuando el usuario no indica
    uno con -o, por ejemplo 'foto.png' -> 'foto_cifrada.png'."""
    return ruta_imagen.with_name(f"{ruta_imagen.stem}_{sufijo}.png")


def ejecutar_cifrado(ruta_imagen: Path, contrasena: str, ruta_salida: Path) -> None:
    """Flujo completo de cifrado: carga la imagen, la cifra con clave
    derivada de la contrasena y guarda el resultado como PNG de 16 bits
    con los metadatos (salt, nonce, tag, rangos) necesarios para descifrar
    y autenticar mas adelante."""
    imagen = cargar_imagen_rgb(ruta_imagen)
    cifrado, metadatos = cifrar(imagen, contrasena)
    guardar_png16(ruta_salida, cifrado, metadatos)
    print(f"Imagen cifrada: {ruta_salida}")


def ejecutar_descifrado(ruta_imagen: Path, contrasena: str, ruta_salida: Path) -> None:
    """Flujo completo de descifrado: lee la imagen cifrada y sus
    metadatos, verifica la autenticacion y, si es valida, guarda la imagen
    recuperada. Si la contrasena es incorrecta o el archivo fue alterado,
    aborta con un mensaje de error sin escribir nada."""
    cifrado, metadatos = cargar_png16(ruta_imagen)
    try:
        descifrado = descifrar(cifrado, metadatos, contrasena)
    except InvalidTag:
        print(
            "Error: autenticacion fallida. La contrasena es incorrecta o el "
            "archivo cifrado fue alterado; no se descifra nada.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    Image.fromarray(descifrado, mode="RGB").save(ruta_salida)
    print(f"Imagen descifrada: {ruta_salida}")


def construir_analizador() -> argparse.ArgumentParser:
    """Define la interfaz de linea de comandos: -c o -d (mutuamente
    excluyentes) para elegir cifrar o descifrar, -p para la contrasena y
    -o, opcional, para la ruta de salida."""
    analizador = argparse.ArgumentParser(
        description="Cifrado autenticado de imagenes con blanqueo por convolucion en el dominio de frecuencia."
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
