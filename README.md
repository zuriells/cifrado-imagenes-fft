# Cifrado Autenticado de Imagenes con Blanqueo por Convolucion (FFT)

Herramienta de linea de comandos para cifrar y descifrar imagenes. El nucleo
combina dos capas:

1. **Blanqueo por convolucion (FFT):** la imagen y una "imagen-clave"
   derivada de la contrasena se multiplican en el dominio de la frecuencia
   (propiedad de convolucion de la Transformada de Fourier), produciendo una
   representacion decorrelacionada. Esta transformacion es *lineal* y **no**
   aporta seguridad por si sola.
2. **Cifrado autenticado (ChaCha20-Poly1305, RFC 8439):** la confidencialidad
   y la **integridad** las aporta un cifrado autenticado estandar. Si la
   contrasena es incorrecta o el archivo fue alterado (aunque sea un solo
   bit), la verificacion falla y el descifrado se rechaza.

La clave se deriva de una **contrasena** (PBKDF2-HMAC-SHA256 + SHAKE-256), asi
que solo se necesitan la imagen cifrada y la contrasena para descifrar; no hay
archivo de clave externo.

## Estructura del repositorio

```
fft_image_cipher.py     Herramienta principal (cifrar/descifrar)
ejemplos/                Imagen de prueba para probar la herramienta
analisis/                Script y resultados de la evaluacion (calidad e integridad)
articulos/                Articulos academicos (espanol e ingles, formato Springer LNCS)
```

## Requisitos

```bash
pip install -r requirements.txt
```

## Uso

Cifrar una imagen:

```bash
python3 fft_image_cipher.py -c ejemplos/prueba2.png -p "tu_contrasena"
```

Descifrar:

```bash
python3 fft_image_cipher.py -d ejemplos/prueba2_cifrada.png -p "tu_contrasena"
```

Parametros:

- `-c IMAGEN` — cifra la imagen indicada.
- `-d IMAGEN` — descifra la imagen (PNG de 16 bits) indicada.
- `-p CONTRASENA` — contrasena para cifrar/descifrar (obligatorio).
- `-o SALIDA` — ruta de salida (opcional; si se omite, se genera
  automaticamente como `<nombre>_cifrada.png` o `<nombre>_descifrada.png`).

Si el descifrado falla la autenticacion (contrasena incorrecta o archivo
manipulado), la herramienta aborta con un error y no escribe ninguna imagen.

## Como funciona

1. Se deriva un secreto maestro de la contrasena (PBKDF2-HMAC-SHA256) y, de
   el, una imagen-clave del tamano de la imagen (SHAKE-256).
2. La imagen y la imagen-clave se transforman al dominio de la frecuencia con
   la FFT, se multiplican elemento a elemento y se aplica la IFFT: esa es la
   representacion *blanqueada* (reversible, pero no segura por si misma).
3. La representacion blanqueada se sella con **ChaCha20-Poly1305**: la salida
   se guarda como un PNG de 16 bits por canal, con el salt, el nonce, la
   etiqueta de autenticacion y los rangos de normalizacion embebidos como
   metadato.
4. Para descifrar, se re-deriva el secreto, se **verifica la etiqueta de
   autenticacion** y solo entonces se invierte el blanqueo (dividiendo en vez
   de multiplicar en el dominio de la frecuencia), recuperando la imagen
   original de forma exacta (sin perdida).

> **Nota de seguridad.** La convolucion en frecuencia, por si sola, es un
> cifrado lineal inseguro (vulnerable a ataques de texto plano conocido: dado
> un par imagen/cifrado se despeja la clave). Por eso aqui se usa solo como
> transformacion reversible y toda la seguridad recae en ChaCha20-Poly1305.

Los detalles matematicos completos, la evaluacion de calidad (histogramas,
entropia, PSNR) y las pruebas de integridad estan documentados en los
articulos dentro de `articulos/`.

## Analisis y evaluacion

La carpeta `analisis/` contiene dos scripts:

- `analysis.py` — evaluacion sobre una imagen: histogramas, entropia de
  Shannon, PSNR con contrasena correcta, NPCR/UACI/sensibilidad de clave del
  nucleo, y pruebas de integridad (contrasena incorrecta, ruido gaussiano,
  reescalado y volteo de un bit), que deben resultar todas en un rechazo por
  fallo de autenticacion. Resultados en `eval_results.json`.
- `analysis_dataset.py` — evaluacion multi-imagen sobre un subconjunto diverso
  de la USC-SIPI Image Database (18 imagenes, 256-1024 px, color y gris):
  resultados agregados (media +/- desv.) de entropia, correlacion de pixeles
  adyacentes (H/V/D) y plano-cifrado, NPCR/UACI, sensibilidad de clave,
  comparacion con AES-256-CTR, e integridad. Genera las figuras de correlacion
  y rendimiento y `dataset_results.json`. Requiere colocar el dataset en
  `DataSet/` (aerials, misc, sequences, textures), descargado de
  https://sipi.usc.edu/database

Estos scripts responden a las revisiones (evaluacion sobre multiples imagenes,
metricas criptograficas adicionales, comparacion con un cifrador estandar y
evaluacion de rendimiento).

## Autores

- M.C. Zuriel López Sosa
- Dra. Barbara Emma Sanchez Rinza
- Dr. Carlos Ignacio Robledo Sanchez

Benemerita Universidad Autonoma de Puebla (BUAP)
