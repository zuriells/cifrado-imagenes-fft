# Cifrado de Imagenes por Convolucion en el Dominio de Frecuencia

Herramienta de linea de comandos para cifrar y descifrar imagenes usando
la propiedad de convolucion de la Transformada de Fourier (FFT). En vez de
requerir un archivo de imagen-clave por separado, la clave se deriva
directamente de una **contrasena** (PBKDF2-HMAC-SHA256 + SHAKE-256), asi
que solo se necesitan la imagen cifrada y la contrasena para descifrar.

## Estructura del repositorio

```
fft_image_cipher.py     Herramienta principal (cifrar/descifrar)
ejemplos/                Imagen de prueba para probar la herramienta
analisis/                Script y resultados de la evaluacion de calidad y robustez
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

## Como funciona

1. La imagen y una "imagen-clave" (generada a partir de la contrasena) se
   transforman al dominio de la frecuencia con la FFT.
2. Se multiplican elemento a elemento y se aplica la IFFT: el resultado es
   la imagen cifrada.
3. Para descifrar, se repite el proceso pero dividiendo en vez de
   multiplicar, usando una clave regenerada con la misma contrasena.
4. La imagen cifrada se guarda como un PNG de 16 bits por canal (no 8),
   con el salt y los rangos de normalizacion embebidos como metadato, para
   que el descifrado sea exacto (sin perdida de informacion).

Los detalles matematicos completos, la evaluacion de calidad (histogramas,
entropia, PSNR) y las pruebas de robustez (ruido gaussiano, reescalado,
contrasena incorrecta) estan documentados en los articulos dentro de
`articulos/`.

## Analisis y evaluacion

La carpeta `analisis/` contiene el script (`analysis.py`) usado para
generar las figuras y metricas de los articulos: comparacion de
histogramas, entropia de Shannon, PSNR con contrasena correcta, y los
ataques de robustez (ruido gaussiano, reescalado, contrasena incorrecta).
Los resultados numericos quedan en `eval_results.json`.

## Autores

- M.C. Zuriel López Sosa
- Dra. Barbara Emma Sanchez Rinza
- Dr. Carlos Ignacio Robledo Sanchez

Benemerita Universidad Autonoma de Puebla (BUAP)
