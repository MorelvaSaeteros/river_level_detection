´´´
Este proyecto contiene los siguientes scripts para la detección de caracteres en la escala graduada:
´´´

# filter.py
Este script filtra las imágenes estáticas con el fin de reconocer de manera automática los dígitos de una escala métrica graduada. Detecta zonas de interés, retornando un ROI filtrado y listo para el procesamiento; con ello se consigue aproximar bordes para la detección de los caracteres numéricos, comparando frames binarizados para identificar dichos caracteres con la BBDD MNIST y con la herramienta de reconocimiento óptico OCR de python - Pytesseract, concluyendo su propósito de identificación de dos maneras.

# train.py & model.py
Se tiene un script para el entrenamiento del modelo CNN para el reconocimiento de caracteres.

# test_model.py 
Este script se utilizó a modo de test  los caracteres detectados tanto con MNIST como Pytesseract, aunque todo está mejor organizado en filter.py 


´´´
Los siguientes scripts contienen la técnica de detección de bordes, que se pueda transferir a un extractor de características YOLOv3 con el propósito de detectar el borde de un río.
´´´


# sample.py
El script retorna un video completo en archivo .mp4 del cúmulo de videos a ser tratados (toma únicamente muestras específicas de información), es decir, tomas de interés estáticas con su respectivo .csv

#  split_train_val.py
Divide la base de datos generada tanto para entrenamiento como para validación. (train.txt y valid.txt). Después con los archivos (.yaml .jpg valid.txt y train.txt) dar paso al entremaniento de la red YOLOv3.

Para el entrenamiento de la red se emplean frames capturados que se guardan con un ID único para la etapa de etiquetado empleando la herramienta LabelImg ya que en la detección de objetos es necesario identificar las características más relevantes en el borde del río, continuando el proceso en COLAB.

# detection_video.py 
Este script contiene la ejecución del modelo entrenado, lee todos los frames, realiza la conversión de tensores, predice el modelo retornando en su detección (coordenadas, confianza, conf_clase, clase), dibuja las cajas delimitadas de la predicción, calcula la línea de tendencia, se calcula también el valor de píxeles en el eje ‘y’ de las “boyas virtuales” para obtener una media del nivel en píxeles. 
También realiza la ejecución del modelo de regresión logística tomando el modelo entrenado de tal manera que se obtiene la perdición del nivel del río.

# level_computing.py 
Permite entrenar el un modelo de regresión logística, con las características únicas como etiquetas ‘normalizando’ los datos, asignando un valor de épocas para la iteración del modelo, calcular los pesos y comparación con el escore y el umbral establecido para decidir si se salva o no el modelo. 

1. Clonar repo
git clone https://github.com/puigalex/deteccion-objetos-video.git

2. Borrar carpeta .git

3. Agregar los archivos y las carpetas al repositorio
- sample.py
- level_computing.py
- deteccion_video.py
- analyze_video.py
- database 

4. Descargar los videos del río de 1 día. Ejemplo araxes06-04-2022
Descomprimir videos en la carpeta /database/video

5. Copiar el archivo yolov3-custom.cfg en la carpeta /config

6. Copiar archivo yolov3_ckpt_99.pth en la carpeta /checkpoints y si no hay crear la carpeta

7. Editar el archivo en /data/custom/classes.names y colocar solo la clase a detectar
*borde


*** Pasos para ejecutar ***

1. Ejecutar el archivo 
python3 sample.py
Este genera el video_out y el csv

2. Rellenar el csv con los datos de nivel de las boyas y el nivel real para el modelo de regresión. En deteccion_video.py comentar las líneas
- 91		92
- 172
- 181
- 248

Luego ejecutar 
python3 deteccion_video.py

Al finalizar debe haber rellenado el csv con los datos de las boyas y el nivel real

Además genera el video_out con el resultado de la regla de 3 del nivel.

3. Generar el modelo de regresión. Ejecutar
python3 level_computing.python3

Debe generar un archivo {nombre río}.pkl en el directorio

4. Descomentar las líneas del punto 2 y ejecutar nuevamente el
python3 deteccion_video

Al finalizar genera el video_out con la regla de tres del nivel y el nivel aproximado con el modelo de regresión.
