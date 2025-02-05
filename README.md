# Laboratorio de procesamiento digital de señales #1
                                                                       Silvana Pardo Cepeda

 # Introducción
En este laboratorio buscamos analizar señales fisiologicas importadas desde difernetes plataformas como physioNet, con ciertas rutas que permiten la utilizacion de un lenguaje de programacion como python, que utilizamos en este caso para lograr una visualizacion y un analisis de esta señal de electrocardiagrama de un caso de apnea, un transtorno de sueño el cual tiene afectaciones cardiovasculares conocidas, logrando obtener datos estadisticos acerca de la señal y los diferentes tipos de ruidos con los que se puede contaminar, sacando de este la relacion señal-ruido(SNR).   

# Descarga e importacion de la señal 
1. Se utilizo la base de datos gratuita physioNet que permitio la descarga de una señal de libre eleccion, 2 archivos fundamentales, el .hea y el .dat estos son indispensables para que 
   se logre la lectura de la señal.  
2. Creamos una carpeta en el escritorio del computador donde agrgamos estos 2 archivos y el archivo de python donde haremos la programacion necesaria.
3. Abrimos el archivo python y anterioremnte debimos haber descargado la libreria wfdb para logar la lectura de la señal desde el archivo, que se debe ver asi (ejemplo):

   
   ![image](https://github.com/user-attachments/assets/9e17b321-2548-407e-bbf6-c9aa9db91cf6)


# Visualización de la señal
   Para lograr obtener la señal graficamente utilizamos la libreria  wfdb, numpy, matplotlib seaborn, os, y sys, el codigo debe verse algo asi:

   ![image](https://github.com/user-attachments/assets/53da0574-b427-4164-b118-2db7f681f8f1)

   

   Al correr el codigo vemos la grafica asi:

   
   ![image](https://github.com/user-attachments/assets/e9095841-7526-4bd2-a4ed-85b7d3efaa95)

   Lo que nos indica que debemos tomar solamente una seccion o rango de datos para no ver la grafica tan saturada, lo logramos asi:

   
  ![image](https://github.com/user-attachments/assets/42a4b99d-972f-4de8-b228-3cdceec048cd)

   
   ![image](https://github.com/user-attachments/assets/851e7d20-79e6-4696-8e64-53571dcae660)

![image](https://github.com/user-attachments/assets/b42cf349-4149-4098-a664-69075558ca11)



   Esto lo que hace es darle un valor a N y que en el tiempo seleccionado(10s) solo tome 1 dato cada 1000 datos, cada 10s, en el try, se carga la señal usando wfdb.rdrecord(archivo), extrayendo el primer canal y obteniendo la frecuencia de muestreo (fs). A partir de esto, se calcula cuántas muestras corresponden a 10 segundos (muestras_10s). Si el archivo no se encuentra, el except muestra un mensaje de error para evitar que el programa falle.


   
# Cálculo de estadísticos descriptivos
En esta parte lo que hicimos fue hallar a traves de calculos matematicos diferentes parametros como son: 

1.  Media de la señal: Representa el valor promedio de la señal.
2. Desviación estándar:  Indica la dispersión de la señal con respecto a la media.
3. Coeficiente de variación: Relación entre la desviación estándar y la media.

 Donde se nos pidio hallarlos de manera larga(matematica literal) y con funciones programadas, decidimos tomar nada mas los datos de los primeros 10s para tener un rango de datos mas pequeño para no hacer los calculos tan extensos, se ve algo asi:  
![image](https://github.com/user-attachments/assets/41587c32-34d7-4fd3-a4c9-c4df8a395008)


![image](https://github.com/user-attachments/assets/cfd3cee9-775d-4029-88c7-fea7807b6e85)


![image](https://github.com/user-attachments/assets/41ee6794-ad01-4f92-85b4-8c48da930186)


El código grafica la señal reducida utilizando matplotlib, donde el eje **x** representa las muestras y el **y** la amplitud del ECG. Luego, se extraen los primeros 10 segundos de la señal (3600 muestras a 360 Hz). Para calcular la media, la desviación estándar y el coeficiente de variación, se usan funciones de numpy y se imprimen los resultados en la consola con print.


4. Histogramas
5. Función de probabilidad 
Se puede  visualizar en estos items la distribución de valores de la señal, se generó un histograma y una función de densidad de probabilidad.

   ![image](https://github.com/user-attachments/assets/0f4e8d4e-8c3e-4179-a6dc-cc4d84e6515a)


![image](https://github.com/user-attachments/assets/db0dc56b-5390-4c2b-a109-5e6026a809f9)

En esta parte del código calculamos yse muestra la función de probabilidad del ECG con un histograma usando seaborn. 
Primero, se guardan los datos en una lista y, con os, se organizan y leen los archivos.
Luego, con numpy, se convierten en un formato más fácil de maneja, y ya pasamos a graficar la señal completa sin reducción con matplotlib.

# Ruido y Relación Señal-Ruido (SNR)
    La relación señal-ruido (SNR) es una métrica que indica la calidad de una señal. Se definió como la razón entre la potencia de la señal y la potencia del ruido:

  ![image](https://github.com/user-attachments/assets/e79c761b-6fcf-4d05-a59a-b23f15f35661)

   
    Se agregaron tres tipos de ruido en la señal y se ve asi:

Ruido gaussiano (ruido blanco):  es un tipo de interferencia que se genera con valores aleatorios y se suma a la señal para ver cómo la afecta.  



Ruido impulso (picos aleatorios):  se crea con una onda que cambia lentamente, imitando interferencias, y se suma a la señal original.  


Ruido tipo artefacto (ruido de movimiento): añade picos aleatorios en algunas partes del ECG, simulando errores o interrupciones en la señal.


![image](https://github.com/user-attachments/assets/862373df-5626-426a-9a78-77a56d2cb0b8)


Posteriaorme al tener la señal con los diferentes ruidos, iniciamos a calcular el SNR para cada uno:

![image](https://github.com/user-attachments/assets/4a12272c-c59c-49d1-ac74-b6d81ec54bd3)

![image](https://github.com/user-attachments/assets/6f11f6bd-3b4e-468b-8d29-de72125b657f)

En esta parte, calculamos el SNR(relación señal-ruido), que mide qué tan fuerte es la señal en comparación con el ruido.  

Se crea una función llamada calcular_snr, que recibe la señal original y la señal con ruido. 
Primero, se obtiene la potencia de la señal elevando su media al cuadrado.
Luego, se calcula el ruido restando la señal original de la señal con ruido. 
Finalmente, se usa la fórmula del SNR para obtener su valor y guardarlo en una variable.


Grficamente obtuvimos:

![image](https://github.com/user-attachments/assets/77cac09d-e0ed-4b60-924f-a8ed55993779)



Y para obtener estos items graficados, seguimos utilizando la misma etsructura:

![image](https://github.com/user-attachments/assets/caa2e4c6-16df-45c1-b0c1-26803b29f1f5)



# Conclusiones
Se logro el proceso de descarga e importacion de la señal ECG, permitiendo entender la estructura de los archivos y obteniendo información relevante, como la frecuencia de muestreo y las derivaciones del ECG, aplicamos diferentes formas de obtener una señal reducida respecto a su original,calculando estadísticas  (media, desviación estándar y coeficiente de variación) permitió analizar sus características y variabilidad
Se introdujeron distintos tipos de ruido (gaussiano, de artefacto e impulso) a la señal, para ver el proceso de una señal contaminada ccon su respectivo SNR, El uso de librerías como NumPy, Matplotlib, Seaborn y WFDB facilitó la utilización, análisis y visualización de los datos, reflejando importancia de Python en el procesamiento de señales biomédicas.
# Referencias
Moody, G. B. (2022). PhysioNet. In Encyclopedia of Computational Neuroscience (pp. 2806-2808). New York, NY: Springer New York.
Tandra, R., & Sahai, A. (2008). SNR walls for signal detection. IEEE Journal of selected topics in Signal Processing, 2(1), 4-17.
# Codigo en python 
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Cargar el archivo y la señal
archivo = "a01" 
N = 1000  
duracion_s = 10  

try:
    # Cargar la señal
    record = wfdb.rdrecord(archivo)
    senal = record.p_signal[:, 0]  # Extrae el primer canal de ECG
    fs = record.fs  # Frecuencia de muestreo
    muestras_10s = int(fs * duracion_s)  # Cantidad de muestras en 10 segundos
except FileNotFoundError:
    print(f"Error: El archivo '{archivo}' no se encontró.")
    sys.exit()

# Reducir la señal para graficar
senal_reducida = senal[::N]
tiempo_reducido = np.arange(0, len(senal), N)

# Graficar señal ECG reducida
plt.figure(figsize=(10, 5))
plt.plot(tiempo_reducido, senal_reducida, color='b')
plt.title("ECG Reducido")
plt.xlabel("Muestras")
plt.ylabel("Amplitud (mV)")
plt.grid()
plt.show()

# Extraer los primeros 10 segundos de la señal
senal_10s = senal[:muestras_10s]


# Graficar Función de Probabilidad
plt.figure(figsize=(10, 5))
sns.histplot(senal_10s, bins=50, kde=True, color="blue", stat="density", alpha=0.6)
plt.title("Función de probabilidad de la señal ECG")
plt.xlabel("Valor de la Señal")
plt.ylabel("Densidad de Probabilidad")
plt.grid()
plt.show()

# Cargar señales de un directorio específico
ruta =r"C:\Users\silva\Documents\señales"
ECGs = []
for ecgfilename in sorted(os.listdir(ruta)):
    if ecgfilename.endswith(".dat"):
        base_name = ecgfilename.split(".")[0]
        ecg = wfdb.rdsamp(os.path.join(ruta, base_name))
        ECGs.append(ecg[0])  # Guardar las señales en una lista
ecg = np.array(ECGs)

# Graficar una de las señales cargadas
plt.plot(ECGs[0])
plt.title("ECG saturado de datos")
plt.xlabel("Muestras (s)")
plt.ylabel("Amplitud (mV)")
plt.show()

# Cálculo de estadísticos descriptivos para la señal cargada
sen = ECGs[0][:, 0]

# Estadísticos descriptivos
media_ecg = np.mean(sen)
desviacion_ecg = np.std(sen, ddof=1)
coef_variacion_ecg = desviacion_ecg / media_ecg

# Imprimir Resultados de la señal cargada
print("\nEstadísticos(10 segundos):")
print(f"Media: {media_ecg:.4f}")
print(f"Desviación Estándar: {desviacion_ecg:.4f}")
print(f"Coeficiente de Variación: {coef_variacion_ecg:.4f}\n")
#Calculos sin funciones
suma = 0
for valor in senal:
    suma += valor
media = suma / len(senal)

suma_cuadrados = 0
for valor in senal:
    suma_cuadrados += (valor - media) ** 2
desviacion = (suma_cuadrados / (len(senal) - 1)) ** 0.5

coef_variacion = desviacion / media

print("Cálculo Manual:")
print(f"Media: {media:.4f}")
print(f"Desviación Estándar: {desviacion:.4f}")
print(f"Coeficiente de Variación: {coef_variacion:.4f}\n")
# Graficar Histograma de la señal cargada
plt.hist(sen, bins=50, color="blue", alpha=0.6)
plt.axvline(media_ecg, color="red", linestyle="dashed", linewidth=2, label="Media")
plt.title("Histograma de la Señal ECG Cargada")
plt.xlabel("Valor de la Señal")
plt.ylabel("Frecuencia")
plt.legend()
plt.show()


# Agregar ruido gaussiano
ruido_gaussiano = np.random.normal(0, 0.1, senal_10s.shape)
senal_gaussiana = senal_10s + ruido_gaussiano

# Agregar ruido artefacto (señales de baja frecuencia como base-line wander)
frecuencia_artefacto = 0.5  # Hz
artefacto = 0.5 * np.sin(2 * np.pi * frecuencia_artefacto * np.arange(len(senal_10s)) / fs)
senal_artefacto = senal_10s + artefacto

# Agregar ruido de impulso (picos aleatorios)
senal_impulso = np.copy(senal_10s)
n_picos = int(0.01 * len(senal_10s))  # 1% de los puntos con impulsos
indices_picos = np.random.randint(0, len(senal_10s), n_picos)
senal_impulso[indices_picos] += np.random.choice([-1, 1], n_picos) * np.random.uniform(0.5, 1.5, n_picos)

# Calcular SNR para cada señal
def calcular_snr(senal_original, senal_con_ruido):
    potencia_senal = np.mean(senal_original ** 2)
    ruido = senal_con_ruido - senal_original
    potencia_ruido = np.mean(ruido ** 2)
    snr = 10 * np.log10(potencia_senal / potencia_ruido)
    return snr

snr_gaussiano = calcular_snr(senal_10s, senal_gaussiana)
snr_artefacto = calcular_snr(senal_10s, senal_artefacto)
snr_impulso = calcular_snr(senal_10s, senal_impulso)


# Graficar señales
plt.figure(figsize=(12, 8))
plt.subplot(4, 1, 1)
plt.plot(senal_10s, color='b')
plt.title("Señal ECG Original")
plt.xlabel("Muestras")
plt.ylabel("Amplitud (mV)")
plt.grid()
print(f"SNR con Ruido Gaussiano: {snr_gaussiano:.2f} dB\n")

plt.subplot(4, 1, 2)
plt.plot(senal_gaussiana, color='r')
plt.title("ECG con Ruido Gaussiano")
plt.xlabel("Muestras")
plt.ylabel("Amplitud (mV)")
plt.grid()

plt.subplot(4, 1, 3)
plt.plot(senal_artefacto, color='g')
plt.title("ECG con Ruido de Artefacto")
plt.xlabel("Muestras")
plt.ylabel("Amplitud (mV)")
plt.grid()
print(f"SNR con Ruido de Artefacto: {snr_artefacto:.2f} dB\n")

plt.subplot(4, 1, 4)
plt.plot(senal_impulso, color='m')
plt.title("ECG con Ruido de Impulso")
plt.xlabel("Muestras")
plt.ylabel("Amplitud (mV)")
plt.grid()
print(f"SNR con Ruido de Impulso: {snr_impulso:.2f} dB\n")

plt.tight_layout()
plt.show()

