# Vehicle Counting
Codigo que permite realizar el conteo y clasificacion de vehiculos con Yolov4. 


## Descripcion
Este repositorio utiliza conteo por linea virtual y conteo por regiones. El cual, perimite identificar cuantos y que tipo de vehiculos de desplazan de una region x a una region y.


<p align="center">
  <img width="400" height="400" src="gfx/counting_regions.PNG">
</p>

![](gfx/pipeline.PNG)


El modelo fue entrenado usando [Google colab](https://colab.research.google.com/). Fue usando un tamano del lote de 64 con una taza de aprendizaje de 0.003.
