# Prácticas en Empresa en AEMet

Consisten en:

* Verificación de datos de radiación solar, comparando datos procedentes de distintos modélos con observaciones provenientes de estaciones.

* Realización de tests estadísticos para llegar a conclusiones sobre la precisión de los modelos.

* Aprendizaje y uso de Python, AERONET, Linux.


Los datos fueron extraidos de [AERONET][1]. Se han descargados los datos de AOD de Murcia para el año 2013 para el  Nivel AOT 1.5.

El tratamiento de datos se ha realizado con Python 2, usando iPython Notebook en un entorno Linux pero con las siguientes notas:

* En la rama (*branch*) ***practicas*** se ha usado iPython 3. Corresponde a la versión entregada en la asignatura.
* En la rama ***master*** se ha usado iPython 4, Python 2.7 y numpy 1.10. En esta versión he perfilado la versión de la rama anterior y añadido los estadisticos y gráficas por meses.

Para el filtrado de los datos descargados de AERONET, se ha usado el script desarollado por AEMet (aeronettools.py).

En ambas ramas se ha dejado un archivo pdf con el contenido del iPython Notebook de los tests estadísticos realizados usando las funciones que se encuentran en la carpeta funciones.


Visualización del notebook online [AQUI][2]



[1]: <http://aeronet.gsfc.nasa.gov/cgi-bin/type_piece_of_map_opera_v2_new>
[2]: <http://nbviewer.ipython.org/github/Ruben-VV/practica-python/blob/master/Aerosoles.ipynb>
