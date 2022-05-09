# **Proyecto**: Análisis de mercado inmobiliario

El objetivo de este proyecto es reproducir los pasos que haría un/a Data Scientist cuando se enfrenta a una problemática real. Por eso, consta de tres secciones:

- En la **Parte 1**, te presentamos la problemática sobre la cual vas a trabajar. En esta sección deberás decidir qué datos te ayudarán a trabajar en este problema y dónde puedes conseguirlos.
- En la **Parte 2** te proveemos de un dataset para abordar la problemática planteada. Deberás realizar un Análisis Exploratorio de Datos sobre este dataset.
- En la **Parte 3**, deberás utilizar herramientas de Machine Learning para predecir la variable de interés.

En este proyecto vas a trabajar con un dataset de propiedades en venta publicado en el portal [Properati](http://localhost:8888/notebooks/Proyectos/Proyecto I/www.properati.com.ar).

## **Parte 1**
### Problema

Recientemente te has incorporado al equipo de Datos de una gran inmobiliaria. La primera tarea que se te asigna es ayudar a los tasadores/as a valuar las propiedades, ya que es un proceso difícil y, a veces, subjetivo. Para ello, propones crear un modelo de Machine Learning que, dadas ciertas características de la propiedad, prediga su precio de venta.

### Pensando como un/a Data Scientist

Responde la siguientes pregunta:

1. ¿Qué datos crees que te ayudarían a trabajar en el problema? ¿Por qué?

   > **Importante**: NO deberás buscar esos datos, solamente justificar qué información crees que te ayudaría a resolver la problemática planteada.

   Los datos que considero relevantes y que me ayudarían a determinar el precio de una propiedad son:

   **UBICACIÓN (ZONA)**

   La *ubicación* de la propiedad la considero determinante a la hora de tasar el precio final de una propiedad o el alquiler, ya que, basados en este atributo, adicionamos o quitamos características adicionales a la propiedad respecto al entorno, que aportan facilidad o comodidad a la vida cotidiana de una persona.

   Por ejemplo, la ubicación de una propiedad puede influir en la contaminación acústica que esta posee por su cercanía o lejanía a ciertas calles con tránsito frecuente. Los accesos que permiten cierta ubicación al igual que la lejanía o proximidad a la urbe pueden influir en la cantidad de potenciales clientes que puede transitar en la misma si es que pensamos en la locación de negocio de compra/venta.

   También, los servicios que la propiedad tiene o carece, varían de acuerdo a la ubicación. Por ejemplo puede afectar el acceso a la red de internet, también condiciona los servicios cloacales, la conexión de agua, electricidad o gas. Todos estos servicios básicos esenciales tienen directa relación e impacto en la vida cotidiana de las personas y adicionan calidad de vida el contar con ellos.

   No hay que olvidar el mantenimiento que posee la ubicación por el municipio, acto que vemos reflejado en el alumbrado público, el estado de las calles o veredas, la parquización, etc., pueden influir directamente en el costo de la propiedad.

   **TIPO DE VIVIENDA**

   Muchas características que tienen las propiedades también son condicionadas por el *tipo de vivienda* que estamos analizando ya que algunas de ellas varían por el fin que se le va a dar a la misma. Por ejemplo, la presencia de cochera, pileta o cantidad de mts<sup>2</sup> de la misma pueden variar de acuerdo a si analizamos una casa, un departamento, un penthouse, pero, tal vez no sean relevante si analizamos una propiedad para establecer un negocio, un deposito u otro fin.

   Si tomamos en cuenta casas y departamentos, es probable que el último se limite en mts<sup>2</sup> y no cuente con pileta ni cochera pero pueda tener una ubicación cercana al centro de la ciudad.

   A mi juicio, lo veo condicionante a la hora de tasar el precio de una propiedad ya que este atributo en particular condiciona características que pueden hacer elevar el precio de la misma.

   **SUPERFICIE (mts<sup>2</sup>)**

   Por último y no menos importante, la superficie que tiene la propiedad tiene estrecha relación con el precio, como a su vez, con la cantidad de ambientes que pudiese entender. En mi experiencia al momento de elegir una propiedad para vivir, mayor cantidad de superficie de la misma implica que la misma sea espaciosa, con mayor cantidad de ambientes y habitaciones generando mayor comodidad. Adicionalmente, mayor cantidad de superficie implica que los metros cuadrados construidos de la misma se pueda extender generando un valor agregado a la misma, siempre y cuando sea posible y se permita (casas, casa quinta, chacras, etc.).

## **Parte 2**

### Análisis Exploratorio de Datos

En esta sección, debes realizar un *Análisis Exploratorio de Datos* sobre el dataset de propiedades de Properati. Es importante que respondas las siguientes preguntas durante el análisis:

- ¿Qué tamaño tiene el dataset? ¿Cuántas instancias y cuántas columnas?.
- ¿Cuántos valores faltantes hay en cada columna?.
- ¿Cómo es la distribución de cada variable? Deberás hacer histogramas para las variables numéricas y gráficos de barras para las variables categóricas.
- ¿Cómo se relacionan las variables entre sí? ¿Qué tipo de gráfico será conveniente para presentar esta información?.
- ¿Cómo están correlacionadas las variables numéricas? ¿Qué tipo de gráfico será conveniente para presentar esta información? ¿Cuáles serán los mejores predictores de la variable de interés?.

Vas a encontrar instrucciones para responder estas preguntas. Es importante aclarar que estas instrucciones corresponden al **mínimo entregable** que esperamos en la consigna.

> **Comentarios sobre el dataset**
>
> * Nosotros ya hicimos un *curado* sobre el dataset que puedes descargar directamente de la página de Properati. Muchos de los pasos que hicimos para curar el conjunto de datos los veremos durante el Bloque 2 de la carrera.
> * Si tienes dudas sobre qué representa alguna de las columnas, puedes consultar [aquí](https://www.properati.com.ar/data/). Notarás que algunas columnas fueron descartadas.
> * `Capital Federal` refiere a la Ciudad de Buenos Aires. `Bs.As. G.B.A. Zona Norte`, `Bs.As. G.B.A. Zona Sur` y `Bs.As. G.B.A. Zona Oeste` son regiones que conforman el [Gran Buenos Aires](https://es.wikipedia.org/wiki/Gran_Buenos_Aires), un conjunto de ciudades que rodean a la Ciudad de Buenos Aires.

### Desafío

En el dataset provisto hay mucha información, más allá del problema planteado. Propone una pregunta que pueda ser respondida por el dataset e intenta responderla. ¿Cuáles son los sesgos de la respuesta obtenida? (¿Cuán generalizable es la respuesta obtenida?) ¿Necesitas información complementaria? ¿Cómo la obtendrías?.

Por ejemplo: ¿Cuál es el barrio más caro de Buenos Aires? Probablemente puedas responder esta pregunta con este dataset. Pero podría ocurrir que la respuesta esté sesgada. ¿Cómo? Tal vez las propiedades más caras no se publican de forma online, sino que utilizan otro canal de venta.

#### Consignas mínimo entregable

1. Importa las librerías necesarias para trabajar en la consigna.

2. Carga el dataset usando las funcionalidades de ***Pandas***. Imprimir cuántas filas y columnas tiene, y sus cinco primeras instancias.

3. **Valores Faltantes**: imprime en pantalla los nombres de las columnas y cuántos valores faltantes hay por columna.

4. **Tipos de propiedad**: ¿Cuántos tipos de propiedad hay publicados según este dataset? ¿Cuántos instancias por cada tipo de propiedad hay en el dataset? Responde esta pregunta usando las funcionalidad de Pandas y con un gráfico apropiado de ***seaborn***.

    > **Pistas**: Te puede ser útil googlear cómo rotar las etiquetas del eje ```x```.

5. ¿De qué regiones son las publicaciones? Haz gráficos de barras para las variables `l2` y `l3`. Si te animas, puedes hacer los dos gráficos usando `subplot` de ***Matplotlib***. Dale un tamaño apropiado a la figura para que ambos gráficos se visualicen correctamente.

6. **Filtrando el Dataset**: A partir de los resultados del punto 4. y 5., selecciona las tres clases más abundantes de tipos de propiedad y la región con más propiedades publicadas. Crea un nuevo DataFrame con aquellas instancias que cumplen con esas condiciones e imprime su `shape`.

    > **Checkpoint**: deberías tener un dataset con **91.485 instacias**, **19 columnas**. 

7. Distribuciones y relaciones de a pares: Estudia la distribución y las relaciones de a pares de las variables ```rooms```, ```bedrooms```, ```bathrooms```, ```surface_total```, ```surface_covered```, ```price``` para cada tipo de propiedad. Para ello, ten en cuenta:

    * Obtiene estadísticos que te sirvan para tener una primera idea de los valores que abarcan estas variables. ¿Cuáles crees que toman valores que tal vez no tengan mucho sentido?.
    * Algunas instancias tienen valores de superficie (```surface_total```) muy grandes y dificultan la correcta visualización. Estudia la distribución de esa variable y filtra por un valor razonable que te permita obtener gráficos comprensibles. Puede ser útil un boxplot para determinar un rango razonable.
    * Lo mismo ocurre con valores de superficie total muy chico.
    * Las propiedades no pueden tener ```surface_covered``` mayor a ```surface_total```. Si eso sucede, debes filtrar esas instancias.
    * El rango de precios que toman las propiedades es muy amplio. Estudia la distribución de esa variable y filtra por un valor razonable que te permita obtener gráficos comprensibles. Puede ser útil un boxplot para determinar un rango razonable.
    * Una vez filtrado el dataset, puedes utilizar la función ```pairplot``` de ***seaborn***.

8. **Correlaciones**: Estudia la correlación entre las variables ```rooms```, ```bedrooms```, ```bathrooms```, ```surface_total```, ```surface_covered```, ```price```. ¿Cuáles son las mejores variables para predecir el precio? ¿Qué diferencias encuentras según cada tipo de propiedad?.

## **Parte 3**
### Machine Learning

En esta sección, debes entrenar dos modelos de Machine Learning *—uno de vecinos más cercanos y otro de árboles de decisión—* para predecir el precio de las propiedades tipo `Departamento` en la Ciudad Autónoma de Buenos Aires (`Capital Federal`). Para ello, no debes olvidarte de:

- Elegir una métrica apropiada para evaluar los resultados de los modelos.
- Seleccionar las variables predictoras (`X`) y la variable a predecir (`y`).
- Realizar un Train/Test split de los datos.
- Generar un modelo *benchmark* y evaluarlo.
- Entrenar un modelo de vecinos más cercanos y un modelo de árbol de decisión con hiperparámetros iniciales de su elección.
- Evaluar los modelos obtenidos. Para ello, evalúa la métrica elegida en el conjunto de Test y en el conjunto de Train. También, realiza gráficos de valores reales vs. valores predichos.
- Mejorar el desempeño de sus modelos optimizando el número de vecinos y la profundidad del árbol, respectivamente.
- Entre los modelos entrenados, ¿cuál elegirías para utilizar? ¿Por qué?.
- Ser **crítico/a** con la metodología utilizada. Por ejemplo, responde las siguientes preguntas: ¿Qué información no estás usando que podría ayudar al modelo? ¿Qué información puede estar demás o repetida?.

Estos lineamientos corresponden al **mínimo entregable** de esta sección.

> **IMPORTANTE**
>
> Para asegurarnos que trabajes con un dataset apropiados, debes volver a cargar los datos y realizar el siguiente filtrado:
>
> * Selecciona aquellas propiedades en *Capital Federal* y cuyo tipo de propiedad es *Departamento*, *PH* o *Casa*.
> * Selecciona aquellas propiedades cuya *superficie total* es menor a 1.000 m<sup>2</sup> y mayor a 15 m<sup>2</sup>.
> * Selecciona aquellas propiedades cuya *precio* es menor 4.000.000 dólares.
> * Selecciona las columnas `rooms`, `bedrooms`, `bathrooms`, `surface_total`, `surface_covered` y `price`.
> * Descarta aquellas instacias con valores faltantes.
>
> **CHECKPOINT**
>
> Deberías obtener un dataset con **81.021 instacias** y **6 columnas**.
