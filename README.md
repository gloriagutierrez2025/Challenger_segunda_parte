# Challenger_segunda_parte Telecom X

**Descripción del Proyecto**

Informe para detectar variables relevantes para la cancelación en Servicios empresa Telecom X

**Objetivos**

El objetivo principal es identificar patrones influyentes en la cancelación de servicios en empresa Telecom X, con la ayuda de herramientas estadísticas y modelos de clasificación que den cuenta de forma objetiva cuáles son las mejores variables para utilizar en los modelos de predicción.

**Estructura** 
El proyecto se trabajó en Google Colab con un cuaderno principal y varias carpetas de apoyo.
Contiene la secuencia completa de pasos: importación de librerías, carga y exploración de datos, preparación y limpieza, modelización y evaluación de resultados.

**Librerías utilizadas:**

-import pandas as pd

-import seaborn as sns

-import numpy as np

-import requests

-import matplotlib.pyplot as plt

-import plotly.express as px

-import plotly.graph_objects as go

-import statsmodels.api as sm

-from sklearn.model_selection import train_test_split

-from sklearn.neighbors import KNeighborsClassifier

-from sklearn.compose import ColumnTransformer

-from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

-from sklearn.pipeline import Pipeline

-from sklearn.metrics import classification_report

-from sklearn.linear_model import LogisticRegression

-from sklearn.dummy import DummyClassifier

-from sklearn.tree import DecisionTreeClassifier

-import matplotlib.pyplot as plt

-from sklearn.tree import plot_tree

-import pickle 

-from sklearn.ensemble import RandomForestClassifier

-from sklearn.metrics import confusion_matrix, classification_report

-from imblearn.over_sampling import SMOTE

-from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

-from sklearn.metrics import PrecisionRecallDisplay

**Clasificación de variables:** 

Numéricas

Categóricas 

**Normalización y codificación**

A las numéricas se les aplicó normalización con StandardScaler para que todas quedaran en la misma escala.

A las categóricas se les aplicó One-Hot Encoding (cuando eran solo categorías sin orden) o Label Encoding (cuando tenían un orden 
natural). Esto permitió convertir texto en números que los modelos pudieran interpretar.

**División en entrenamiento y prueba**

El dataset se dividió en 70% para entrenamiento y 30% para prueba.

Se usó la opción stratify=y para que la proporción entre clases se mantuviera igual en ambos conjuntos.

**Justificación:**

Normalizar evita que una variable con valores grandes “pese más” que las demás en el modelo.

Codificar categorías era necesario porque los modelos no leen datos en formato texto.

Separar en entrenamiento y prueba asegura que el modelo se evalúe con datos nuevos y no se sobreajuste.

Evaluación de desbalance y aplicación de Smote para balancear.
Gráfico
<img width="548" height="465" alt="Image" src="https://github.com/user-attachments/assets/8728d66d-e45d-425d-8ddd-b2efae1f4aa5" />

<img width="455" height="469" alt="Image" src="https://github.com/user-attachments/assets/1dc8d3f2-8b72-486d-b1b9-87c1fcea562d" />

Realización de análisis de correlación y selección de variables.

Gráfico
<img width="709" height="540" alt="Image" src="https://github.com/user-attachments/assets/a7dc0621-2ff3-4dfb-b20c-d7fe78df1db8" />

Box
<img width="1160" height="444" alt="Image" src="https://github.com/user-attachments/assets/0dba035f-3a56-45fa-9db0-a9c21867b5ed" />

Apicación de modelos de clasificación.
Modelo del Árbol antes de Smote
<img width="1185" height="486" alt="Image" src="https://github.com/user-attachments/assets/77765e1a-bfd6-4bde-8198-c7cca3b7b9f5" />

Modelo del Árbol después de Smote
<img width="1213" height="485" alt="Image" src="https://github.com/user-attachments/assets/706ada6c-08c8-443a-9d33-6d6b92b748aa" />

Evaluación el rendimiento de los modelos.
Gráfico
<img width="455" height="537" alt="Image" src="https://github.com/user-attachments/assets/70ab9add-ed7e-4795-9c4d-93aac87d4023" />

Conclusión

Con la información analizada se decidió evaluar en los modelos tres variables que tenían mayor correlación respecto a cancelación, los modelos reflejaron que la antiguedad puede ser un indicador relevante en clientes que cancelaban sus servicios, reflejando a menor antiguedad mayor indice de cancelación en los servicios de Telecom X. Los clientes que recién han contratado servicios y no han generado aún fidelidad son los más propensos a cancelar, para esto se hacen algunas recomendaciones a continuación:

Recomendaciones para disminuir la evasión, enfocada en las tres variables de mayor correlación con nuestra variable objetivo Cancelación:

**Riesgo cancelación en clientes con menos antiguedad, como acciones a seguir podemos ofrecer:**

-Onboarding personalizado y seguimiento inicial.

-Incentivos para primeros meses (descuentos, beneficios extras).

-Comunicación proactiva para ver satisfacción de clientes o si existe algún problema que pueda influir en cancelación de servicios.

**Riesgo por cancelación total facturado, como acciones a seguir:**

-Revisar planes y facturación, ofrecer planes más flexibles, descuentos por fidelidad, acumulación de beneficios.

-Comunicación proactiva para ver satisfacción de clientes o si existe algún problema que pueda influir en cancelación de servicios.

**Riesgo por cancelación pago mensual alto, como acciones a seguir:**

-Opciones de fraccionamiento.

-Comunicación proactiva para ver satisfacción de clientes o si existe algún problema que pueda influir en cancelación de servicios.
