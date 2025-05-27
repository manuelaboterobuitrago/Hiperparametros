# Clasificación de cáncer de mama con Random Forest y Optimización de Hiperparámetros
### 🧠 Objetivo

El objetivo principal de este proyecto es aplicar técnicas de ajuste y optimización de hiperparámetros usando modelos de aprendizaje automático para clasificación. 

Este proyecto implementa un pipeline de clasificación usando el conjunto de datos `load_breast_cancer` de `sklearn.datasets`. Se entrenaron y evaluaron tres modelos:

- **Baseline**: Random Forest con hiperparámetros por defecto.
- **Búsqueda Aleatoria**: Optimización de hiperparámetros con `RandomizedSearchCV`.
- **Optuna**: Optimización bayesiana con `Optuna`.

## 🔍 Dataset: Breast Cancer Wisconsin (Diagnostic) Dataset

Este dataset incluye características computadas a partir de imágenes digitalizadas de tumores mamarios. La variable objetivo es binaria:

- `0`: tumor **benigno**
- `1`: tumor **maligno**

### 📊 Descripción de las variables

Las variables independientes son todas numéricas y representan estadísticas básicas sobre los núcleos celulares de los tumores. Cada variable tiene tres versiones: media (`mean`), error estándar (`se`) y valor máximo (`worst`). Ejemplos:

| Variable | Descripción |
|----------|-------------|
| `mean radius` | Promedio del radio de los núcleos celulares |
| `mean texture` | Desviación estándar de los valores de gris |
| `mean perimeter` | Promedio del perímetro |
| `mean area` | Promedio del área de los núcleos |
| `mean smoothness` | Variación local en la suavidad de los contornos |
| `...` | ... se repite para `se` y `worst` en todas las variables anteriores |

Hay un total de **30 características numéricas** derivadas de las imágenes médicas.

---

## 📊 Mini Análisis Exploratorio de Datos (EDA)

- Número de instancias: 569
- Número de variables (características): 30
- Clases objetivo: `0 = Maligno`, `1 = Benigno`
- Las características están escaladas antes del entrenamiento.
- Se aplicó **reducción de dimensionalidad (PCA)** a 10 componentes principales, reteniendo más del 95% de la varianza.


## 📈 Resultados

| Modelo             | ROC AUC (Entrenamiento) | ROC AUC (Test) |
|--------------------|--------------------------|----------------|
| Baseline           | 0.987461                 | 0.993882       |
| Búsqueda Aleatoria | 0.988545                 | 0.992063       |
| Optuna             | 0.989680                 | 0.992394       |

> 🔍 El modelo **Baseline** obtuvo el mejor desempeño en el conjunto de prueba, aunque los tres modelos presentaron resultados sobresalientes.

## 📌 Conclusiones

- Los tres modelos superaron el 98% en AUC, lo que indica un excelente poder predictivo.
- La búsqueda de hiperparámetros no siempre garantiza una mejora en test; en este caso, el modelo base fue el mejor en ese conjunto.
- Optuna fue el modelo más eficiente en validación cruzada, lo cual sugiere un mejor desempeño generalizable.

## 📦 Librerías utilizadas

- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `sklearn` (para modelos, métricas, validación)
- `optuna` (optimización bayesiana)

---
## 👤 Autor

**Nombre:** Manuela Botero Buitrago  
