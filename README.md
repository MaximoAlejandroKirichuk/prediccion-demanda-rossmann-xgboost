# Predicción de demanda retail con Machine Learning

**Impacto de la ingeniería de variables temporales en algoritmos de Machine Learning para la predicción de demanda.**

Trabajo práctico grupal de investigación — *Programación en Python*, Ingeniería en Sistemas, Universidad Abierta Interamericana (UAI).

**Integrantes:** Benjamín Aguila · Maximo Kirichuk · Lucas Molinari · Maximo Yapura

---

## ¿De qué trata el proyecto?

Una cadena de tiendas necesita estimar cuánto va a vender cada local por día para planificar stock y personal. Este trabajo mide **cuánto mejora un modelo de Machine Learning cuando, además de las variables de contexto (promociones, feriados), le damos variables temporales** (qué vendió la tienda los días anteriores y en qué momento del año estamos).

Usamos el dataset público **Rossmann Store Sales** (Kaggle): ventas diarias de 1.115 tiendas entre 2013-01-01 y 2015-07-31.

## Configuraciones comparadas

| Configuración | Variables | Idea |
|---|---|---|
| **Baseline 1** | — | Predice el promedio histórico de la tienda |
| **Baseline 2** | — | Predice el promedio de la tienda para ese día de la semana |
| **Config A** | `StoreMean`, `Promo`, `StateHoliday`, `SchoolHoliday` | Solo contexto |
| **Config A + DOW** | Config A + `DayOfWeek` | Experimento extra |
| **Config B** | Config A + calendario (`DayOfWeek`, `Month`, `WeekOfYear`, `DayOfMonth`) + lags (`lag_1`, `lag_7`, `lag_14`, `lag_28`) | Contexto + variables temporales |

Además se corre una **ablación** (A + calendario, A + lags, B completa) para ver qué pieza aporta la mejora.

El modelo es **XGBoost** (`XGBRegressor`) con los mismos hiperparámetros en todas las configuraciones, para que la comparación sea justa.

## Metodología

- **Datos:** se usan solo los días con tienda abierta y ventas > 0.
- **Validación temporal (*expanding window*):** 5 folds, cada uno testea sobre 42 días (6 semanas) y entrena con todo lo anterior. No se usa split aleatorio porque filtraría información del futuro.
- **Sin filtración (*leakage*):** `StoreMean` se calcula solo con el train de cada fold.
- **Lags por fecha de calendario:** se busca la venta de la misma tienda en la fecha exacta `hoy − N días`; si ese día no existe (tienda cerrada) queda `NaN`.
- **Horizonte de 1 día:** para predecir el día D se usa lo ocurrido hasta D−1.
- **Métricas:** MAE, RMSE, MAPE (%) y R², reportadas como promedio y desvío entre folds.

## Estructura del repositorio

```
├── Tp_python_A_y_B.ipynb   # Notebook completo: Config A (secciones 1-14) y Config B (15-22)
└── README.md
```

## Cómo ejecutarlo

1. Descargar `train.csv` desde [Rossmann Store Sales (Kaggle)](https://www.kaggle.com/c/rossmann-store-sales/data).
2. Abrir `Tp_python_A_y_B.ipynb` en [Google Colab](https://colab.research.google.com/) (*Archivo → Subir notebook*).
3. Ejecutar las celdas en orden. En la sección 3 aparece un botón para **subir `train.csv`**; el notebook no depende de rutas ni de Google Drive.
4. Los CSV con las métricas se descargan al final de cada configuración.

Dependencias (ya incluidas en Colab, salvo que falte alguna): `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`.

## Resultados

Resultados de la **Configuración A** sobre el dataset completo (promedio de los 5 folds):

| Modelo | MAE | MAPE (%) | R² |
|---|---|---|---|
| Baseline: media por tienda | 1420.06 | 22.01 | 0.6125 |
| Baseline: media tienda × día | 1207.93 | 17.57 | 0.7155 |
| XGBoost Config A | 1046.69 | 15.74 | 0.7673 |
| XGBoost Config A + DOW | 916.22 | 13.89 | 0.8165 |

Los resultados de la **Configuración B** y la comparación final con A se agregan una vez ejecutado el notebook sobre el dataset real.

## Contenido del notebook

- Exploración rápida (ventas por día de la semana y por promoción).
- Gráficos de comparación de modelos, MAE por fold y ablación.
- Ventas reales vs. predichas en el último fold, importancia de variables y error por día de la semana.
