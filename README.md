# Regresi-n-log-stica
Este proyecto implementa un modelo de Regresión Logística para predecir la supervivencia de los pasajeros del Titanic utilizando el dataset de seaborn.

## 📂 Contenido del proyecto
- **Carga del dataset** con `seaborn.load_dataset("titanic")`.  
- **Preprocesamiento**:
  - Imputación de valores nulos (mediana en numéricas, moda en categóricas).  
  - Escalado robusto de numéricas (`RobustScaler`).  
  - Codificación one-hot de categóricas (`OneHotEncoder`).  
- **Construcción de pipeline completo** con `ColumnTransformer` y `Pipeline`.  
- **Entrenamiento y evaluación** de un modelo de **Regresión Logística** (`LogisticRegression`).  
- **Evaluación de métricas**:
  - `classification_report` con precisión, recall, f1-score y accuracy en validación y test.  
- **Función auxiliar `mostrar_datos_procesados`**:
  - Permite ver cómo quedan las features finales después del preprocesamiento (incluyendo nombres de variables generadas por el OneHotEncoder).  

---

## ⚙️ Requisitos
Instalar dependencias principales:
```bash
pip install pandas seaborn scikit-learn
