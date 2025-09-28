import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

df = sns.load_dataset("titanic")
print(df.head(10))

X = df.drop("survived", axis=1)
y = df["survived"]

#Dividomos en 60-20-20
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

#Separamos colum num y categ
num_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
cat_cols = X_train.select_dtypes(include=['object']).columns

num_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])


cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

processor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

full_pipeline = Pipeline([
    ('processor', processor),
    ('clf', LogisticRegression(solver="newton-cg", max_iter=1000))
])


full_pipeline.fit(X_train, y_train)

y_pred = full_pipeline.predict(X_val)
y_test_pred = full_pipeline.predict(X_test)


def mostrar_datos_procesados(pipeline, X_data, num_cols, cat_cols):
    # Obtener el processor del pipeline
    processor = pipeline.named_steps['processor']
    
    # Transformar los datos
    processed_data = processor.transform(X_data)
    
    # Obtener nombres de características
    # Para columnas numéricas
    num_feature_names = list(num_cols)
    
    # Para columnas categóricas (one-hot encoding)
    cat_processor = processor.named_transformers_['cat']
    onehot_encoder = cat_processor.named_steps['onehot']
    cat_feature_names = onehot_encoder.get_feature_names_out(cat_cols)
    
    # Combinar todos los nombres de características
    all_feature_names = num_feature_names + list(cat_feature_names)
    
    # Crear DataFrame
    df_processed = pd.DataFrame(processed_data, columns=all_feature_names)
    
    return df_processed

# Uso de la función con tus datos
df_processed = mostrar_datos_procesados(full_pipeline, X_train, num_cols, cat_cols)

print(df_processed.head(10))

print("Validación:")
print(classification_report(y_val, y_pred))

print("Test:")
print(classification_report(y_test, y_test_pred))
