# Modelo de triagem de risco de AVC das Fases 1 e 2, reaproveitado como ferramenta
# do fluxo. Treina o Random Forest (mesmo pré-processamento e hiperparâmetros
# otimizados pelo algoritmo genético) e salva em results/fase3/modelos/.
# Rodar da raiz para treinar: python -m fase3.fluxos.triagem_avc
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fase3 import config

NUMERICAS = ['age', 'avg_glucose_level', 'bmi']
CATEGORICAS = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
BINARIAS = ['hypertension', 'heart_disease']

# hiperparâmetros do campeão da Fase 2 (Random Forest otimizado pelo GA, fitness F2)
PARAMS_RF = {'n_estimators': 200, 'max_depth': 5, 'min_samples_split': 2, 'min_samples_leaf': 10,
             'max_features': 'sqrt', 'class_weight': 'balanced'}

_modelo = None


def treinar():
    df = pd.read_csv(config.CSV_AVC)
    df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
    df = df[df['gender'] != 'Other'].drop(columns=['id'])
    X, y = df.drop(columns=['stroke']), df['stroke']
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify=y,
                                                              random_state=config.RANDOM_STATE)
    prep = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), NUMERICAS),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAS),
        ('bin', 'passthrough', BINARIAS),
    ])
    pipe = Pipeline([('prep', prep), ('modelo', RandomForestClassifier(random_state=config.RANDOM_STATE,
                                                                        n_jobs=-1, **PARAMS_RF))])
    pipe.fit(X_trainval, y_trainval)
    from sklearn.metrics import fbeta_score, recall_score, roc_auc_score
    pred = pipe.predict(X_test)
    print(f'Teste: recall={recall_score(y_test, pred):.3f} | F2={fbeta_score(y_test, pred, beta=2):.3f} | '
          f'AUC={roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1]):.3f}')
    config.DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, config.ARQ_MODELO_AVC)
    print(f'Modelo salvo em {config.ARQ_MODELO_AVC}')
    return pipe


def carregar():
    global _modelo
    if _modelo is None:
        _modelo = joblib.load(config.ARQ_MODELO_AVC) if config.ARQ_MODELO_AVC.exists() else treinar()
    return _modelo


def nomes_features(pipe):
    prep = pipe.named_steps['prep']
    return NUMERICAS + list(prep.named_transformers_['cat'].get_feature_names_out(CATEGORICAS)) + BINARIAS


def estimar_risco(paciente, n_fatores=4):
    """Recebe o dicionário do prontuário e devolve (probabilidade, fatores que mais
    pesaram). Os fatores vêm do SHAP (como nas Fases 1 e 2); se o shap não estiver
    disponível, cai para a importância global do modelo."""
    pipe = carregar()
    X = pd.DataFrame([{
        'age': paciente['idade'], 'gender': paciente['gender'], 'hypertension': paciente['hypertension'],
        'heart_disease': paciente['heart_disease'], 'ever_married': paciente['ever_married'],
        'work_type': paciente['work_type'], 'Residence_type': paciente['Residence_type'],
        'avg_glucose_level': paciente['avg_glucose_level'], 'bmi': paciente['bmi'],
        'smoking_status': paciente['smoking_status'],
    }])
    prob = float(pipe.predict_proba(X)[0, 1])
    nomes = nomes_features(pipe)
    x_t = pipe.named_steps['prep'].transform(X)
    try:
        import shap
        valores = np.array(shap.TreeExplainer(pipe.named_steps['modelo']).shap_values(x_t))
        contrib = valores[0, :, 1] if valores.ndim == 3 else valores[0]
    except Exception:
        contrib = pipe.named_steps['modelo'].feature_importances_ * np.sign(x_t[0])
    fatores = pd.Series(contrib, index=nomes)
    fatores = fatores.reindex(fatores.abs().sort_values(ascending=False).index)[:n_fatores]
    return prob, {nome: round(float(v), 3) for nome, v in fatores.items()}


if __name__ == '__main__':
    treinar()
