import os
import io
import math
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb

# =============================
# ⚙️ Config Streamlit
# =============================
st.set_page_config(page_title="Prévision de Demande – Logistique", page_icon="📦", layout="wide")

st.title("📦 Prévision de Demande – Dashboard Logistique")
st.caption("Vitrine freelance – simulation, entraînement LightGBM, export des prédictions.")

# =============================
# 📥 Données – Chargement
# =============================
DEFAULT_CSV_PATH = "demande_logistique_dataset.csv"  # place ton CSV à la racine du projet

uploaded = st.sidebar.file_uploader("Uploader un CSV (optionnel)", type=["csv"]) 

@st.cache_data(show_spinner=True)
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        # fallback : essaie le fichier local, sinon erreur guidée
        if os.path.exists(DEFAULT_CSV_PATH):
            df = pd.read_csv(DEFAULT_CSV_PATH)
        else:
            st.stop()
    # parse date si possible
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df

df_raw = load_data(uploaded)

st.success(f"Données chargées : {df_raw.shape[0]} lignes × {df_raw.shape[1]} colonnes")
with st.expander("Aperçu des données (5 premières lignes)"):
    st.dataframe(df_raw.head())

# =============================
# 🧪 Préprocessing & Features
# =============================
CAT_COLS = ["site_id", "produit_id", "meteo", "jour_semaine"]


def has_original_cats(df: pd.DataFrame) -> bool:
    return all(c in df.columns for c in ["site_id", "produit_id"]) and any(c in df.columns for c in ["meteo", "jour_semaine"]) 


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # sécurité colonnes
    for col in ["date", "quantite_demandee"]:
        if col not in df.columns:
            st.error(f"Colonne '{col}' absente du CSV. Vérifie tes noms de colonnes.")
            st.stop()

    # Si les colonnes catégorielles originales existent, on les garde pour groupby
    if has_original_cats(df):
        sort_keys = ["site_id", "produit_id", "date"]
    else:
        # jeu déjà dummifié → on ne peut trier que par date
        sort_keys = ["date"]

    df = df.sort_values(sort_keys).reset_index(drop=True)

    # Variables calendrier
    df['dayofweek'] = df['date'].dt.weekday
    df['month'] = df['date'].dt.month
    df['is_weekend'] = df['dayofweek'].isin([5,6]).astype(int)

    # Encodage cyclique
    df['day_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * (df['month']-1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (df['month']-1) / 12)

    # Lags & Rolling (si on a les clés d'origine)
    if has_original_cats(df):
        group_keys = ["site_id", "produit_id"]
        df['demand_shift_1']  = df.groupby(group_keys)['quantite_demandee'].shift(1)
        df['demand_shift_7']  = df.groupby(group_keys)['quantite_demandee'].shift(7)
        df['demand_shift_14'] = df.groupby(group_keys)['quantite_demandee'].shift(14)
        for w in [3,7,14,30]:
            df[f'demand_roll_{w}'] = (
                df.groupby(group_keys)['quantite_demandee']
                  .transform(lambda x: x.shift(1).rolling(window=w, min_periods=1).mean())
            )
        # Différences
        df['demand_diff_1'] = df.groupby(group_keys)['quantite_demandee'].diff(1)
        df['demand_diff_7'] = df.groupby(group_keys)['quantite_demandee'].diff(7)
    else:
        # fallback minimal si dummifié
        df['demand_shift_1'] = df['quantite_demandee'].shift(1)
        df['demand_roll_7'] = df['quantite_demandee'].shift(1).rolling(7, min_periods=1).mean()

    # Imputation par médiane globale (simple & robuste)
    med = df['quantite_demandee'].median()
    for col in df.columns:
        if df[col].dtype.kind in "biufc" and df[col].isna().any():
            df[col] = df[col].fillna(med)

    return df


def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # encoder uniquement si colonnes originales présentes
    cols_to_encode = [c for c in CAT_COLS if c in df.columns]
    if cols_to_encode:
        df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)
    return df


with st.spinner("Préprocessing en cours…"):
    df_feat = add_time_features(df_raw)
    df_model = one_hot_encode(df_feat)

st.write("**Jeu de données après features** :", df_model.shape)
with st.expander("Voir les colonnes (features)"):
    st.write(list(df_model.columns))

# =============================
# ✂️ Split temporel & Train
# =============================
FEATURE_DROP = ["date", "quantite_demandee"]
X = df_model.drop(columns=[c for c in FEATURE_DROP if c in df_model.columns])
y = df_model['quantite_demandee']

split_index = int(len(df_model)*0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[:split_index].copy()  # placeholder fix
# Correct y_test
y_test = y.iloc[split_index:]

# Hyperparamètres par défaut (les tiens, meilleurs trouvés)
DEFAULT_PARAMS = {
    'learning_rate': 0.09222773295678245,
    'num_leaves': 46,
    'max_depth': 15,
    'min_data_in_leaf': 21,
    'feature_fraction': 0.9391439020293932,
    'bagging_fraction': 0.91853647928786,
    'bagging_freq': 7
}

st.sidebar.subheader("Paramètres LightGBM")
use_defaults = st.sidebar.checkbox("Utiliser mes meilleurs params (recommandé)", value=True)

params = DEFAULT_PARAMS.copy()
if not use_defaults:
    params['learning_rate']     = st.sidebar.slider('learning_rate', 0.01, 0.3, float(params['learning_rate']))
    params['num_leaves']        = st.sidebar.slider('num_leaves', 20, 300, int(params['num_leaves']))
    params['max_depth']         = st.sidebar.slider('max_depth', 3, 20, int(params['max_depth']))
    params['min_data_in_leaf']  = st.sidebar.slider('min_data_in_leaf', 10, 200, int(params['min_data_in_leaf']))
    params['feature_fraction']  = st.sidebar.slider('feature_fraction', 0.5, 1.0, float(params['feature_fraction']))
    params['bagging_fraction']  = st.sidebar.slider('bagging_fraction', 0.5, 1.0, float(params['bagging_fraction']))
    params['bagging_freq']      = st.sidebar.slider('bagging_freq', 1, 7, int(params['bagging_freq']))

params.update({
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'verbosity': -1,
    'random_state': 42,
})

@st.cache_resource(show_spinner=True)
def train_model(X_train, y_train, params):
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    return model

with st.spinner("Entraînement du modèle…"):
    model = train_model(X_train, y_train, params)

# Évaluation
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
rmse = mean_squared_error(y_test, preds, squared=False)

c1, c2, c3 = st.columns(3)
c1.metric("MAE", f"{mae:.2f}")
c2.metric("RMSE", f"{rmse:.2f}")
c3.metric("N Features", f"{X.shape[1]}")

# =============================
# 📈 Graphiques
# =============================
with st.expander("Comparaison valeurs réelles vs prédictions (test)", expanded=True):
    comp = pd.DataFrame({
        'date': df_model['date'].iloc[split_index:].values,
        'y_test': y_test.values,
        'y_pred': preds
    })
    comp = comp.sort_values('date')
    st.line_chart(comp.set_index('date'))

# =============================
# 🔎 Importance des variables
# =============================
with st.expander("Importance des variables"):
    fi = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_.astype(float)
    }).sort_values('importance', ascending=False).head(30)
    st.bar_chart(fi.set_index('feature'))

# =============================
# 🧮 Simulation de scénario
# =============================
st.header("🧪 Simulation de scénario")

# Construire des listes pour UI à partir des données si dispo
sites = sorted(df_raw['site_id'].unique().tolist()) if 'site_id' in df_raw.columns else []
prods = sorted(df_raw['produit_id'].unique().tolist()) if 'produit_id' in df_raw.columns else []
meteo_vals = sorted(df_raw['meteo'].unique().tolist()) if 'meteo' in df_raw.columns else []
jours = sorted(df_raw['jour_semaine'].unique().tolist()) if 'jour_semaine' in df_raw.columns else []

colA, colB, colC, colD = st.columns(4)
site_sel = colA.selectbox("Site", options=sites if sites else ["(non dispo)"])
prod_sel = colB.selectbox("Produit", options=prods if prods else ["(non dispo)"])
date_sel = colC.date_input("Date scénario", value=pd.to_datetime(df_raw['date']).max() if 'date' in df_raw.columns else datetime.today())
meteo_sel = colD.selectbox("Météo", options=meteo_vals if meteo_vals else ["Ensoleillé","Nuageux","Pluvieux","Orageux"])

colE, colF, colG, colH = st.columns(4)
jour_sel = colE.selectbox("Jour semaine", options=jours if jours else ["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"])
temp_sel = colF.number_input("Température (°C)", value=int(df_raw['temp_celsius'].median()) if 'temp_celsius' in df_raw.columns else 20)
promo_sel = colG.selectbox("Promo active", options=[0,1], index=0)
stock_sel = colH.number_input("Stock disponible", value=int(df_raw['stock_disponible'].median()) if 'stock_disponible' in df_raw.columns else 200)

livr_sel = st.slider("Temps de livraison moyen (jours)", 1, 10, int(df_raw['temps_livraison_moyen_j'].median()) if 'temps_livraison_moyen_j' in df_raw.columns else 3)


def build_scenario_row(df_raw: pd.DataFrame) -> pd.DataFrame:
    # point de départ : dernière observation du couple (site, produit) si dispo
    base = None
    if all(c in df_raw.columns for c in ['site_id','produit_id']):
        hist = df_raw[(df_raw['site_id']==site_sel) & (df_raw['produit_id']==prod_sel)].copy()
        if not hist.empty:
            hist['date'] = pd.to_datetime(hist['date'], errors='coerce')
            base = hist.sort_values('date').tail(30)  # sert à calculer les rollings
    if base is None or base.empty:
        base = df_raw.copy().sort_values('date').tail(30)

    # on fabrique une nouvelle ligne avec nos entrées UI
    new = {
        'date': pd.to_datetime(date_sel),
        'quantite_demandee': np.nan,  # inconnue à prédire
        'stock_disponible': stock_sel,
        'temps_livraison_moyen_j': livr_sel,
        'temp_celsius': temp_sel,
        'promo_active': promo_sel,
    }
    if 'site_id' in df_raw.columns: new['site_id'] = site_sel
    if 'produit_id' in df_raw.columns: new['produit_id'] = prod_sel
    if 'meteo' in df_raw.columns: new['meteo'] = meteo_sel
    if 'jour_semaine' in df_raw.columns: new['jour_semaine'] = jour_sel

    base_plus = pd.concat([base, pd.DataFrame([new])], ignore_index=True)
    base_plus = add_time_features(base_plus)
    base_plus = one_hot_encode(base_plus)

    # garde uniquement la dernière ligne pour la prédiction
    row = base_plus.tail(1).copy()
    # aligne les colonnes sur X (features du modèle)
    for col in X.columns:
        if col not in row.columns:
            row[col] = 0
    row = row[X.columns]
    return row

if st.button("🔮 Prédire la demande du scénario"):
    try:
        x_row = build_scenario_row(df_raw)
        y_hat = float(model.predict(x_row)[0])
        st.success(f"Demande prédite : **{y_hat:.1f}** unités")
    except Exception as e:
        st.error(f"Erreur de prédiction : {e}")

# =============================
# 💾 Export – modèle & prédictions
# =============================
col1, col2 = st.columns(2)

with col1:
    if st.button("💾 Sauvegarder le modèle (joblib)"):
        buf = io.BytesIO()
        joblib.dump(model, buf)
        buf.seek(0)
        st.download_button(
            label="Télécharger model_lgbm.joblib",
            data=buf,
            file_name="model_lgbm.joblib",
            mime="application/octet-stream"
        )

with col2:
    comp = pd.DataFrame({
        'date': df_model['date'].iloc[split_index:].values,
        'y_test': y_test.values,
        'y_pred': preds
    })
    csv_bytes = comp.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Exporter prédictions test (CSV)", csv_bytes, file_name="predictions_test.csv", mime="text/csv")

st.caption("© 2025 – Adel Abbou – Prévision de Demande – Streamlit Vitrine")