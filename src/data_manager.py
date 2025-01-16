import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

class AgriculturalDataManager:

    def __init__(self):
        """Initialise les variables pour stocker les données et outils nécessaires."""
        self.monitoring_data = None
        self.weather_data = None
        self.soil_data = None
        self.yield_history = None
        self.scalar = StandardScaler()

    def load_data(self):
        """Charge les données à partir des fichiers CSV et gère les exceptions."""
        try: 
            self.monitoring_data = pd.read_csv("data/monitoring_cultures.csv", parse_dates=["date"])
            self.weather_data = pd.read_csv("data/meteo_detaillee.csv", parse_dates=["date"])
            self.soil_data = pd.read_csv("data/sols.csv")
            self.yield_history = pd.read_csv("data/historique_rendements.csv", parse_dates=["date"])
        except FileNotFoundError as e:
            print(f"Erreur : fichier introuvable. {e}")
        except Exception as e:
            print(f"Erreur lors du chargement des données : {e}")

    def clean_data(self):
        """Nettoie les données, par exemple en s'assurant que les valeurs sont cohérentes."""
        self.weather_data['rayonnement_solaire'] = self.weather_data['rayonnement_solaire'].abs()

    def meteo_data_hourly_to_daily(self):
        """Convertit les données horaires météo en moyennes journalières."""
        try:
            self.weather_data['date'] = pd.to_datetime(self.weather_data['date'], errors='coerce')
            self.weather_data = (
                self.weather_data
                .set_index('date')
                .resample('D')
                .mean()
                .reset_index()
            )
        except Exception as e:
            print(f"Erreur lors de l'agrégation des données météo : {e}")

    def _setup_temporal_indices(self):
        """Configure les index temporels pour les différentes séries de données."""
        try:
            self.monitoring_data.set_index('date', inplace=True)
            self.weather_data.set_index('date', inplace=True)
            self.yield_history.set_index('date', inplace=True)
        except Exception as e:
            print(f"Erreur lors de la configuration des index temporels : {e}")

    def prepare_features(self):
        """Prépare les données en les fusionnant et en les enrichissant avec les données historiques."""
        try:
            self.monitoring_data = self.monitoring_data.sort_values(by="date")
            self.weather_data = self.weather_data.sort_values(by="date")

            # Fusion des données de monitoring et météo
            data = pd.merge_asof(
                self.monitoring_data,
                self.weather_data,
                on="date",
                direction='nearest'
            )

            # Ajout des données de sol
            data = pd.merge(data, self.soil_data, how='left', on="parcelle_id")            

            # Enrichissement avec l'historique des rendements
            data = self._enrich_with_yield_history(data)

            # Nettoyage des colonnes inutiles
            data.drop(columns=['latitude_y', 'longitude_y'], errors='ignore', inplace=True)
            data.rename(columns={'latitude_x': 'latitude', 'longitude_x': 'longitude'}, inplace=True)

            data.drop(columns=['culture_y'], errors='ignore', inplace=True)
            data.rename(columns={'culture_x': 'culture'}, inplace=True)

            # Sauvegarde des caractéristiques dans un fichier CSV
            data.to_csv("data/features.csv", index=False)
            print("Colonnes des données finales :", data.columns)

            return data

        except Exception as e:
            print(f"Erreur lors de la préparation des caractéristiques : {e}")

    def _enrich_with_yield_history(self, data):
        """Enrichit les données actuelles avec les rendements historiques."""
        try:
            data = pd.merge(
                data,
                self.yield_history,
                how="left",
                on=["parcelle_id", "date"]
            )
            return data
        except Exception as e:
            print(f"Erreur lors de l'enrichissement avec les rendements historiques : {e}")
            return data

    def get_temporal_patterns(self, parcelle_id):
        """Analyse les patterns temporels pour une parcelle donnée."""
        try:
            features = pd.read_csv("data/features.csv", parse_dates=["date"])
            parcelle_data = features[features["parcelle_id"] == parcelle_id]

            if "ndvi" not in parcelle_data.columns:
                raise KeyError("Colonne NDVI introuvable dans les données.")

            if parcelle_data.empty:
                raise ValueError(f"Aucune donnée trouvée pour parcelle_id : {parcelle_id}")

            parcelle_data.sort_values(by="date", inplace=True)
            parcelle_data.set_index("date", inplace=True)

            ndvi_series = parcelle_data["ndvi"].dropna()
            if len(ndvi_series) < 12:
                raise ValueError("Pas assez de points de données pour une décomposition saisonnière.")

            # Décomposition saisonnière
            decomposition = seasonal_decompose(ndvi_series, model="additive", period=12)

            # Modélisation linéaire pour analyser la tendance
            date_ordinal = parcelle_data.index.map(datetime.toordinal).values.reshape(-1, 1)
            ndvi_values = ndvi_series.values.reshape(-1, 1)

            trend_model = LinearRegression().fit(date_ordinal, ndvi_values)
            trend_slope = trend_model.coef_[0][0]
            trend_intercept = trend_model.intercept_[0]

            trend = {
                "pente": trend_slope,
                "intercept": trend_intercept,
                "variation_moyenne": trend_slope / ndvi_series.mean() if ndvi_series.mean() != 0 else 0
            }

            history = {
                "ndvi_trend": decomposition.trend.dropna(),
                "ndvi_seasonal": decomposition.seasonal,
                "ndvi_residual": decomposition.resid.dropna(),
                "ndvi_moving_avg": ndvi_series.rolling(window=30).mean().dropna(),
                "summary_stats": {
                    "mean_ndvi": ndvi_series.mean(),
                    "std_ndvi": ndvi_series.std(),
                    "min_ndvi": ndvi_series.min(),
                    "max_ndvi": ndvi_series.max(),
                }
            }

            return history, trend

        except Exception as e:
            print(f"Erreur dans get_temporal_patterns : {e}")
            return None, None

    def calculate_risk_metrics(self, data):
        """Calcule les métriques de risque basées sur les caractéristiques normalisées."""
        try:
            required_columns = ['parcelle_id', 'culture', 'rendement_estime', 'ph', 'matiere_organique']
            for col in required_columns:
                if col not in data.columns:
                    raise KeyError(f"Colonne requise manquante : {col}")

            # Normalisation des colonnes nécessaires
            normalized_data = self.scalar.fit_transform(data[['rendement_estime', 'ph', 'matiere_organique']])

            # Calcul de l'index de risque
            data['risk_index'] = (
                0.5 * normalized_data[:, 0] +  # Poids pour rendement
                0.3 * normalized_data[:, 1] +  # Poids pour pH
                0.2 * normalized_data[:, 2]    # Poids pour matière organique
            )

            # Attribution des catégories de risque
            data['risk_category'] = pd.cut(
                data['risk_index'],
                bins=[-np.inf, -1, 0, 1, np.inf],
                labels=['Très Bas', 'Bas', 'Modéré', 'Élevé']
            )

            # Agrégation par parcelle et culture
            grouped_data = data.groupby(['parcelle_id', 'culture']).agg(
                avg_risk_index=('risk_index', 'mean'),
                most_frequent_risk_category=('risk_category', lambda x: x.mode()[0] if not x.mode().empty else None)
            ).reset_index()

            # Sauvegarde des données agrégées
            grouped_data.to_csv("data/grouped_risk_metrics.csv", index=False)

            return grouped_data

        except Exception as e:
            print(f"Erreur lors du calcul des métriques de risque : {e}")
            return None

if __name__ == "__main__":
    data_manager = AgriculturalDataManager()

    # Chargement et préparation des données
    data_manager.load_data()
    data_manager.clean_data()
    data_manager.meteo_data_hourly_to_daily()
    print("------------------------------------------------------------------------------------------------------------")
    features = data_manager.prepare_features()
    # Analyse des patterns temporels pour une parcelle spécifique
    parcelle_id = "P001"
    history, trend = data_manager.get_temporal_patterns(parcelle_id)

    # Calcul des métriques de risque
    risk_metrics = data_manager.calculate_risk_metrics(features)

    print("---------------------------------------------Métriques de risque---------------------------------------------------------------")
    print(risk_metrics.head())

    if trend:
        print("----------------------------------------------Tendance NDVI-----------------------------------------------------------------------")
        print(f"Pente de la tendance : {trend['pente']:.4f} unités/an")
        print(f"Variation moyenne : {trend['variation_moyenne'] * 100:.2f}%")
print("------------------------------------------------------------------------------------------------------------")

