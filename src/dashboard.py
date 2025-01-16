from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Select, CustomJS, Span, HoverTool, ColorBar, LinearColorMapper, BasicTicker
from bokeh.plotting import figure, show, curdoc
from bokeh.palettes import RdYlBu11 as palette
import pandas as pd
import numpy as np
from data_manager import AgriculturalDataManager

class AgriculturalDashboard:
    def __init__(self, data_manager):
        """
        Initialise la classe AgriculturalDashboard.
        """
        self.data_manager = data_manager
        self.full_yield_source = None
        self.full_ndvi_source = None
        self.yield_source = None
        self.ndvi_source = None
        self.stress_source = None
        self.full_stress_source = None
        self.create_data_sources()

    def create_data_sources(self):
        """
        Prépare les sources de données à partir du gestionnaire de données.
        """
        try:
            self.data_manager.load_data()
            self.features_data = self.data_manager.prepare_features()

            # Préparation des données de rendements et NDVI
            yield_data = self.features_data[['parcelle_id', 'date', 'rendement_estime']].dropna()
            ndvi_data = self.features_data[['parcelle_id', 'date', 'ndvi']].dropna()

            # Sources complètes
            self.full_yield_source = ColumnDataSource(yield_data)
            self.full_ndvi_source = ColumnDataSource(ndvi_data)

            # Sources dynamiques (initialement vides)
            self.yield_source = ColumnDataSource(data={key: [] for key in yield_data.columns})
            self.ndvi_source = ColumnDataSource(data={key: [] for key in ndvi_data.columns})

            print("Sources de données préparées avec succès.")
        except Exception as e:
            print(f"Erreur lors de la préparation des sources de données : {e}")

    def create_yield_history_plot(self, select_widget):
        """
        Crée un graphique de l'historique des rendements par parcelle.
        """
        try:
            p = figure(
                title="Historique des Rendements par Parcelle",
                x_axis_type="datetime",
                height=400,
                tools="pan,wheel_zoom,box_zoom,reset,save",
                x_axis_label="Date",
                y_axis_label="Rendement (t/ha)"
            )
            p.line(x='date', y='rendement_estime', source=self.yield_source, line_width=2, color="blue", legend_label="Rendement")
            p.circle(x='date', y='rendement_estime', source=self.yield_source, size=8, color="red", legend_label="Points de Rendement")

            p.add_tools(HoverTool(
                tooltips=[("Date", "@date{%F}"), ("Rendement", "@rendement_estime{0.2f}")],
                formatters={"@date": "datetime"},
                mode="vline"
            ))

            callback = CustomJS(
                args=dict(source=self.yield_source, full_source=self.full_yield_source, select=select_widget),
                code="""
                const full_data = full_source.data;
                const filtered = source.data;
                const selected = select.value;

                // Réinitialiser les données filtrées
                for (let key in filtered) {
                    filtered[key] = [];
                }

                // Filtrer les données par parcelle sélectionnée
                for (let i = 0; i < full_data['parcelle_id'].length; i++) {
                    if (full_data['parcelle_id'][i] === selected) {
                        for (let key in filtered) {
                            filtered[key].push(full_data[key][i]);
                        }
                    }
                }

                source.change.emit();
                """
            )
            select_widget.js_on_change("value", callback)

            return p
        except Exception as e:
            print(f"Erreur lors de la création du graphique de l'historique des rendements : {e}")
            return None

    def create_ndvi_temporal_plot(self, select_widget):
        """
        Crée un graphique montrant l'évolution du NDVI avec des seuils historiques.
        """
        try:
            p = figure(
                title="Évolution du NDVI et Seuils Historiques",
                x_axis_type="datetime",
                height=400,
                tools="pan,wheel_zoom,box_zoom,reset,save",
                x_axis_label="Date",
                y_axis_label="NDVI"
            )
            p.line(
                x='date',
                y='ndvi',
                source=self.ndvi_source,
                line_width=2,
                color="green",
                legend_label="NDVI"
            )
            p.add_tools(HoverTool(
                tooltips=[("Parcelle", "@parcelle_id"), ("Date", "@date{%F}"), ("NDVI", "@ndvi{0.2f}")],
                formatters={"@date": "datetime"},
                mode="vline"
            ))
            p.legend.location = "top_left"

            # Ajouter une ligne de seuil historique
            p.add_layout(Span(location=0.5, dimension='width', line_color='blue', line_dash='dashed', line_width=2))

            callback = CustomJS(
                args=dict(source=self.ndvi_source, full_source=self.full_ndvi_source, select=select_widget),
                code="""
                const full_data = full_source.data;
                const filtered = source.data;
                const selected_parcel = select.value;

                // Réinitialiser les données filtrées
                for (let key in filtered) {
                    filtered[key] = [];
                }

                // Filtrer les données par parcelle sélectionnée
                for (let i = 0; i < full_data['parcelle_id'].length; i++) {
                    if (full_data['parcelle_id'][i] === selected_parcel) {
                        for (let key in filtered) {
                            filtered[key].push(full_data[key][i]);
                        }
                    }
                }

                source.change.emit();
                """
            )
            select_widget.js_on_change("value", callback)

            return p
        except Exception as e:
            print(f"Erreur lors de la création du graphique NDVI : {e}")
            return None

    def create_layout(self):
        """
        Organise les graphiques et widgets dans une mise en page.
        """
        try:
            parcels = self.get_parcelle_options()
            if not parcels:
                print("Aucune parcelle disponible.")
                return None

            select_widget = Select(title="Sélectionner une parcelle:", value=parcels[0], options=parcels)

            yield_plot = self.create_yield_history_plot(select_widget)
            ndvi_plot = self.create_ndvi_temporal_plot(select_widget)

            if not yield_plot or not ndvi_plot:
                print("Un ou plusieurs graphiques n'ont pas pu être créés.")
                return None

            layout = column(select_widget, row(yield_plot, ndvi_plot))
            return layout
        except Exception as e:
            print(f"Erreur lors de la création de la mise en page : {e}")
            return None

    def get_parcelle_options(self):
        """
        Récupère les options de parcelles disponibles.
        """
        try:
            if self.data_manager.monitoring_data is None:
                raise ValueError("Les données de monitoring ne sont pas chargées.")

            parcels = sorted(self.data_manager.monitoring_data["parcelle_id"].unique())
            return parcels
        except Exception as e:
            print(f"Erreur lors de la récupération des options de parcelle : {e}")
            return []

if __name__ == "__main__":
    data_manager = AgriculturalDataManager()
    dashboard = AgriculturalDashboard(data_manager)
    layout = dashboard.create_layout()
    if layout:
        curdoc().add_root(layout)
        show(layout)
    else:
        print("La mise en page n'a pas pu être créée.")
