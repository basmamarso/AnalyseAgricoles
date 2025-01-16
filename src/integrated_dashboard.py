from dashboard import AgriculturalDashboard
from map_visualization1 import AgriculturalMap
import streamlit as st

class IntegratedDashboard:
    def __init__(self, data_manager):
        """
        Initialize the Integrated Dashboard.
        Combines Bokeh and Folium visualizations.
        """
        self.data_manager = data_manager
        self.bokeh_dashboard = AgriculturalDashboard(data_manager)
        self.map_view = AgriculturalMap(data_manager)

    def initialize_visualizations(self):
        """
        Initialize all visual components (Bokeh and Folium).
        """
        try:
            # Load data using the data manager
            self.data_manager.load_data()

            # Initialize Bokeh layout
            self.bokeh_dashboard.create_data_sources()
            self.bokeh_layout = self.bokeh_dashboard.create_layout()

            # Initialize Folium map
            self.map_view.create_base_map()
            self.map_view.add_yield_history_layer()
            self.map_view.add_risk_heatmap()

            print("Visualizations initialized successfully.")
        except Exception as e:
            print(f"Error initializing visualizations: {e}")

    def create_streamlit_dashboard(self):
        """
        Create a Streamlit interface integrating all visualizations.
        """
        try:
            st.title("Tableau de Bord Agricole Intégré")

            # Sidebar for parcel selection
            st.sidebar.title("Options de Visualisation")
            parcels = self.bokeh_dashboard.get_parcelle_options()
            selected_parcelle = st.sidebar.selectbox("Sélectionnez une parcelle :", parcels)

            # Update visualizations for selected parcel
            self.update_visualizations(selected_parcelle)

            # Display Bokeh visualizations
            st.header("Visualisations Bokeh")
            if self.bokeh_layout:
                st.bokeh_chart(self.bokeh_layout, use_container_width=True)
            else:
                st.warning("Bokeh layout could not be generated.")

            # Display Folium map
            st.header("Carte Interactive (Folium)")
            if self.map_view.map:
                map_file = "integrated_map.html"
                self.map_view.map.save(map_file)
                with open(map_file, "r") as file:
                    html_content = file.read()
                st.markdown(
                    f'<iframe srcdoc="{html_content}" width="100%" height="600px" style="border:none;"></iframe>',
                    unsafe_allow_html=True,
                )
            else:
                st.warning("Folium map could not be generated.")

            print("Streamlit dashboard created successfully.")
        except Exception as e:
            st.error(f"Erreur lors de la création du tableau de bord : {e}")
            print(f"Error creating Streamlit dashboard: {e}")

    def update_visualizations(self, parcelle_id):
        """
        Update all visualizations for a given parcel.
        """
        try:
            # Update Bokeh plots
            self.bokeh_dashboard.update_plots(parcelle_id)

            # Update Folium map layers
            self.map_view.create_base_map()
            self.map_view.add_yield_history_layer()
            self.map_view.add_risk_heatmap()

            print(f"Visualizations updated for parcelle_id: {parcelle_id}")
        except Exception as e:
            print(f"Error updating visualizations: {e}")


if __name__ == "__main__":
    # Initialize data manager
    from data_manager import AgriculturalDataManager
    data_manager = AgriculturalDataManager()

    # Create and run the dashboard
    dashboard = IntegratedDashboard(data_manager)
    dashboard.create_streamlit_dashboard()
