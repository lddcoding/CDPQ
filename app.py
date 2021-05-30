import streamlit as st
from multiapp import MultiApp
from apps import tableau_comparatif, data_stats # import your app modules here

app = MultiApp()

# Add all your application here
app.add_app("Tableau Comparatif",  tableau_comparatif.app)
app.add_app("Graphique Comparatif", data_stats.app)

# The main app
app.run()