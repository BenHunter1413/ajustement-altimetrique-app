import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io # Import io for in-memory file handling

st.set_page_config(page_title="Ajustement Altimétrique", layout="wide")
st.title("📏 Application d'Ajustement Altimétrique")

# Champs d'informations générales
st.sidebar.title("📝 Informations Générales")
nom_projet = st.sidebar.text_input("Nom du projet", value="Mon Projet Altimétrique")
date_scan = st.sidebar.date_input("Date du scan")
ref_altimetrique = st.sidebar.text_input("Référence altimétrique de vérification", value="NGF-IGN69")

def trouver_offset_min_abs_mean(DZ, coarse_step=0.001, fine_step=0.0001):
    """
    Trouve l'offset qui minimise la moyenne des valeurs absolues des différences.
    Utilise une recherche grossière puis une recherche fine pour une meilleure précision.
    """
    offset_range = np.arange(DZ.min(), DZ.max(), coarse_step)
    best_offset = offset_range[np.argmin([np.mean(np.abs(DZ - o)) for o in offset_range])]
    fine_range = np.arange(best_offset - coarse_step, best_offset + coarse_step, fine_step)
    return fine_range[np.argmin([np.mean(np.abs(DZ - o)) for o in fine_range])]

def trouver_offset_max_interval(DZ, interval, coarse_step=0.0005, fine_step=0.00005):
    """
    Trouve l'offset qui maximise le pourcentage de points dans un intervalle donné.
    Utilise une recherche grossière puis une recherche fine pour une meilleure précision.
    """
    offset_range = np.arange(DZ.min() - 0.01, DZ.max() + 0.01, coarse_step)
    best_offset = offset_range[np.argmax([np.mean(np.abs(DZ - o) <= interval) for o in offset_range])]
    fine_range = np.arange(best_offset - coarse_step, best_offset + coarse_step, fine_step)
    return fine_range[np.argmax([np.mean(np.abs(DZ - o) <= interval) for o in fine_range])]

uploaded_file = st.file_uploader("📂 Charger un fichier Excel", type=["xlsx", "xls"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        if "Z1" not in df.columns or "Z2" not in df.columns:
            st.error("Les colonnes 'Z1' et 'Z2' sont requises dans votre fichier Excel.")
        else:
            # Calculate DZ and drop NaNs, then ensure df is filtered accordingly
            DZ = df["Z2"] - df["Z1"]
            DZ = DZ.dropna()
            if DZ.empty:
                st.warning("Aucune donnée valide pour le calcul de DZ après suppression des valeurs manquantes.")
            else:
                st.success("Fichier chargé avec succès. Calcul des ajustements en cours...")
                # Filter the original DataFrame based on the valid DZ indices
                df = df.loc[DZ.index].copy()
                
                resultats = {}
                tableau_comparatif = []

                def appliquer_offset(nom, offset, label):
                    """
                    Applique un offset aux différences d'altitude (DZ) et calcule les statistiques.
                    Stocke les résultats pour affichage et comparaison.
                    """
                    adjusted = DZ - offset
                    stats = {
                        "Méthode": label,
                        "Offset (mm)": round(offset * 1000, 2),
                        "Médiane (mm)": round(np.median(adjusted) * 1000, 2),
                        "Écart-type (mm)": round(np.std(adjusted) * 1000, 2),
                        "Min (mm)": round(np.min(adjusted) * 1000, 2),
                        "Max (mm)": round(np.max(adjusted) * 1000, 2),
                        "-3cm<x<+3cm (%)": f"{round(np.mean(np.abs(adjusted) <= 0.03) * 100, 2)} %",
                        "-2cm<x<+2cm (%)": f"{round(np.mean(np.abs(adjusted) <= 0.02) * 100, 2)} %",
                        "-1.5cm<x<+1.5cm (%)": f"{round(np.mean(np.abs(adjusted) <= 0.015) * 100, 2)} %",
                        "-1cm<x<+1cm (%)": f"{round(np.mean(np.abs(adjusted) <= 0.01) * 100, 2)} %",
                        ">3cm (nombre de points)": int(np.sum(np.abs(adjusted) > 0.03))
                    }
                    interval_data = []
                    for limite in [0.005, 0.01, 0.015, 0.02, 0.03]:
                        percent = round(np.mean(np.abs(adjusted) <= limite) * 100, 2)
                        count = int(np.sum(np.abs(adjusted) <= limite))
                        interval_data.append({
                            "Intervalle": f"-{limite*100:.1f}cm < x < +{limite*100:.1f}cm",
                            "Pourcentage": f"{percent} %",
                            "Nombre de points": count
                        })
                    df_copy = df.copy()
                    df_copy["DZ_ajusté"] = adjusted
                    resultats[nom] = (stats, df_copy, interval_data)
                    tableau_comparatif.append(stats)

                # Apply different offset methods
                appliquer_offset("Sans Ajustement", 0.0, "Sans ajustement")
                appliquer_offset("Moyenne", DZ.mean(), "Moyenne")
                appliquer_offset("Mediane", DZ.median(), "Médiane")
                appliquer_offset("Optimisation composite", trouver_offset_min_abs_mean(DZ), "Optimisation composite")
                appliquer_offset("Max ±1cm", trouver_offset_max_interval(DZ, 0.01), "Max % ±1cm")
                appliquer_offset("Max ±2cm", trouver_offset_max_interval(DZ, 0.02), "Max % ±2cm")

                st.sidebar.subheader("⚙️ Méthode Manuelle")
                offset_manuel_cm = st.sidebar.number_input("Définir un offset manuel (cm)", value=0.0, step=0.1)
                appliquer_offset("Manuelle", offset_manuel_cm / 100, f"Manuelle ({offset_manuel_cm} cm)")

                # Determine the best method based on the percentage of points within ±1cm
                # Ensure the value is converted to float for comparison
                meilleure_methode = max(resultats.items(), 
                                        key=lambda x: float(x[1][0]["-1cm<x<+1cm (%)"].replace('%', '').strip()))[0]
                st.markdown(f"### ⭐ Méthode suggérée : **{meilleure_methode}**")

                st.subheader("📊 Tableau comparatif des méthodes")
                df_comparatif = pd.DataFrame(tableau_comparatif)
                st.dataframe(df_comparatif)

                st.subheader("📈 Histogramme superposé des DZ ajustés (en cm)")
                hist_data = pd.DataFrame()
                for nom, (stats, data, _) in resultats.items():
                    temp = pd.DataFrame({"DZ ajusté (cm)": (data["DZ_ajusté"] * 100).round(2), "Méthode": stats["Méthode"]})
                    hist_data = pd.concat([hist_data, temp])
                bins = [-4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4]
                fig_hist = px.histogram(hist_data, x="DZ ajusté (cm)", color="Méthode", barmode="overlay", 
                                        category_orders={"DZ ajusté (cm)": bins},
                                        title="Distribution des DZ ajustés par méthode")
                fig_hist.update_xaxes(dtick=0.5)
                st.plotly_chart(fig_hist, use_container_width=True)

                onglets = st.tabs([f"🔍 {nom}" for nom in resultats.keys()])
                for i, nom in enumerate(resultats.keys()):
                    with onglets[i]:
                        stats, data, intervals = resultats[nom]
                        st.write("### Statistiques générales")
                        st.json(stats) # Display stats as JSON for better readability
                        st.write("### 📊 Statistiques par intervalle")
                        st.dataframe(pd.DataFrame(intervals))

                # Export to Excel
                if st.button("📄 Exporter le rapport Excel"):
                    with st.spinner("Génération du rapport Excel..."):
                        try:
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                # Write comparative table
                                df_comparatif.to_excel(writer, sheet_name='Tableau Comparatif', index=False)

                                # Write details for each method
                                for nom, (stats, data, intervals) in resultats.items():
                                    sheet_name = stats["Méthode"].replace('/', '_').replace('\\', '_') # Sanitize sheet name
                                    
                                    # Write general statistics
                                    stats_df = pd.DataFrame.from_dict(stats, orient='index', columns=['Valeur'])
                                    stats_df.index.name = 'Statistique'
                                    stats_df.to_excel(writer, sheet_name=sheet_name, startrow=0, startcol=0)

                                    # Write interval statistics
                                    intervals_df = pd.DataFrame(intervals)
                                    intervals_df.to_excel(writer, sheet_name=sheet_name, startrow=len(stats_df) + 2, startcol=0, index=False)

                                    # Write adjusted data
                                    # Ensure 'DZ_ajusté' column is present before writing
                                    if 'DZ_ajusté' in data.columns:
                                        data.to_excel(writer, sheet_name=sheet_name, startrow=len(stats_df) + len(intervals_df) + 4, startcol=0, index=False)
                                    else:
                                        st.warning(f"La colonne 'DZ_ajusté' n'a pas été trouvée pour la méthode '{nom}'. Les données brutes ne seront pas exportées pour cette méthode.")

                            processed_data = output.getvalue()
                            st.success("Rapport Excel généré avec succès ! Cliquez sur le bouton ci-dessous pour le télécharger.")
                            
                            st.download_button(
                                label="📥 Télécharger le rapport Excel",
                                data=processed_data,
                                file_name="rapport_ajustement_altimetrique.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        except Exception as e:
                            st.error(f"Une erreur est survenue lors de la génération du rapport Excel : {str(e)}")
                            st.info("Veuillez vérifier les données ou les dépendances (comme 'xlsxwriter').")

    except Exception as e:
        st.error(f"Une erreur inattendue est survenue lors du traitement du fichier : {str(e)}")
        st.info("Veuillez vérifier que votre fichier Excel contient les colonnes 'Z1' et 'Z2' et est au format correct.")

