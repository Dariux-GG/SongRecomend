import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Cargar datos
df = pd.read_csv("C:/Users/angel/OneDrive/Escritorio/PersonalP/df.csv")

# Columnas a usar para similitud
feature_cols = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
                'instrumentalness', 'liveness', 'valence', 'tempo', 'duration']

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[feature_cols])

st.title("🎧 Recomendador de Canciones")

modo = st.radio("¿Cómo quieres buscar?", ["Buscar por canción", "Buscar por artista"])

n = st.slider("Número de recomendaciones", 1, 20, 10)

if modo == "Buscar por canción":
    user_input = st.text_input("Escribe parte del nombre de la canción")

    if user_input:
        matches = df[df['song_name'].str.lower().str.contains(user_input.lower())]
        if matches.empty:
            st.warning("No se encontraron canciones.")
        else:
            match_options = matches['song_name'] + " - " + matches['artist']
            selected = st.selectbox("Elige una canción:", match_options)

            if st.button("Buscar"):
                selected_index = matches.index[match_options.tolist().index(selected)]
                target_vector = scaled_features[selected_index].reshape(1, -1)
                similarities = cosine_similarity(target_vector, scaled_features)[0]
                sim_scores = list(enumerate(similarities))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                sim_scores = [s for s in sim_scores if s[0] != selected_index]
                top_indices = [i[0] for i in sim_scores[:n]]

                result_df = df.iloc[top_indices][['song_name', 'artist', 'popularity']].copy()
                result_df['similarity'] = [sim_scores[i][1] for i in range(n)]

                st.subheader("Canciones recomendadas:")
                st.dataframe(result_df)

elif modo == "Buscar por artista":
    artist_input = st.text_input("Escribe parte del nombre del artista")

    if artist_input:
        artist_matches = df[df['artist'].str.lower().str.contains(artist_input.lower())]
        if artist_matches.empty:
            st.warning("No se encontraron artistas.")
        else:
            unique_artists = artist_matches['artist'].drop_duplicates()
            selected_artist = st.selectbox("Elige un artista:", unique_artists)

            artist_songs = df[df['artist'] == selected_artist]
            selected_song = st.selectbox("Elige una canción de ese artista:", artist_songs['song_name'])

            if st.button("Buscar"):
                idx = artist_songs[artist_songs['song_name'] == selected_song].index[0]
                target_vector = scaled_features[idx].reshape(1, -1)
                similarities = cosine_similarity(target_vector, scaled_features)[0]
                sim_scores = list(enumerate(similarities))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                sim_scores = [s for s in sim_scores if s[0] != idx]
                top_indices = [i[0] for i in sim_scores[:n]]

                result_df = df.iloc[top_indices][['song_name', 'artist', 'popularity']].copy()
                result_df['similarity'] = [sim_scores[i][1] for i in range(n)]

                st.subheader("Canciones recomendadas:")
                st.dataframe(result_df)
