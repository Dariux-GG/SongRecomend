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

print("🎧 Recomendador de Canciones")

modo = input("¿Cómo quieres buscar? (1: Por canción, 2: Por artista): ").strip()
n = int(input("Número de recomendaciones (1-20): "))

if modo == "1":
    user_input = input("Escribe parte del nombre de la canción: ").strip()
    matches = df[df['song_name'].str.lower().str.contains(user_input.lower())]

    if matches.empty:
        print("No se encontraron canciones.")
    else:
        match_options = matches['song_name'] + " - " + matches['artist']
        for i, option in enumerate(match_options):
            print(f"{i+1}: {option}")
        selected_idx = int(input("Elige una canción por número: ")) - 1

        selected_index = matches.index[selected_idx]
        target_vector = scaled_features[selected_index].reshape(1, -1)
        similarities = cosine_similarity(target_vector, scaled_features)[0]
        sim_scores = list(enumerate(similarities))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = [s for s in sim_scores if s[0] != selected_index]
        top_indices = [i[0] for i in sim_scores[:n]]

        result_df = df.iloc[top_indices][['song_name', 'artist', 'popularity']].copy()
        result_df['similarity'] = [sim_scores[i][1] for i in range(n)]

        print("\nCanciones recomendadas:")
        print(result_df)

elif modo == "2":
    artist_input = input("Escribe parte del nombre del artista: ").strip()
    artist_matches = df[df['artist'].str.lower().str.contains(artist_input.lower())]

    if artist_matches.empty:
        print("No se encontraron artistas.")
    else:
        unique_artists = artist_matches['artist'].drop_duplicates().tolist()
        for i, artist in enumerate(unique_artists):
            print(f"{i+1}: {artist}")
        artist_idx = int(input("Elige un artista por número: ")) - 1
        selected_artist = unique_artists[artist_idx]

        artist_songs = df[df['artist'] == selected_artist]
        for i, song in enumerate(artist_songs['song_name']):
            print(f"{i+1}: {song}")
        song_idx = int(input("Elige una canción por número: ")) - 1

        idx = artist_songs.index[song_idx]
        target_vector = scaled_features[idx].reshape(1, -1)
        similarities = cosine_similarity(target_vector, scaled_features)[0]
        sim_scores = list(enumerate(similarities))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = [s for s in sim_scores if s[0] != idx]
        top_indices = [i[0] for i in sim_scores[:n]]

        result_df = df.iloc[top_indices][['song_name', 'artist', 'popularity']].copy()
        result_df['similarity'] = [sim_scores[i][1] for i in range(n)]

        print("\nCanciones recomendadas:")
        print(result_df)

else:
    print("Opción inválida.")
