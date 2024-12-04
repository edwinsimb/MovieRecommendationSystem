import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from  utils import get_similar_users, get_most_prefered_movie_by_user, get_movie_recommendation

movies = joblib.load('pkl/movies.pkl')
mainListGenres = joblib.load('pkl/mainListGenres.pkl')
genres_userId_matrix = joblib.load('pkl/genres_userId_matrix.pkl')
user_similarity_df = joblib.load('pkl/user_similarity_df.pkl')
knn = joblib.load('pkl/knn.pkl')
csr_data = joblib.load('pkl/csr_data.pkl')
movie_user_matrix = joblib.load('pkl/movie_user_matrix.pkl')
movie_user_matrix_recommend = joblib.load('pkl/movie_user_matrix_recommend.pkl')

# Título de la aplicación
st.title("Sistema Personalizado para Recomendar Películas")

# Ingresar 10 parámetros numéricos
st.write("Ingresa tu nivel de satisfacción para las siguientes géneros de películas:")

# Crear dos columnas
col1, col2 = st.columns(2)

# Crear un formulario para ingresar los parámetros
params = []

with col1:
    for i in range(0, 5):  # Primeros 5 parámetros en la primera columna
        param = st.slider(f'**Género:** {mainListGenres[i]}', min_value=0.0, max_value=5.0, value=0.0, step=0.01)
        params.append(param)

with col2:
    for i in range(5, 10):  # Últimos 5 parámetros en la segunda columna
        param = st.slider(f'**Género:** {mainListGenres[i]}', min_value=0.0, max_value=5.0, value=0.0, step=0.01)
        params.append(param)

# Agregar una línea fina como separador
st.markdown("---")  # Esta línea crea una línea horizontal

# Botón para crear el DataFrame
if st.button("Crear DataFrame", key="create_dataframe"):
    # Crear un DataFrame a partir de los parámetros ingresados
    data = {'idNewUser': params}
    user_movie_matrix_newUser = pd.DataFrame(data, index=mainListGenres)

    genres_userId_matrix_Predict = pd.concat([user_movie_matrix_newUser, genres_userId_matrix], axis=1)
    userId2Recommend = get_similar_users(user_similarity_df, newUser='idNewUser', n_recommend_users = 10)
    movieId2Recommend = get_most_prefered_movie_by_user(userId2Recommend, movie_user_matrix, n_most_movies = 5)

    final_movie_recommendation = []
    for recommendMovieId in movieId2Recommend:
        final_movie_recommendation.append(get_movie_recommendation(recommendMovieId, knn, csr_data, movies, movie_user_matrix_recommend, n_movies_to_reccomend=3))

    dfRecommendationFinal = pd.concat(final_movie_recommendation, axis=0)
    dfRecommendationFinal = dfRecommendationFinal.drop_duplicates(subset='Title', keep=False)
    dfRecommendationFinal = dfRecommendationFinal.sort_values('Distance', ascending=False).reset_index()
    dfRecommendationFinal = dfRecommendationFinal.drop('index', axis=1)

    # Mostrar el DataFrame resultante
    st.write("Peliculas recomendadas:")
    st.dataframe(dfRecommendationFinal, use_container_width=True)