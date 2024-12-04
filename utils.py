import pandas as pd

def get_similar_users(user_similarity_df, newUser='idNewUser', n_recommend_users = 5):
    similar_users = user_similarity_df.loc[newUser]
    similar_users = similar_users[(similar_users > 0) & (similar_users < 1)].sort_values(ascending=False)

    return similar_users[:n_recommend_users].index.tolist()



def get_most_prefered_movie_by_user(userIdSearhList, movie_user_matrix, n_most_movies = 3):
    most_idMoviesList = []
    for userIdSearh in userIdSearhList:
        try:
            most_idMovies = movie_user_matrix.loc[userIdSearh][movie_user_matrix.loc[userIdSearh] > 0].sort_values(ascending=False)
            most_idMovies = most_idMovies[:n_most_movies].index.tolist()
            most_idMoviesList += most_idMovies
        except KeyError:
            pass

    if len(most_idMoviesList)>0:
        return most_idMoviesList
    else:
        return []


def get_movie_recommendation(movieIdRef, knn, csr_data, movies, movie_user_matrix_recommend, n_movies_to_reccomend=5):
    # Filtrar la película basada en el movieIdRef
    movie_list = movies[movies['movieId'] == movieIdRef] 
    
    if len(movie_list) == 0:
        return pd.DataFrame({'Title': [], 'Genres': [], 'Distance': []})
    
    # Obtener el movieId de la película seleccionada
    movie_idx = movie_list.iloc[0]['movieId']
    
    # Verificar si el movieId está en el movie_user_matrix_recommend
    user_movie_row = movie_user_matrix_recommend[movie_user_matrix_recommend['movieId'] == movie_idx]
    
    if user_movie_row.empty:
        return pd.DataFrame({'Title': [], 'Genres': [], 'Distance': []})
    
    # Obtener el índice de la película en la matriz
    movie_idx = user_movie_row.index[0]
    
    # Obtener las distancias y los índices de las películas recomendadas
    distances, indices = knn.kneighbors(csr_data[movie_idx], n_neighbors=n_movies_to_reccomend + 1)
    
    # Crear una lista de recomendaciones
    rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
    
    recommend_frame = []
    
    for val in rec_movie_indices:
        recommended_movie_id = movie_user_matrix_recommend.iloc[val[0]]['movieId']
        idx = movies[movies['movieId'] == recommended_movie_id].index
        
        if len(idx) == 0:
            continue  # Si no se encuentra el índice, continuar con la siguiente iteración
        
        recommend_frame.append({
            'Title': movies.iloc[idx]['title'].values[0],
            'Genres': movies.iloc[idx]['genres'].values[0],
            'Distance': val[1]
        })
    
    # Crear un DataFrame para las recomendaciones
    df = pd.DataFrame(recommend_frame, index=range(1, len(recommend_frame) + 1))
    
    if df.empty:
        return "No recommendations found."
    
    return df