#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import pandas as pd
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#from sklearn.metrics.pairwise import manhattan_distances
#from sklearn.metrics.pairwise import euclidean_distances

#Читаем Dataset
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')
df = pd.read_csv("movie_dataset.csv")

#Функции чтобы получить из индекса, имя
def get_title_from_index(index):
	return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
	return df[df.title == title]["index"].values[0]


# collaborative filtering

#У нас есть Dataset для рейтинга и Dataset имя фильма соединим их вместе
ratings = pd.merge(movies,ratings).drop(['genres','timestamp'],axis=1)
userRatings = ratings.pivot_table(index=['userId'],columns=['title'],values='rating')
userRatings = userRatings.dropna(thresh=10, axis=1).fillna(0,axis=1)
#метод pearson, modify method of Cosine
corrMatrix = userRatings.corr(method='pearson')

#Делаем рекомендации по похожим рейтингу
def get_similar(movie_name,rating):
    similar_ratings = corrMatrix[movie_name]*(rating-2.5)
    similar_ratings = similar_ratings.sort_values(ascending=False)
    #print(type(similar_ratings))
    return similar_ratings

#Пользователь выбирает фильмы которые мы нравится и rating
romantic_lover = [("(500) Days of Summer (2009)",5),("Alice in Wonderland (2010)",3),("Aliens (1986)",1),
                  ("2001: A Space Odyssey (1968)",2)]
#
similar_movies = pd.DataFrame()
for movie,rating in romantic_lover:
    similar_movies = similar_movies.append(get_similar(movie,rating),ignore_index = True)

similar_movies.head(10)

similar_movies.sum().sort_values(ascending=False).head(20)

#action_lover = [("Amazing Spider-Man, The (2012)",5),("Mission: Impossible III (2006)",4),("Toy Story 3 (2010)",2),("2 Fast 2 Furious (Fast and the Furious 2, The) (2003)",4)]
# similar_movies = pd.DataFrame()
# for movie,rating in action_lover:
#     similar_movies = similar_movies.append(get_similar(movie,rating),ignore_index = True)

# similar_movies.head(10)
# similar_movies.sum().sort_values(ascending=False).head(20)


#content-based filtering

#Из Dataset выбираем нужные нам функции(features)
features = ['keywords','cast','genres','director']

#Где у нас NAN изменяем на пустой string
for feature in features:
	df[feature] = df[feature].fillna('')
#Выставляем все функции в одну строку
def combine_features(row):
	try:
		return row['keywords'] +" "+row['cast']+" "+row["genres"]+" "+row["director"]
	except:
		print ("Error:", row)	
#применяем на все строки
df["combined_features"] = df.apply(combine_features,axis=1)

#матрицу в которой Сказано каждое слово сколько раз была повторена
cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])

#Используя метод cosine для измерения между векторами
cosine_sim = cosine_similarity(count_matrix) 

#Указываем фильм который хочем найти похоже на него фильмы
movie_user_likes = "Aliens"

#Получаем индекс фильма
movie_index = get_index_from_title(movie_user_likes)

#Делаем чтобы  у нас был связан каждый фильм с индексом похожего на него фильма(tupels)
similar_movies =  list(enumerate(cosine_sim[movie_index]))

#Получаем 10 похожий фильмы В правильном порядке
sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)

i=0
for element in sorted_similar_movies:
		print (get_title_from_index(element[0]))
		i=i+1
		if i>10:
			break
        
