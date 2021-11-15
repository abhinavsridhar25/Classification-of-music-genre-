# The data consists of 
# 
# - title of song
# - name of musical group 
# - genre of song
# 
# with songs from the `Country` and `Hip-hop` genres. 
# 
# My aim is to use song lyrics to distinguish the genres. I have compiled about 5,000 words relevant to song lyrics. Differences in the frequency of words will help classify songs as `Country` or `Hip-hop`.


# import some packages

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import helper_functions

# change some settings

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 8)
plt.rcParams['figure.figsize'] = (10,8)

# indicate paths to files

import os 
home = os.environ["HOME"]
path_to_data = f"{home}/shared/project/lyrics.csv"
path_to_dictionary = f"{home}/shared/project/words.csv"


lyrics = pd.read_csv(path_to_data)
lyrics


# Loading the data into the table `lyrics`. Note the there are 4820 columns. 

lyrics.columns


# Note that each song has a unique title. Besides `Title`, `Artist` and `Genre`, there are about 5000 columns corresponding to words. The columns indicate the frequency of the words in songs. 


lyrics.loc[lyrics["Title"] == "In Your Eyes",['Title', 'Artist', 'Genre','like','love','the']]


lyrics.drop(columns = ["Title", "Artist", "Genre"]).sum(axis = 1)


stemming = pd.read_csv(path_to_dictionary)
stemming



q1_1 = (sum(stemming['Stem'] == stemming['Word']))/stemming.shape[0]
print(q1_1)


q1_2 = stemming.loc[stemming['Word']=='message'].iloc[0]['Stem']
print(q1_2)


assert q1_2 in "message"


q1_3 = stemming.loc[stemming['Stem']=='singl'].iloc[0]['Word']
print(q1_3)



assert "singl" in q1_3 



stemming["length difference"] = stemming["Word"].str.len() - stemming["Stem"].str.len()
stemming


max_diff = stemming['length difference'].max()
q1_4 = stemming.loc[stemming['length difference']==max_diff].iloc[0]['Word']
print(q1_4)



assert q1_4 in stemming["Word"].values



training_proportion = 0.8

number_songs = len(lyrics)
number_training = int(number_songs * training_proportion)


lyrics_shuffled = lyrics.sample(frac = 1, random_state = 42)

training_set = lyrics_shuffled.iloc[:number_training]
testing_set = lyrics_shuffled.iloc[number_training:]



proportion_country_training = (sum(training_set['Genre']== 'Country'))/ training_set.shape[0]
proportion_country_testing = (sum(testing_set['Genre']== 'Country'))/ testing_set.shape[0]
print(proportion_country_training)
print(proportion_country_testing)


plt.barh(["Testing","Training"], [proportion_country_testing, proportion_country_training])
plt.title("Proportion of Country songs in Training/Testing sets");


# The idea is to use the k-nearest neighbors approach to classification. 

words = ["like", "love"]
unlableled_points = ["Sangria Wine"]
labeled_points = ["In Your Eyes", "Insane In The Brain"]

helper_functions.generate_scatterplot(words, unlableled_points, labeled_points, testing_set, training_set)



words = ["like", "love"]
unlableled_points = ["Sangria Wine"]
labeled_points = ["In Your Eyes", "One Time", "Insane In The Brain"]

helper_functions.generate_scatterplot(words, unlableled_points, labeled_points, testing_set, training_set)



in_your_eyes = training_set.loc[training_set["Title"] == "In Your Eyes",["like","love"]] 
sangria_wine = testing_set.loc[testing_set["Title"] == "Sangria Wine",["like","love"]]

distance = np.sum((sangria_wine.values - in_your_eyes.values)**2)
distance = np.sqrt(distance)

distance



one_time = training_set.loc[training_set["Title"] == "One Time", ["like","love"]] 
sangria_wine = testing_set.loc[testing_set["Title"] == "Sangria Wine", ["like","love"]]

distance = np.sum((sangria_wine.values - one_time.values)**2)
distance = np.sqrt(distance)

distance



def distance_two_songs(row_1, row_2, words):
    coordinates_1 = row_1[words]
    coordinates_2 = row_2[words]

    distance = np.sqrt(np.sum((coordinates_1.values - coordinates_2.values)**2))
    
    return distance



in_your_eyes = training_set.loc[training_set["Title"] == "In Your Eyes",:] 
sangria_wine = testing_set.loc[testing_set["Title"] == "Sangria Wine",:]

q2_1 = distance_two_songs(in_your_eyes, sangria_wine, ["like","love","the"])


assert 0.03 < q2_1 < 0.05


words = ["like", "love", "the"]
row = testing_set.loc[testing_set["Title"] == "Sangria Wine", :]

distance = helper_functions.compute_distances(row, training_set, words)
distance



training_set_with_distance = training_set.copy()
training_set_with_distance["distance"] = distance
training_set_with_distance




# ### Question 4
# 
# We learn from Question 3.2 that the nearest point might have the wrong genre. So we should study many nearby points. We fix an odd number $k$ like $3,5,7,\ldots$. We will calculate the $k$ nearest points in the `training_set` to the song in the `testing_set`. Among the $k$ nearest points will count the number of `Country` and `Hip-hop` songs. 
# 
# - If we have more `Country` than `Hip-hop` then we will clasify the song as `Country`
# - If we have more  `Hip-hop` than `Country` then we will clasify the song as `Hip-hop`
# 
# Note that we use the same number $k$ throughout the classification. We need to evaluate the prediction to determine a choice of $k$.
#

training_set_with_distance_top_15 = training_set_with_distance.sort_values("distance", ascending = True).head(15)
training_set_with_distance_top_15


# We have a mix of `Hip-hop` and `Country` in the `Genre` column. Determine the number of `Hip-hop` songs and `Country` songs.


count_country_nearest_neighbors = sum(training_set_with_distance_top_15['Genre']=='Country')
count_hiphop_nearest_neighbors = sum(training_set_with_distance_top_15['Genre']=='Hip-hop')
print(count_country_nearest_neighbors)
print(count_hiphop_nearest_neighbors)


# 
# Note that we need to determine the most common genre among the nearest songs. Instead of counting the number of `Country` and `Hip-hop`, we can compute the mode meaning the most common value.
#


def compute_mode(column, table):
    return table[column].mode().values[0]


# The function `compute_mode` has input 
# 
# - `column` : string indicating columns of the table 
# - `table` : table containing data
# 
# and output the mode of a column in a table. 
# 
# Using the `compute_mode` on the following table to predict the genre of `Sangria Wine` based on the 31 nearest songs.

training_set_with_distance_top_31 = training_set_with_distance.sort_values("distance", ascending = True).head(31)
training_set_with_distance_top_31

# 
# We have been working with the words `["like", "love", "the"]`. However, we need to choose words that help us to differentiate between genres. Based on common words in the `Country` and `Hip-hop` genres, we will try the words `["street","style","truck","lone"]`. 

words = ["like","love","the","street","style","truck","lone"]


# We want to calculate the accuracy of the predictions to determine the relevance of these words to classifying songs into genres. 
#  
# We can make a copy of the training set. We will add a column with the distance to a song in the testing set.

training_set_with_distance = training_set.copy()

# Here we will take $k=15$. 

k = 15

#k = 31
k = 7


# We need to iterate through the rows of `testing_set` to determine predictions.

predictions = []

# iterate through the rows of testing_set
for idx, row in testing_set.iterrows():
    #compute distance from a song to the songs in training_set
    distance = helper_functions.compute_distances(row, training_set_with_distance, words)
    training_set_with_distance["distance"] = distance
    
    # sort the songs in traing_set by distance
    training_set_with_distance_top_k = training_set_with_distance.sort_values("distance", ascending = True).head(k)
    
    # determine mode 
    prediction = compute_mode("Genre", training_set_with_distance_top_k)
    
    # record the prediction
    predictions.append(prediction)


# Check: How many songs have we classified `Country`? How many songs have we classified `Hip-hop`?

count_country_testing = predictions.count('Country')
count_hiphop_testing = predictions.count('Hip-hop')
print(count_country_testing)
print(count_hiphop_testing)


# 
# We can compute the accuracy of the predictions. Here we need to calculate
# 
# $$\displaystyle \frac{\text{number correct predictions}}{\text{number of predictions}}$$
# 
# We can add `predictions` to a copy of `testing_set`.


testing_set_with_predictions = testing_set.copy()
testing_set_with_predictions["predictions"] = predictions
testing_set_with_predictions


# We can compare the `Genre` column and the `predictions` column.


accuracy = np.sum(testing_set_with_predictions["Genre"] == testing_set_with_predictions["predictions"]) / len(testing_set_with_predictions)
accuracy


# We find that 67\% of the classification are correct. 
# 
# Check : If we used $k=31$ then what is the accurcy of the classifications?

q5_2 = np.sum(testing_set_with_predictions["Genre"] == testing_set_with_predictions["predictions"]) / len(testing_set_with_predictions)
print(q5_2)

# 
# Having computed the accuracy for $k=31$ and $k=15$, let us compute the accuracy for $k=7$. 

result_for_k75 =np.sum(testing_set_with_predictions["Genre"] == testing_set_with_predictions["predictions"]) / len(testing_set_with_predictions)
print(q5_3)



# We can generate a chart to show the accuracy for different choices of $k$. 



plt.plot([7,15,31], [q5_3,0.67,q5_2], "o:")

plt.xticks([7,15,31])
plt.ylabel("Accuracy")
plt.xlabel("Value of k")

plt.title('Classfication from ["like","love","the","street","style","truck","lone"]');


# We learn that $k=7$ give the most accurate classifications. If we experimented with the collection of words, then we would be able to make even more accurate classfications!
