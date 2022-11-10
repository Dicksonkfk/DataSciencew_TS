%matplotlib inline

import pandas as pd
import string
import seaborn as sns
import matplotlib.pyplot as plt
import collections
import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

# load dataset
lyrics = pd.read_csv("taylor_swift_lyrics_2006-2020_all.csv")

#inspect the first few rows
## YOUR CODE HERE ##
lyrics.head()

#get info about the DataFrame
## YOUR CODE HERE ##
lyrics.info()

# get a list of all the albums in this collection
## YOUR CODE HERE ##
print(lyrics.album_name.unique())

# this is a function to map the name of the album to the year it was released
def album_release(row):  
    if row['album_name'] == 'Taylor Swift':
        return '2006'
    elif row['album_name'] == 'Fearless (Taylor’s Version)':
        return '2008'
    elif row['album_name'] == 'Speak Now (Deluxe)':
        return '2010'
    elif row['album_name'] == 'Red (Deluxe Edition)':
        return '2012'
    elif row['album_name'] == '1989 (Deluxe)':
        return '2014'
    elif row['album_name'] == 'reputation':
        return '2017'
    elif row['album_name'] == 'Lover':
        return '2019'
    elif row['album_name'] == 'evermore (deluxe version)':
        return '2020'
    #ok, we know folklore was actually released in Dec 2020, but this will make our analysis easier
    elif row['album_name'] == 'folklore (deluxe version)':
        return '2021'
    #this is slightly differently formatted because the album name is recorded two ways.
    elif 'midnights' in row['album_name']:
        return '2022'
    
    return 'No Date'


# apply the function to the album
## YOUR CODE HERE ##
lyrics['album_years'] = lyrics.apply(lambda row: album_release(row), axis= 1 )
# inspect the first few rows of the DataFrame
## YOUR CODE HERE ##
lyrics.head()

#lowercase
## YOUR CODE HERE ##
lyrics['clean_lyric'] = lyrics['lyric'].str.lower()
#remove punctuation
## YOUR CODE HERE ##
lyrics['clean_lyric'] = lyrics['clean_lyric'].str.replace('[^\w\s]', '')
lyrics.head()

#remove stopwords (see the next cell for illustration)
#create a small list of English stop words, feel free to edit this list
stop = ['the', 'a', 'this', 'that', 'to', 'is', 'am', 'was', 'were', 'be', 'being', 'been']


#there are three steps in one here - explained below
#we make a list of words with `.split()`
#then we remove all the words in our list
#then we join the words back together into a string
lyrics['clean_lyric'] = lyrics['clean_lyric'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

#see what `.split()` does
#lyrics['clean_lyric_list'] = lyrics['clean_lyric'].apply(lambda x: x.split())
#print(lyrics.head())

#see what `.join()` does
#lyrics['clean_lyric_list_rejoined'] = lyrics['clean_lyric_list'].apply(lambda x: ' '.join(x))
#print(lyrics.head())

#remove those illustration columns
#lyrics.drop(['clean_lyric_list', 'clean_lyric_list_rejoined'], axis=1, inplace=True)
#print(lyrics.head())

#there are many pre-built lists of stopwords, including one from sklearn.
#Most exclude too many words to be appropriate for song lyric analysis.
#from sklearn.feature_extraction import text
#skl_stop = text.ENGLISH_STOP_WORDS
#print(skl_stop)

#create a new column to reflect if the lyrics contain midnight
## YOUR CODE HERE ##
lyrics['midnight'] = lyrics['clean_lyric'].str.contains('midnight')
sum(lyrics['midnight'])

#night, day, and other time-related words
night = ['night','midnight', 'dawn', 'dusk', 'evening', 'late', 'dark', '1am', '2am', '3am', '4am']
day = ['day', 'morning', 'light', 'sun', 'dawn', 'noon', 'golden', 'bright']
time = ['today', 'tomorrow', 'yesterday']

#create a regular expression string for each list of words
## YOUR CODE HERE ##
night_line = '|'.join(night)
day_line = '|'.join(day)
time_line = '|'.join(time)
#create a new column for each category of words
## YOUR CODE HERE ##
lyrics['night'] = lyrics['clean_lyric'].str.contains(night_line)
lyrics['day'] = lyrics['clean_lyric'].str.contains(day_line)
lyrics['time'] = lyrics['clean_lyric'].str.contains(time_line)
#count the number of times each category of word appears in the lyrics
## YOUR CODE HERE ##
ct1 = sum(lyrics['night'])
ct2 = sum(lyrics['day'])
ct3 = sum(lyrics['time'])
#print the count of each word category
## YOUR CODE HERE ##
print ("night words: ", ct1)
print ("day words: ", ct2)
print ("time words: ", ct3)

#inspect the first few rows
## YOUR CODE HERE ##
lyrics.head()

#create a new dataframe for yearly mentions that groups mentions by year
## YOUR CODE HERE ##
yearly_mentions = lyrics.groupby('album_year').sum().reset_index()
yearly_mentions
#plot the mentions of night over years
## YOUR CODE HERE ##
plt.plot(yearly_mentions['album_year'], yearly_mentions['night'])
plt.title("Taylor Swift Night Mentions")
plt.show()


#reinstate the album name
#read the album_year_name.csv
year_name = pd.read_csv('album_year_name.csv')

#sort both dataframes by year
yearly_mentions.sort_values(by='album_year', ascending=True, inplace=True)
year_name.sort_values(by='album_year', ascending=True, inplace=True)

#add the new column for album name
yearly_mentions['album_name'] = year_name['album_name']

#sort the lyrics by the night column to find the albums with the most night references
## YOUR CODE HERE ##
yearly_mentions.sort_values(by='night', ascending=False)

#sort the lyrics by the day column to find the albums with the most day references
## YOUR CODE HERE ##
yearly_mentions.sort_values(by='day', ascending=False)


#add the new column for album name
yearly_mentions['album_name'] = year_name['album_name']
plt.plot(yearly_mentions['album_year'], yearly_mentions['night'], label = "Taylor Swift 'Night' Counts")
plt.plot(yearly_mentions['album_year'], yearly_mentions['day'], label = "Taylor Swift 'Day' Counts")
plt.title('Taylor Swift Day vs Night Mentions')
plt.legend()
plt.show()

#create a position variable that includes both the track number and line number
## YOUR CODE HERE ##
lyrics['position'] = lyrics['track_n'] + (lyrics['line']/1000)
#create a new DataFrame that is grouped by position
## YOUR CODE HERE ##
positional_mentions = lyrics.groupby('position').sum().reset_index()

#increase the size of the plot 
fig = plt.gcf()
fig.set_size_inches(25,10)

#create a plot with two lines to show frequency of day vs. night references by position in the album
## YOUR CODE HERE ##
plt.plot(positional_mentions['position'], positional_mentions['night'], label = "Taylor Swift 'Night' Counts")
plt.plot(positional_mentions['position'], positional_mentions['day'], label = "Taylor Swift 'Day' Counts")
plt.legend()
plt.title('Taylor Swift Day vs Night Mentions by Album Position')
plt.show()

#run this cell to tokenize the words in the clean_lyric column
lyrics['lyrics_tok'] = lyrics['clean_lyric'].str.split(' ')

#inspect the first few lines
## YOUR CODE HERE ##
lyrics.head()

#determine what words overall are the most frequently used words
#create a list of all the words in the lyrics_tok column
word_list = [word for list_ in lyrics['lyrics_tok'] for word in list_]

#use the counter function to count the number of times each word appears
## YOUR CODE HERE ##
hz= collections.Counter(word_list)
#sort the word frequencies to find out the most common words she's used. 
## YOUR CODE HERE ##
hz= sorted(hz.items(), key=lambda x: x[1], reverse=True)
#call the word frequency
## YOUR CODE HERE ##
hz

#run this cell to add a package from NLTK for our sentiment analyzer.
nltk.download('vader_lexicon')

#run this cell to see how the sentiment analyzer works
sia = SentimentIntensityAnalyzer()
sia.polarity_scores("I love Taylor Swift!")

#create a new column called polarity and apply the sia method to the clean_lyric column with a lambda expression
## YOUR CODE HERE ##
lyrics['polarity'] = lyrics['clean_lyric'].apply(lambda x: sia.polarity_scores(x))
lyrics.head()

#run this cell to transform the polarity dictionary into columns of the DataFrame
lyrics[['neg', 'neu', 'pos', 'compound']] = lyrics['polarity'].apply(pd.Series)
lyrics.drop('polarity', axis=1)

#inspect the first few rows
## YOUR CODE HERE ##
lyrics.head()

#calculate overall sentiment for pos, neg, sentiment
## YOUR CODE HERE ##
pos = sum(lyrics['pos'])
neg = sum(lyrics['neg'])
stm = sum(lyrics['sentiment'])
#print the overall sentiments
## YOUR CODE HERE ##
print ("positive count: ", pos)
print ("positive count: ", neg)
print ("positive count: ", sentiment)

#create a new DataFrame using the groupby method for the album_year
## YOUR CODE HERE ##
yearly_sentiment = lyrics.groupby('album_years').sum.reset_index()
#visualize sentiment over time 
## YOUR CODE HERE ##
plt.plot(yearly_sentiment['album_year'], yearly_sentiment['sentiment'])
plt.title(" Taylor Swift's Average album Sentiment")
plt.show()

#create a DataFrame filtered for only night mentions
## YOUR CODE HERE ##
night = lyrics[lyrics['night'] == True]
#create a DataFrame filtered for only day mentions
## YOUR CODE HERE ##
day = lyrics[lyrics['day'] == True]
#print the length of the night and day DataFrames
## YOUR CODE HERE ##
print("night: ", len(night))
print("day: ", len(day))

#calculate the sentiment of each day and night DataFrame from the compound values
## YOUR CODE HERE ##
night_cpd = night['sentiment'].sum()
day_cpd = day['sentiment'].sum()
#print the results
## YOUR CODE HERE ##
print("night sentiment: ", night_cpd)
print("day sentiment: ", day_cpd)
