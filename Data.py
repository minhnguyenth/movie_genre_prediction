import pandas as pd
import numpy as np
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import PCA

import nltk
#nltk.download('stopwords')



# title processing

def clean_text(text):
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower())

    words = text.split()
    porter = PorterStemmer()
    porter_words = [porter.stem(word) for word in words]

    stop = stopwords.words('english')

    removed_stop_words = [w for w in porter_words if w not in stop]

    return " ".join(removed_stop_words)

def bag_of_word_vectorize(df):

    df["title"] = df["title"].apply(clean_text)


    countvec = CountVectorizer()
    wordvec = countvec.fit_transform(df["title"].ravel())

    tfidf = TfidfTransformer()
    np.set_printoptions(precision=2)
    tfidf_wordvec = tfidf.fit_transform(wordvec).toarray()

    original_title_reduced = pd.DataFrame(tfidf_wordvec)

    data = pd.concat([df, original_title_reduced], axis=1)



# title processing using word to vec
def process_title(data):

    # Word2Vec approach

    vectorizer = CountVectorizer(stop_words='english', decode_error='ignore', analyzer='word', min_df=4, max_features=3000)

    corpus = data['title'].values.astype('U')

    wordvec = vectorizer.fit_transform(corpus.ravel())
    wordvec = wordvec.toarray()
    words = vectorizer.get_feature_names()
    original_title = pd.DataFrame(wordvec, columns=words)

    # Number of vectors for d=60
    d = 60

    # Fit PCA model
    pca2 = PCA(n_components=d)
    pca2.fit(original_title)
    pc = pca2.components_
    original_title_reduced = pca2.transform(original_title)

    pca2.explained_variance_ratio_.sum()

    original_title_reduced = pd.DataFrame(original_title_reduced)

    data = pd.concat([data, original_title_reduced], axis=1)

    return data


# process date

# Convert date to days
def date_to_nth_day(date):
    date = pd.to_datetime(date)
    new_year_day = pd.Timestamp(year=date.year, month=1, day=1)
    return (date - new_year_day).days + 1



#Numericalize the date
#If the date is in invalidd format, pick the date in the middle of a year
def handle_date(df):
    #df['releasedateNum'] = 0
    months = ["January", "February", "March","April", "May", "June", "July", "August", "September", "October", "November", "December"]

    dateTypes = []
    num_days = {}
    for index, row in df.iterrows():
        #remove production country part
        pre_date = row['releasedate']
        if not isinstance(pre_date, float):
            if '(' in pre_date and any(x in pre_date for x in months):
                 cleaned_date = re.sub(r'\(.*?\)', "", pre_date)
                 numericalized_date = date_to_nth_day(cleaned_date)
                 df.loc[index, 'releasedate'] = numericalized_date

            else:
                df.loc[index, 'releasedate'] = 183 #366/2
        else:
            df.loc[index, 'releasedate'] = 183  # 366/2




# numericalize data
def numericalize_data(data_set):
    from sklearn.preprocessing import LabelEncoder

    class_le = LabelEncoder()

    # drop unnecessary features to reduce model's complexity
    data_set = data_set.drop(['country', 'imdb', 'year', 'title'], axis=1)
    data_set.genre = class_le.fit_transform(data_set.genre.values)
    data_set.actor1 = class_le.fit_transform(data_set.actor1.values)
    data_set.actor2 = class_le.fit_transform(data_set.actor2.values)
    data_set.director = class_le.fit_transform(data_set.director.values)
    return data_set


#Getting data
data_set =  pd.read_json("usa_revise.json", lines=True)
handle_date(data_set)
data_set = process_title(data_set)
X = numericalize_data(data_set)

X.to_json("usa_movies_processed.json",lines=True, orient="records")