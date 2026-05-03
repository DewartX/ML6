import re
import nltk
import pandas as pd

from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

tqdm.pandas()

# Для правильной работы NLTK
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

#Набор стоп слов
stop_words = set(stopwords.words('english'))

# Функции
def preprocess_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)	
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    text = word_tokenize(text)
    text = [word for word in text if word not in stop_words]

    #Лемматизация
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text]

    #Повторим очистку от стоп-слов
    text = [word for word in text if word not in stop_words]
    return text
#____________________________________________________________________________________________
# Работа с базой данных
data = pd.read_csv('reviews.csv')
data['label'] = data['sentiment'].progress_apply(lambda label: 1 if label == 'positive' else 0)

# Вызов функции:
data['processed'] = data['review'].progress_apply(preprocess_text)

#Пример отзыва
print("\nНеобработанный отзыв:")
print(data['review'].iloc[0])
print('_'*50)
print("\nОбработанный отзыв (первые 30 слов):")
print(data['processed'].iloc[0][:30])

data[['processed', 'label']].to_csv('reviews_preprocessed.csv', index=False, header=True)