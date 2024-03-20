from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import requests
from bs4 import BeautifulSoup
import json
import multiprocessing

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def extract_text_and_links_from_website(url):
    # Send a GET request to the URL
    response = requests.get(url)
    # Parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')
    # Extract text from the HTML
    text = soup.get_text()
    # Extract all URLs from the webpage
    links = [link.get('href') for link in soup.find_all('a', href=True)]
    # Filter out links that are pointing to Wikipedia
    wikipedia_links = ['https://en.wikipedia.org' + link for link in links if link.startswith('/wiki/')]
    return text, wikipedia_links


def process_url(title):
    url = f'https://en.wikipedia.org/wiki/{title}'
    website_content, website_links = extract_text_and_links_from_website(url)

    tokens = word_tokenize(website_content.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and word.isalnum()]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    cleaned_content = ' '.join(tokens)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform([cleaned_content])
    feature_names = vectorizer.get_feature_names_out()
    top_keywords_idx = X.toarray().argsort()[0][-10:][::-1]
    top_keywords = [feature_names[idx] for idx in top_keywords_idx]
    print("Processed:", url + "; Top keywords:", top_keywords)
    return url, top_keywords, website_links

if __name__ == '__main__':
    with open('article_titles.txt', 'r') as file:
        article_titles = [next(file).strip() for _ in range(50000)]

    pool = multiprocessing.Pool(processes=4)
    results = pool.map(process_url, article_titles)

    top_keywords_map = {}
    links_map = {}
    url_array = []
    for url, top_keywords, website_links in results:
        top_keywords_map[url] = top_keywords
        links_map[url] = website_links
        url_array.append(url)

    with open('../output/top_keywords.json', 'w') as file:
        json.dump(top_keywords_map, file)

    with open('../output/links.json', 'w') as file:
        json.dump(links_map, file)

    with open('../output/urls.txt', 'w') as file:
        for url in url_array:
            file.write(url + '\n')
