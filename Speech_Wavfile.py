

# Author : Soumya Chakraborty


# Importing all the necessary libraries
import gensim
import nltk
import speech_recognition as sr
from pydub import AudioSegment
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.decomposition import NMF
from nltk.collocations import *
from textblob import TextBlob
import os
import ast
import moviepy.editor as mp
import argparse
import re
import ffmpeg
from nltk.tokenize.regexp import RegexpTokenizer
from nltk.corpus import stopwords
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel



# Method definition to detect video files, extract the audio out of them and finally extracting the speech using Google Speech API and 
# storing them as text files with the same names. Languages supported are English and German(Deutsch)
def video2speech(testpath):
    rec = sr.Recognizer()
    rec.energy_threshold = 500
    if not os.path.exists(testpath):
        print('ERROR : The path to the videos does not exist !!')
    else:
        for filename in os.listdir(testpath):
            if filename.endswith(('.avi', '.mp4', '.mov', '.mkv', '.flv')):
                newfile = os.path.splitext(filename)[0] + '.wav'
                clip = mp.VideoFileClip(os.path.join(testpath, filename))
                clip.audio.write_audiofile(os.path.join(testpath, newfile))
                audio_file = os.path.join(testpath, newfile)
                audio_seg = AudioSegment.from_wav(audio_file)
                print('\n The contents of the test video are :\n')
                length = len(audio_seg)
                start, end, flag, counter = 0, 0, 0, 1
                interval = 180000
                textfile = os.path.join(testpath, os.path.splitext(audio_file)[0] + '.txt')
                text_out = open(textfile, 'w+')
                for i in range(0, 2 * length, interval):
                    if i == 0:
                        start = 0
                        end = interval
                    else:
                        start = end
                        end = start + interval
                    if end >= length:
                        end = length
                        flag = 1
                    chunk = audio_seg[start:end]
                    filename = os.path.splitext(audio_file)[0] + '(' + str(counter) + ').wav'
                    chunk.export(os.path.join(testpath, filename), format='wav')
                    counter = counter + 1
                    with sr.AudioFile(os.path.join(testpath, filename)) as source:
                        audiofile = rec.listen(source)
                        recorded = rec.recognize_google(audiofile)
                        print(recorded)
                        text_out.write(recorded + ' ')
                    if flag == 1:
                        text_out.close()
                        break



                        
# Method definition to extract tokens out of the speech textfiles. Languages supported are English and German(Deutsch)
def tokenization(testpath):
    for filename in os.listdir(testpath):
        if filename.endswith('.txt'):
            file_content = open(os.path.join(testpath, filename), encoding='utf-8').read()
            b = TextBlob('"' + file_content + '"')
            if b.detect_language() == 'en':
                stop_words = stopwords.words('english')
                word_tokens = simple_preprocess(str(file_content), deacc=True)
                return word_tokens, stop_words
            elif b.detect_language() == 'de':
                stop_words = stopwords.words('german')
                tokenizer = RegexpTokenizer('[a-zA-Z_äöüÄÖÜß]{4,}')
                word_tokens = tokenizer.tokenize(file_content)
                return word_tokens, stop_words
            else:
                print('ERROR : Unfortunately this language is not supported !!')
                
    
 

# Method definition to extract commonly occurring multigrams (bigrams and trigrams) out of the tokens
def multigram(tokens, stop_words, num):
    nostops = [word for word in tokens if word not in stop_words]
    if num==2:
        bigram_measures = nltk.collocations.BigramAssocMeasures()
        finder = BigramCollocationFinder.from_words(nostops)
        finder.apply_freq_filter(3)
        bigrams = [[a, b] for (a, b) in finder.nbest(bigram_measures.pmi, 30)]
        return bigrams
    elif num==3:
        trigram_measures = nltk.collocations.TrigramAssocMeasures()
        finder = TrigramCollocationFinder.from_words(nostops)
        finder.apply_freq_filter(3)
        trigrams = [[a, b, c] for (a, b, c) in finder.nbest(trigram_measures.pmi, 30)]
        return trigrams
    elif num>3:
        print('Unfortunately higher order multigrams are not supported !!')


 

# Method definition to extract the terms for comparison out of some domain (Insurance) related textfiles
def terms_from_file(filepath):
    for filename in os.listdir(filepath):
        if os.path.splitext(filename)[0] == 'Insurance' and filename.endswith('.txt'):
            ins_terms = dict()
            with open(filename) as file:
                for line in file:
                    index = line.find('=')
                    val = line[:index].strip()
                    number = str(line[index + 1:].strip())
                    ins_terms[val] = number
    life_insurance = ins_terms['life_insurance_glossary']
    car_insurance = ins_terms['car_insurance_glossary']
    health_insurance = ins_terms['health_insurance_glossary']
    return life_insurance, car_insurance, health_insurance
 
    
    
# Definition of the LDA Model which analyses the speech-to-text content and finds the closest matching topics
def ldamodel_test(testpath):
    word_tokens, stop_words = tokenization(testpath)
    tokens = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in word_tokens]
    id2word = corpora.Dictionary(tokens)
    corpus = [id2word.doc2bow(text) for text in tokens]
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=5, random_state=10,
                                                iterations=100,
                                                update_every=1, chunksize=50, passes=100, alpha=0.09,
                                                per_word_topics=True, eta=0.8)
    print('\nPerplexity Score: ' + str(lda_model.log_perplexity(corpus)) + '\n')
    for i, topic in lda_model.show_topics(formatted=True, num_topics=5, num_words=12):
        print('TOPIC #' + str(i + 1) + ': ' + topic + '\n')
        res = re.findall('"(.*?)"', topic)
        wordcloud = WordCloud(width=1000, height=500).generate(' '.join(res))
        plt.figure(figsize=(6, 6), facecolor='black')
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.show()
        coherence_model_lda = CoherenceModel(model=lda_model, texts=tokens, dictionary=id2word, coherence='c_v')
        print('\nCoherence Score: ', coherence_model_lda.get_coherence())




# Definition of the Non-negative Factorisation Model which analyses the speech-to-text content and finds the closest matching topics
def nmfmodel_test(testpath, num_topics, num_words):
    word_tokens, stop_words = tokenization(testpath)
    bigrams = multigram(word_tokens, stop_words, 2)
    trigrams = multigram(word_tokens, stop_words, 3)
    tokens = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in word_tokens]
    tokens = [word for word in tokens if word != []]
    multigrams = tokens + bigrams + trigrams
    data_processed = [' '.join(text) for text in multigrams]
    vectorizer = CountVectorizer(analyzer='word', max_features=5000, ngram_range=(1, 3))
    x_counts = vectorizer.fit_transform(data_processed)
    transformer = TfidfTransformer(smooth_idf=False)
    tfidf = transformer.fit_transform(x_counts)

    # Normalizing the Tf-Idf values and applying the NMF Model
    num_topics = int(num_topics)
    tfidf_norm = normalize(tfidf, norm='l1', axis=1)
    nmf_model = NMF(n_components=num_topics, init='nndsvd', max_iter=4000, tol=0.0001, shuffle=False,
                    l1_ratio=0.9, solver='cd')
    nmf_model.fit_transform(tfidf_norm)

    # Obtaining the NMF topics
    features = vectorizer.get_feature_names()
    word_dict = {}
    print('\n The topics generated are as follows : \n')
    for i in range(num_topics):
        words_ids = nmf_model.components_[i].argsort()[:-int(num_words) - 1:-1]
        words = [features[key] for key in words_ids]
        word_dict['Topic # ' + '{:02d}'.format(i + 1)] = words
    for item, value in word_dict.items():
        print(str(item) + ' : ' + str(value) + '\n')
        res = re.findall("'(.*?)'", str(value))
        wordcloud = WordCloud(width=1000, height=500).generate(' '.join(res))
        plt.figure(figsize=(8, 8), facecolor='black')
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.show()




# Definition of the method which analyses the speech-to-text content and compares them with the term documents
# in order to find the closest matching topics which ar predefined. It uses only the counts of matching terms
def insurance_classification(testpath, life_insurance, car_insurance, health_insurance):
    word_tokens, stop_words = tokenization(testpath)
    bigrams = multigram(word_tokens, stop_words, 2)
    trigrams = multigram(word_tokens, stop_words, 3)
    nostops = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in word_tokens]
    nostops = [word for word in nostops if word != []]
    multigrams = nostops + bigrams + trigrams
    multigrams_life = [[i] for i in ast.literal_eval(life_insurance)]
    multigrams_car = [[i] for i in ast.literal_eval(car_insurance)]
    multigrams_health = [[i] for i in ast.literal_eval(health_insurance)]
    life, car, health = 0, 0, 0
    for term in multigrams:
        res = re.findall('[(\w\s+)]', str(term).lower())
        for a, b, c in zip(multigrams_life, multigrams_car, multigrams_health):
            match_life = re.findall('[(\w\s+)]', str(a).lower())
            match_car = re.findall('[(\w\s+)]', str(b).lower())
            match_health = re.findall('[(\w\s+)]', str(c).lower())
            if res == match_life:
                life = life + 1
            elif res == match_car:
                car = car + 1
            elif res == match_health:
                health = health + 1
    if max(life, car, health) == 0:
        print('Unfortunately it does not match any token !!')
    elif life == max(life, car, health):
        print('\nThe primary content is : LIFE INSURANCE\n')
    elif car == max(life, car, health):
        print('\nThe primary content is : AUTO INSURANCE\n')
    elif health == max(life, car, health):
        print('\nThe primary content is : HEALTH INSURANCE\n')



# Definition of the method which analyses the speech-to-text content and compares them with the term documents
# in order to find the closest matching topics which ar predefined. It uses the NLP technique of finding 
# and comparing on the basis of the TF-IDF scores
def insurance_tfidf(testpath, life_insurance, car_insurance, health_insurance):
    word_tokens, stop_words = tokenization(testpath)
    bigrams = multigram(word_tokens, stop_words, 2)
    trigrams = multigram(word_tokens, stop_words, 3)
    nostops = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in word_tokens]
    nostops = [word for word in nostops if word != []]
    multigrams = nostops + bigrams + trigrams
    data_processed = [' '.join(text) for text in multigrams]
    vectorizer = TfidfVectorizer(analyzer='word', max_features=5000, ngram_range=(1, 3), smooth_idf=False)
    X = vectorizer.fit_transform(data_processed)
    match_life = vectorizer.transform(ast.literal_eval(life_insurance))
    match_car = vectorizer.transform(ast.literal_eval(car_insurance))
    match_health = vectorizer.transform(ast.literal_eval(health_insurance))
    life = cosine_similarity(X, match_life).mean().mean()
    print('Cosine Similarity score for Life Insurance glossary : {:.8f}'.format(life))
    car = cosine_similarity(X, match_car).mean().mean()
    print('Cosine Similarity score for Auto Insurance glossary : {:.8f}'.format(car))
    health = cosine_similarity(X, match_health).mean().mean()
    print('Cosine Similarity score for Health Insurance glossary : {:.8f}'.format(health))
    if life == max(life, car, health):
        print('\nThe primary content is : LIFE INSURANCE\n')
    elif car == max(life, car, health):
        print('\nThe primary content is : AUTO INSURANCE\n')
    elif health == max(life, car, health):
        print('\nThe primary content is : HEALTH INSURANCE\n')




# The main() function where the arguments are set and the other methods are called
def main():
    parser = argparse.ArgumentParser(description='Setting the arguments for the Speech to Text model')
    parser.add_argument('-f', '--filepath', required=True, default='',
                        help='The path containing the wave files containing speech content')
    parser.add_argument('-t', '--testpath', required=True, default='',
                        help='The path containing the files for testing our model')
    parser.add_argument('-nt', '--numtopics', required=True, default='',
                        help='The number of topics to be extracted from the text')
    parser.add_argument('-w', '--numwords', required=True, default='',
                        help='The number of words to be displayed for each topic')
    args = parser.parse_args(['-f', r'C:\Xxxxxx\PycharmProjects\Speech_Recognition', '-t',
                              r'C:\Xxxxxx\PycharmProjects\Speech_Recognition\output', '-nt', '3', '-w',
                              '15'])

    video2speech(args.testpath)
    life_insurance, car_insurance, health_insurance = terms_from_file(args.filepath)
    # insurance_classification(args.testpath, life_insurance, car_insurance, health_insurance)
    insurance_tfidf(args.testpath, life_insurance, car_insurance, health_insurance)
    nmfmodel_test(args.testpath, args.numtopics, args.numwords)




if __name__ == '__main__':
    main()
