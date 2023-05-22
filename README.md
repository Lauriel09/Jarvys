
import tensorflow as tf 

import nltk 

import sklearn 

import pandas as pd 

import numpy as np 

import cv2 

import spacy 

from textblob import TextBlob 

import time 

import logging 

from logging.handlers import RotatingFileHandler

from sklearn.feature_extraction.text import CountVectorizer

from google.cloud import speech_v1p1beta1 as speech

from tensorflow.keras.layers import Dense, Dropout, LSTM

from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

nltk.download('stopwords')

nlp = spacy.load('pt_core_news_sm') 

logger = logging.getLogger('chatbot_logger') 

logger.setLevel(logging.DEBUG) 

handler = RotatingFileHandler('chatbot.log', maxBytes=2000, backupCount=5) 

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s') 

handler.setFormatter(formatter) 

logger.addHandler(handler) 

def tokenizar(frase): 

    doc = nlp(frase) 

    tokens = [token.lemma_ for token in doc if not token.is_stop] 

    return tokens 

def preprocessar_dados(dados): 

    vetorizador = CountVectorizer(analyzer=tokenizar, ngram_range=(1, 2)) 

    bag_of_words = vetorizador.fit_transform(dados['Padrao']) 

    classes = np.array(pd.get_dummies(dados['Classe'])) 

    return bag_of_words.toarray(), classes, vetorizador.get_feature_names(), vetorizador

def treinar_modelo_dpl(x_train, y_train): 

    modelo = tf.keras.Sequential([ 

        Dense(256, input_shape=(x_train.shape[1],), activation='relu'), 

        Dropout(0.5), 

        Dense(128, activation='relu'), 

        Dropout(0.5), 

        Dense(64, activation='relu'), 

        Dropout(0.5), 

        Dense(y_train.shape[1], activation='softmax') 

    ]) 

    modelo.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.001), 

                   metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), 'accuracy', 'mse', tf.keras.metrics.AUC()]) 

    tb_callback = TensorBoard(log_dir='logs') 

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, restore_best_weights=True) 

    modelo.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.1, callbacks=[tb_callback, es]) 

    return modelo 

def stem(frase): 

    stemmer = nltk.stem.RSLPStemmer() 

    tokens = tokenizar(frase) 

    palavras = [stemmer.stem(palavra.lower()) for palavra in tokens] 

    return palavras 

def bag_of_words(frase, vetorizador): 

    tokens = stem(frase) 

    bag = vetorizador.transform([frase]).toarray() 

    return bag 

def carregar_modelo(): 

    modelo = tf.keras.models.load_model('modelo.h5') 

    vetorizador = CountVectorizer(analyzer=tokenizar, ngram_range=(1, 2)) 

    palavras = np.load('palavras.npy', allow_pickle=True) 

    classes = np.load('classes.npy', allow_pickle=True) 

    return modelo, palavras, classes, vetorizador 

def salvar_modelo(modelo, palavras, classes, vetorizador): 

    modelo.save('modelo.h5') 

    np.save('palavras.npy', palavras) 

    np.save('classes.npy', classes) 

    vetorizador.save('vetorizador.joblib') 

    logger.info("Modelo treinado e salvo com sucesso.") 

def treinar_e_salvar_modelo(dados): 

    x_train, y_train, palavras, vetorizador = preprocessar_dados(dados) 

    modelo = treinar_modelo_dpl(x_train, y_train) 

    salvar_modelo(modelo, palavras, y_train, vetorizador) 

def carregar_dados(arquivo): 

    dados = pd.read_csv(arquivo, names=['Padrao', 'Classe']) 

    return dados 

def reconhecer_comandos(lista_comandos, frase): 

    tokens = nltk.word_tokenize(frase.lower()) 

    comando_reconhecido = next((comando for comando in lista_comandos if comando in tokens), None) 

    return comando_reconhecido 

def analisar_sentimento(frase): 

    tb = TextBlob(frase) 

    sentimento = tb.sentiment.polarity 

    if sentimento > 0: 

        print("O usuário está feliz.")

    elif sentimento < 0: 

        print("O usuário está chateado.") 

    else: 

        print("O usuário está neutro.") 

def gerar_resposta(pergunta, modelo, palavras, vetorizador): 

    X = bag_of_words(pergunta, vetorizador) 

    resultado = modelo.predict(np.array(X))[0] 

    threshold = 0.7 

    respostas = [[i,r] for i,r in enumerate(resultado) if r>threshold] 

    respostas.sort(key=lambda x: x[1], reverse=True) 

    return_list = [] 

    for r in respostas: 

        return_list.append({"resposta": classes[r[0]], "probabilidade": str(r[1])}) 

    return return_list 

def transcrever_audio(): 

    client = speech.SpeechClient() 

    sampling_rate_hertz = 16000 

    language_code = "pt-BR" 

    config = speech.RecognitionConfig( 

        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16, 

        sample_rate_hertz=sampling_rate_hertz, 

        language_code=language_code, 

    ) 

    streaming_config = speech.StreamingRecognitionConfig(config=config, interim_results=True) 

    with microfone() as fonte: 

        audio_generator = fonte.record (duration = 9) 

        requests = (speech.StreamingRecognizeRequest(audio_content=content) for content in audio_generator) 

        responses = client.streaming_recognize(streaming_config=streaming_config, requests=requests) 

        for resposta in responses: 

            for resultado in resposta.results: 

                texto = resultado.alternatives[0].transcript 

                print("Texto reconhecido: {}".format(texto)) 

                return texto

if __name__ == "__main__":

    arquivo_dados = 'dados.csv'

    lista_comandos = ['sair', 'parar', 'cancelar'] 

    dados = carregar_dados(arquivo_dados) 

    treinar_e_salvar_modelo(dados) 

    modelo, palavras, classes, vetorizador = carregar_modelo() 

    while True: 

        texto = transcrever_audio() 

        comando = reconhecer_comandos(lista_comandos, texto) 

        if comando: 

            print("Comando reconhecido: {}".format(comando)) 

            break 

        resposta = gerar_resposta(texto, modelo, palavras, vetorizador) 

        print("Respostas: ") 

        for r in resposta: 

            print("- " + r['resposta'] + " (Probabilidade: " + r['probabilidade'] + ")")

```
