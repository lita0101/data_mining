# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 14:13:15 2023

@author: USER
"""

import streamlit as st
from function import load_data1
from io import StringIO
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#fungsi untuk melakukan input data dan membangun dataframe dari data yang sudah diinputkan
def input():
    p_sepal = st.sidebar.number_input('Maksimal panjang sepal: ')
    panjang_sepal = st.sidebar.slider('panjang sepal', 0.0, 7.90, p_sepal, 0.1)
    if p_sepal > 7.90 or p_sepal < 0.0:
        st.sidebar.warning("Inputan tidak valid, panjang sepal berkisar antara 0 dan 7.90")
        panjang_sepal = 0.0
        # panjang_sepal = st.sidebar.slider('panjang sepal', 0.0, 7.90, p_sepal, 0.1)
    l_sepal = st.sidebar.number_input('Maksimal lebar sepal: ')
    lebar_sepal = st.sidebar.slider('lebar sepal', 0.0, 4.40, l_sepal, 0.1)
    if l_sepal > 4.40 or l_sepal < 0.0:
        st.sidebar.warning("Inputan tidak valid, lebar sepal berkisar antara 0 dan 4.40")
        lebar_sepal = 0.0
        # lebar_sepal = st.sidebar.slider('lebar sepal', 0.0, 4.40, l_sepal, 0.1)
    p_petal = st.sidebar.number_input('Maksimal panjang petal: ')
    panjang_petal = st.sidebar.slider('panjang petal', 0.0, 6.90, p_petal, 0.1)
    if p_petal > 6.90 or p_petal < 0.0:
        st.sidebar.warning("Inputan tidak valid, panjang petal berkisar antara 0 dan 6.90")
        panjang_petal = 0.0
        # panjang_petal = st.sidebar.slider('panjang petal', 0.0, 6.90, p_petal, 0.1)
    l_petal = st.sidebar.number_input('Maksimal lebar petal: ')
    lebar_petal = st.sidebar.slider('lebar petal', 0.0, 2.50, l_petal, 0.1)
    if l_petal > 2.50 or l_petal < 0.0:
        st.sidebar.warning("Inputan tidak valid, lebar petal berkisar antara 0 dan 2.50")
        lebar_petal = 0.0
        # lebar_petal = st.sidebar.slider('lebar petal', 0.0, 2.50, l_petal, 0.1)
    data = {'panjang sepal': panjang_sepal,
            'lebar sepal': lebar_sepal,
            'panjang petal': panjang_petal,
            'lebar petal': lebar_petal
            }
    fitur = pd.DataFrame(data, index=[0])
    return fitur

def app():
    st.title("Prediksi Bunga Iris")
    #pilihan untuk user
    option = st.radio("Select Option", ("Prediksi Jenis", "Upload File", "Input Form"))
    if option == "Prediksi Jenis":
        df = input() #memanggil fungsi input untuk user melakukan input data
        st.subheader('Inputan user')
        table = df.to_html(index=False)
        st.write(table, unsafe_allow_html=True)
        st.text("")
        st.text("")
        # knn = pickle.load(open('trainmodel.sav', 'rb'))
        # pred_data = knn.predict(np.array(df).reshape(1, -1))
        
        if st.button("Prediksi"):
            if df.values[0][0] == 0.0 and df.values[0][1] == 0.0 and df.values[0][2] == 0.0 and df.values[0][3] == 0.0:
                st.warning("Belum input data")
            else:
                    data = load_data1()
                    # if df.values == 0:
                    #     st.warning("Belum input data")
                    
                    # #membagi menjadi fitur dan target
                    X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
                    # data["Label"] = data["Label"].replace({"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2})
                    y = data['Label']
                    
                    # st.dataframe(X)
                    # st.dataframe(y)
                    
                    # X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
                    
                    # st.dataframe(X)
                    
                    #membagi menjadi data test dan data train dengan sebaran sara 80% train dan 20% test
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=4
                    )
                    
                    # k = 4
                    #Train Model and Predict
                    #membangun model KNN demgan menggunakan K=4
                    knn = KNeighborsClassifier(n_neighbors = 4)
                    #melatih algoritma KNN yang sudah dibangun
                    # knn = pickle.load(open('trainmodel.sav', 'rb'))
                    knn.fit(X_train,y_train)
                    #melakukan prediksi ke data inputan user sehingga dapat ditebak jenisnya
                    yhat1 = knn.predict(df.values)
                
                    #untuk memunculkan hasil prediksi
                    st.subheader('prediksi (hasil klasifikasi)')
                    if yhat1 == "Iris-setosa":
                        st.success("Iris-setosa")
                    if yhat1 == "Iris-versicolor":
                        st.success("Iris-versicolor")
                    if yhat1 == "Iris-virginica":
                        st.success("Iris-virginica")
                    # st.text(yhat1)
    
    if option == "Upload File":
        # st.write("nilai data test: ")
        text = st.number_input("Masukkan data test")
        text = text * 0.01;
        #melihat data yang harus diinputkan user
        st.markdown("Perhatikan struktur **:green[Dataset]** yang diinputkan seperti data **dibawah**")
        data = load_data1()
        st.dataframe(data.sample(5))
        st.text("")
        st.text("")
        st.text("")
        
        # if st.button('Predict'):
        #     knn = pickle.load(open('trainmodel.sav', 'rb'))
        #     pred_data = knn.predict(df.values)
        #     st.subheader('prediksi (hasil klasifikasi)')
        #     st.write(pred_data)
        
        #fungsi untuk melakukan load file
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            # To read file as bytes:
            #mengambil value dari dataset yang sudah diinputkan user
            # bytes_data = uploaded_file.getvalue()
                # st.write(bytes_data)
            
                # To convert to a string based IO:
            
            # stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                # st.write(stringio)
            
                # To read file as string:
                # string_data = stringio.read()
                # st.write(string_data)
            
                # Can be used wherever a "file-like" object is accepted:
            
            #melakukan pembacaan data yang sudah diinputkan oleh user
            dataframe = pd.read_csv(uploaded_file)
            st.title("Data Anda: ")
            #memunculkan data yang sudah diinputkan oleh user
            df = pd.DataFrame(dataframe)
            st.dataframe(df)
            st.text("")
            #mengganti label data menjadi numerik
            data["Label"] = data["Label"].replace({"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2})
            
            #membagi data menjadi data fitur dan data target
            #data fitur
            X = dataframe[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values 
            #data target
            y = dataframe['Label'].values
    
            #melakukan scaling data pada data fitur agar persebaran data sama
            # X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=text, random_state=4
            )
            
            st.title("Data train")
            df1 = pd.DataFrame(X_train)
            df1.columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
            df1.index = range(1, len(df1) + 1)
            st.dataframe(df1)
            
            st.title("Data test")
            df2 = pd.DataFrame(X_test)
            df2.columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
            df2.index = range(1, len(df2) + 1)
            st.dataframe(df2)
            
            X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=4
            )
            
            #load model knn yang sudah di save
            knn = pickle.load(open('trainmodel.sav', 'rb'))
            #melakukan prediksi data menggunakan model knn yang sudah di load
            pred_data = knn.predict(X)
            #mengitung akurasi pelatihan knn dengan membandingkan nilai y (target sesungguhnya/aktual) dengan nilai yang hasil prediksi mesin
            akurasi = accuracy_score(y, pred_data)
            st.text("Akurasi model KNN dengan data anda: " + str(akurasi*100) + "%")
            # st.text("Accuracy from your data: "+str(treeAccuracy*100)+"%")
                
            st.text("")
            st.text("")
            #menghitung/membuat confusion matix
            st.header("Confusion Matrix dari Perdiksi Data")
            #membuat confusion matrix hasil perbandingan antara nilai y (target sesungguhnya/aktual) dengan nilai yang hasil prediksi mesin
            cm = confusion_matrix(y,pred_data)
            species_labels = ["Iris-Virginica", "Iris-Setosa", "Iris-Versicolor"]
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=species_labels)
            fig, ax = plt.subplots(figsize=(8, 6))
            disp.plot(ax=ax, cmap="Blues", xticks_rotation="horizontal", colorbar=False)
            ax.set_xlabel("Predicted Labels")
            ax.set_ylabel("True Labels")
            st.pyplot(fig)
            
            # plt.figure(figsize=(8,6))
            # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            # plt.xlabel("Predicted Labels")
            # plt.ylabel("True Labels")
            # #memunculkan confusion matrix
            # st.pyplot(plt.gcf())  # Use plt.gcf() to get the current figure
            
            if st.checkbox("Edit Parameter KNN"):
                #input nilai K
                K = st.number_input('Masukkan Nilai K', value=1)
                
                dataframe["Label"] = dataframe["Label"].replace({"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2})
                
                X = dataframe[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values 
                y = dataframe['Label'].values

                X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=4
                )
                
                # st.title("Data train")
                # df1 = pd.DataFrame(X_train)
                
                # st.title("Data test")
                # df2 = pd.DataFrame(X_test)
                
                #Train Model and Predict
                #membangun model KNN, melakukan prediksi
                knn = KNeighborsClassifier(n_neighbors = K)
                knn.fit(X_train,y_train)
                yhat1 = knn.predict(X_test)
                
                #menghitung akurasi KNN
                akurasi = accuracy_score(y_test, yhat1)
                st.text("Akurasi model KNN dengan K = "+ str(K) +" menggunakan data anda: " + str(akurasi*100) + "%")
                
                st.text("")
                st.text("")
                #membuat confusion matrix
                st.header("Confusion Matrix dari Perdiksi Data")
                cm = confusion_matrix(y_test,yhat1)
                # plt.figure(figsize=(8,6))
                # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                # plt.xlabel("Predicted Labels")
                # plt.ylabel("True Labels")
                # st.pyplot(plt.gcf())  # Use plt.gcf() to get the current figure
                species_labels = ["Iris-Virginica", "Iris-Setosa", "Iris-Versicolor"]
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=species_labels)
                fig, ax = plt.subplots(figsize=(8, 6))
                disp.plot(ax=ax, cmap="Blues", xticks_rotation="horizontal", colorbar=False)
                ax.set_xlabel("Predicted Labels")
                ax.set_ylabel("True Labels")
                st.pyplot(fig)
                    
    if option == "Input Form":
        col1, col2 = st.columns(2)
        #memunculkan banyak data yang ingin digunakan
        num_rows = st.number_input("Banyak Data", min_value=1, value=1, step=1)
        data = []
            
        #program untuk input data
        for i in range(num_rows):
            st.header(f"Input Bunga Iris (Data {i+1})")
            row = {}
            
            row['SepalLengthCm'] = st.number_input('Sepal Length Cm', min_value=0.0, step=0.1, key=f"SepalLengthCm_{i}")
            
            row['SepalWidthCm'] = st.number_input('Sepal Width Cm', min_value=0.0, step=0.1, key=f"SepalWidthCm_{i}")
            
            row['PetalLengthCm'] = st.number_input('Petal Length Cm', min_value=0.0, step=0.1, key=f"PetalLengthCm_{i}")
            
            row['PetalWidthCm'] = st.number_input('Petal Width Cm', min_value=0.0, step=0.1, key=f"PetalWidthCm_{i}")
            
            # row['Label'] = st.selectbox('Label',["Iris-setosa", "Iris-versicolor", "Iris-virginica"], key=f"Label_{i}")
            
            st.text("")
            st.text("")
            st.text("")
            st.text("")
            st.text("")
            st.text("")
            data.append(row)
        
        #membuat/menyatukan data yang diinputkan menjadi sebuah dataframe/dataset
        dataframe = pd.DataFrame(data)
        st.title("Data Anda: ")
        #memunculkan data yang sudah diinputkan
        df = pd.DataFrame(dataframe)
        # st.dataframe(df)
        table = df.to_html(index=False)
        st.write(table, unsafe_allow_html=True)
        # data['Label'] = data['Label'].replace({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
        
        data = load_data1()
        # if df.values == 0:
        #     st.warning("Belum input data")
        
        # #membagi menjadi fitur dan target
        X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
        # data["Label"] = data["Label"].replace({"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2})
        y = data['Label']
        
        # st.dataframe(X)
        # st.dataframe(y)
        
        # X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
        
        # st.dataframe(X)
        
        #membagi menjadi data test dan data train dengan sebaran sara 80% train dan 20% test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=4
        )
        
        # st.title("Data train")
        # df1 = pd.DataFrame(X_train)
        
        # st.title("Data test")
        # df2 = pd.DataFrame(X_test)
        # k = 4
        #Train Model and Predict
        #membangun model KNN demgan menggunakan K=4
        knn = KNeighborsClassifier(n_neighbors = 4)
        #melatih algoritma KNN yang sudah dibangun
        # knn = pickle.load(open('trainmodel.sav', 'rb'))
        knn.fit(X_train,y_train)
        #melakukan prediksi ke data inputan user sehingga dapat ditebak jenisnya
        yhat1 = knn.predict(df.values)
    
        #untuk memunculkan hasil prediksi
        st.subheader('prediksi (hasil klasifikasi)')
        st.write(yhat1)
        # if yhat1 == "Iris-setosa":
        #     st.success("Iris-setosa")
        # if yhat1 == "Iris-versicolor":
        #     st.success("Iris-versicolor")
        # if yhat1 == "Iris-virginica":
        #     st.success("Iris-virginica")
        # #membagi antara data fitur dan data target pada inputan user
        # X = dataframe[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
        # dataframe["Label"] = dataframe["Label"].replace({"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2})
        # y = dataframe['Label'].values
        
        # #melakukan scaling/normalisasi dataset
        # X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
        
        # #membagi menjadi data test dan data train dengan rasio 80:20 (80% data train, 20% data test) dengan skala keacakan 4
        # X_train, X_test, y_train, y_test = train_test_split(
        #     X, y, test_size=0.2, random_state=4
        # )
                
        # k = 4
        # #Train Model and Predict
        # knn1 = KNeighborsClassifier(n_neighbors = k)
        # knn1.fit(X_train,y_train)
        # yhat1 = knn1.predict(X_test)
        #melakukan prediksi pada data test
        # y_pred1 = knn1.predict(X_test)
        
        # cm = confusion_matrix(y_test, y_pred1)
        
        # #membuat plot dan memunculkan plot confusion matrix
        # species_labels = ["Iris-Virginica", "Iris-Setosa", "Iris-Versicolor"]
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=species_labels)
        # fig, ax = plt.subplots(figsize=(8, 6))
        # disp.plot(ax=ax, cmap="Blues", xticks_rotation="horizontal", colorbar=False)
        # ax.set_xlabel("Predicted Labels")
        # ax.set_ylabel("True Labels")
        # st.pyplot(fig)

        # # plt.figure(figsize=(7, 5))
        # # sns.heatmap(cm, annot=True)
        # # plt.xlabel('Predicted')
        # # plt.ylabel('Truth')
        # # st.pyplot(plt)
        
        # #memunculkan akurasi dari hasil testing semua fitur KNN dengan cara menghitung hasil tebakan mesin dibandingkan dengan hasil target aktual/sebenarnya
        # st.text("Akurasi dari model KNN dengan menggunakan K = 4 (dengan semua fitur)")
        # st.text("yaitu " + str(metrics.accuracy_score(y_test, yhat1) * 100) + "%")

        # plt.figure(figsize=(8,6))
        # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        # plt.xlabel("Predicted Labels")
        # plt.ylabel("True Labels")
        # st.pyplot(plt.gcf())  # Use plt.gcf() to get the current figure
        
        if st.checkbox("Edit Parameter KNN"):
            #sama kek edit parameter KNN tadi, jadinya cmn nambah nilai K, terus dataset yang sudah diinputin user itu ditrain lagi dengan nilai K yang baru
            K = st.number_input('Masukkan Nilai K', value=1)
            
            dataframe["Label"] = dataframe["Label"].replace({"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2})
            
            X = dataframe[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values 
            y = dataframe['Label'].values
    
            X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=4
            )
            
            #Train Model and Predict
            knn = KNeighborsClassifier(n_neighbors = K)
            knn.fit(X_train,y_train)
            yhat1 = knn.predict(X_test)
            
            akurasi = accuracy_score(y_test, yhat1)
            st.text("Akurasi model KNN dengan K = "+ str(K) +" menggunakan data anda: " + str(akurasi*100) + "%")
            
            st.text("")
            st.text("")
            st.header("Confusion Matrix dari Perdiksi Data")
            cm = confusion_matrix(y_test,yhat1)
            species_labels = ["Iris-Virginica", "Iris-Setosa", "Iris-Versicolor"]
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=species_labels)
            fig, ax = plt.subplots(figsize=(8, 6))
            disp.plot(ax=ax, cmap="Blues", xticks_rotation="horizontal", colorbar=False)
            ax.set_xlabel("Predicted Labels")
            ax.set_ylabel("True Labels")
            st.pyplot(fig)
            # plt.figure(figsize=(8,6))
            # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            # plt.xlabel("Predicted Labels")
            # plt.ylabel("True Labels")
            # st.pyplot(plt.gcf())  # Use plt.gcf() to get the current figure