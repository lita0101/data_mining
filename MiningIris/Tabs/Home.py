# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 14:13:05 2023

@author: USER
"""
import streamlit as st
import pandas as pd
from PIL import Image
from function import load_data1
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_decision_regions
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def app():
    #untuk judul aplikasi
    st.write("""
             # Klasifikasi Bunga Iris
             Aplikasi untuk melakukan klasifikasi terhadap **bunga Iris**
             """)
    
    #membuka dan memunculkan gambar
    img = Image.open('irisimg.png')
    st.image(img)
    
    #membuka dan memunculkan dataset
    st.text("")
    st.text("")
    st.markdown("**:green[Dataset]** yang digunakan untuk melakukan **Train dan Testing Model**")
    data = load_data1()
    st.dataframe(data)
    dataset_link = "https://github.com/achmatim/data-mining/blob/main/Dataset/iris.csv"
    st.markdown(f"Akses Dataset: [Klik Disini]({dataset_link})")
    st.text("")
    
    #membagi data untuk melihat persebaran data
    data0 = data[:50]
    data1 = data[50:100]
    data2 = data[100:]
    
    #untuk melihat persebaran data
    st.header("Sebaran Data Iris")
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.scatter(data0['SepalLengthCm'], data0['SepalWidthCm'], color="green", marker='+')
    plt.scatter(data1['SepalLengthCm'], data1['SepalWidthCm'], color="blue", marker='.')
    plt.scatter(data2['SepalLengthCm'], data2['SepalWidthCm'], color="red", marker='*')
    st.pyplot(plt)
    
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.scatter(data0['PetalLengthCm'], data0['PetalWidthCm'], color="green", marker='+')
    plt.scatter(data1['PetalLengthCm'], data1['PetalWidthCm'], color="blue", marker='.')
    plt.scatter(data2['PetalLengthCm'], data2['PetalWidthCm'], color="red", marker='*')
    st.pyplot(plt)
    
    st.header("Persebaran Data")
    st.text("Split data dengan 80% Training and 20% Testing")
    col1, col2, col3 = st.columns(3)
    with col1:
        
        #menghitung banyak nya data per spesies iris dalam dataset
        st.text("Spesies dari dataset:")
        
        # Assuming the column name is 'species'
        species_counts = data['Label'].value_counts()
        
        for label, count in species_counts.items():
            st.write(f"{label}: {count}")
    with col2:
        #menghitung banyaknya data training per spesies (menunjukkan dalam data training, berapa banyak per spesies)
        #membagi data fitur dan data target
        X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values 
        data["Label"] = data["Label"].replace({"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2})
        y = data['Label']
        
        #melakukan scaling/normalisasi dataset
        X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
        
        #membagi menjadi data test dan data train dengan rasio 80:20 (80% data train, 20% data test) dengan skala keacakan 4
        #0.2 itu banyak 20% jadinya 0.2 merupakan rasio data testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=4
        )
        st.text("Training Dataset:")
        training_species_counts = y_train.value_counts()
        
        #penghitungan banyaknya data training per spesies
        for species, count in training_species_counts.items():
            if species == 0:
                species = "Iris-setosa"
            if species == 1:
                species = "Iris-versicolor"
            if species == 2:
                species = "Iris-virginica"
            st.write(f"{species}: {count}")
    with col3:
        
        #menghitung banyaknya data training per spesies (menunjukkan dalam data training, berapa banyak per spesies)
        st.text("Testing Dataset:")
        testing_species_counts = y_test.value_counts()
        
        #penghitungan banyaknya data training per spesies
        for species, count in testing_species_counts.items():
            if species == 0:
                species = "Iris-setosa"
            if species == 1:
                species = "Iris-versicolor"
            if species == 2:
                species = "Iris-virginica"
            st.write(f"{species}: {count}")
            
    #membuat dan memunculkan correlation heatmap
    st.text("")
    st.text("")
    st.header("Correlation Heatmap")
    # menentukan ukuran plot
    plt.figure(figsize=(15, 10))
    # membuat korelasi heatmap
    heatmap = sns.heatmap(data.corr(), vmin=-1, vmax=1, annot=True)
    # menentukan format plot (judul, ukuran teks)
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 12}, pad=12)
    st.pyplot(plt)
    
    #membagi data fitur dan data target
    X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values 
    data["Label"] = data["Label"].replace({"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2})
    y = data['Label'].values
    
    x_data = data[
        [
            "SepalLengthCm",
            "SepalWidthCm",
            "PetalLengthCm",
            "PetalWidthCm",
        ]
    ]
    y_target = data["Label"]

    scaler = MinMaxScaler()
    x_data = scaler.fit_transform(x_data)
    x_data = pd.DataFrame(
        x_data,
        columns=[
            "SepalLengthCm",
            "SepalWidthCm",
            "PetalLengthCm",
            "PetalWidthCm",
        ],
    )

    X_train, X_test, y_train, y_test = train_test_split(
        x_data, y_target, test_size=0.33, stratify=y_target
    )

    treeClass = tree.DecisionTreeClassifier(
        ccp_alpha=0.0,
        class_weight=None,
        criterion="entropy",
        max_depth=4,
        max_features=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        min_samples_leaf=1,
        min_samples_split=2,
        min_weight_fraction_leaf=0.0,
        splitter="best",
    )
    treeClass.fit(X_train, y_train)
    y_pred = treeClass.predict(X_test)
    treeAccuracy = accuracy_score(y_pred, y_test)
    # joblib.dump(treeAccuracy, "model.sav")
    st.text("Accuracy From This Model: " + str(treeAccuracy * 100) + "%")

    st.text("")
    st.text("")
    st.header("Decission Tree Model for this dataset")
    dot_data = tree.export_graphviz(
    decision_tree=treeClass,
    max_depth=5,
    out_file=None,
    filled=True,
    rounded=True,
    feature_names=[
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm",
    ],
    class_names=["Iris-setosa", "Iris-versicolor", "Iris-virginica"],
)

    st.graphviz_chart(dot_data)

    species_labels = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    st.text("")
    st.text("")
    st.header("Confusion Matrix from testing in this dataset")
    cm = confusion_matrix(y_test, y_pred)
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    # plt.xlabel("Predicted Labels")
    # plt.ylabel("True Labels")
    
    # plt.xticks(np.arange(len(species_labels)), species_labels)
    # plt.yticks(np.arange(len(species_labels)), species_labels)
    # labels = y.unique().astype(str)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=species_labels)
    fig, ax = plt.subplots(figsize=(6, 4))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation="horizontal", colorbar=False)
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    st.pyplot(fig)
    
    #melakukan scaling/normalisasi dataset
    # X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
    
    # #membagi menjadi data test dan data train dengan rasio 80:20 (80% data train, 20% data test) dengan skala keacakan 4
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=4
    # )
    
    # #membuat akurasi
    # st.text("")
    # st.text("")
    # st.header("Akurasi")
    
    # #latihan hanya menggunakan 2 fitur yaitu sepal length dan width untuk dimunculkan proses KNN nya
    # k = 4
    # # Train Model and Predict
    # knn = KNeighborsClassifier(n_neighbors=k)
    # knn.fit(X_train[:, :2], y_train)  # Using only the first 2 features
    # yhat = knn.predict(X_test[:, :2])
    
    # # Define legend labels atau memberikan label
    # legend_labels = ['Iris-Virginica', 'Iris-Setosa', 'Iris-Versicolor']
    
    # #memunculkan hasil KNN meggunakan fitur sepal length dan width
    # plt.figure(figsize=(7, 5))
    # plot_decision_regions(X_train[:, :2], y_train, clf=knn, legend=2)
    # plt.xlabel('Sepal Length (cm)')
    # plt.ylabel('Sepal Width (cm)')
    # plt.title('KNN Decision Regions (Sepal)')
    
    # #membuat label custom untuk menunjukkan label dalam bentuk string
    # # Create a custom legend
    # custom_legend = [plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=8)
    #                  for label, color in zip(legend_labels, ListedColormap(['red', 'blue', 'green']).colors)]
    
    # #mengatur posisi label
    # # Add the legend to the plot
    # plt.legend(handles=custom_legend, loc='upper left')
    
    # #memunculkan plot hasil KNN yang sudah dibuat menggunakan 2 fitur yaitu sepal length dan width
    # st.pyplot(plt)
    
    # #memunculkan akurasi testing/pengujian
    # st.text("Akurasi dari model KNN dengan menggunakan K = 4")
    # st.text("untuk Sepal Length dan Sepal Width adalah " + str(metrics.accuracy_score(y_test, yhat) * 100) + "%") #83.33
        
    # #latihan menggunakan 2 fitur yaitu petal legth dan width
    # k = 4
    # # Train Model and Predict
    # knn = KNeighborsClassifier(n_neighbors=k)
    # knn.fit(X_train[:, 2:], y_train)  # Using the last 2 features
    # yhat = knn.predict(X_test[:, 2:])
    
    # # Define legend labels atau memberikan label
    # legend_labels = ['Iris-Virginica', 'Iris-Setosa', 'Iris-Versicolor']
    
    # #memunculkan hasil KNN meggunakan fitur petal length dan width
    # plt.figure(figsize=(7, 5))
    # plot_decision_regions(X_train[:, 2:], y_train, clf=knn, legend=2)
    # plt.xlabel('Petal Length (cm)')
    # plt.ylabel('Petal Width (cm)')
    # plt.title('KNN Decision Regions (Petal)')
    
    # #membuat label custom untuk menunjukkan label dalam bentuk string
    # # Create a custom legend
    # custom_legend = [plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=8)
    #                  for label, color in zip(legend_labels, ListedColormap(['red', 'blue', 'green']).colors)]
    
    # #mengatur posisi label
    # # Add the legend to the plot
    # plt.legend(handles=custom_legend, loc='upper left')
    
    # #memunculkan plot hasil KNN yang sudah dibuat menggunakan 2 fitur yaitu petal length dan width
    # st.pyplot(plt)
    
    # #memunculkan akurasi testing/pengujian
    # st.text("Akurasi dari model KNN dengan menggunakan K = 4")
    # st.text("untuk Petal Length dan Petal Width adalah " + str(metrics.accuracy_score(y_test, yhat) * 100) + "%") #96,...
    
    # #membuat model KNN untuk melatih 4 fitur sekaligus
    # k = 4
    # #Train Model and Predict
    # knn1 = KNeighborsClassifier(n_neighbors = k)
    # knn1.fit(X_train,y_train)
    # yhat1 = knn1.predict(X_test)
    # #melakukan prediksi pada data test
    # y_pred1 = knn1.predict(X_test)
    
    # #membuat confusion matrix dari hasil testing untuk melihat data yang salah atau benar ditebak oleh mesin
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
    
    # Perform dimensionality reduction using PCA
    # pca = PCA(n_components=2)
    # X_pca = pca.fit_transform(X)
    
    # # Create an instance of the KNN classifier and fit the model to the reduced data
    # k = 4  # Number of neighbors
    # knn = KNeighborsClassifier(n_neighbors=k)
    # knn.fit(X_pca, y)
    
    # # Plot the decision regions using the reduced feature space
    # plt.figure(figsize=(8, 6))
    # plot_decision_regions(X_pca, y.values, clf=knn, legend=2)
    # plt.xlabel('Principal Component 1')
    # plt.ylabel('Principal Component 2')
    # plt.title('KNN Decision Regions')
    # plt.show()
    
    # fig = plt.figure()
    # plt.scatter(X_test, y_test, c='y', alpha=0.8, cmap='viridis')
    
    # plt.xlabel('Principal Component 1')    
    # plt.ylabel('Principal Component 2')
    # plt.colorbar()
    
    # st.pyplot(fig)
    
# import streamlit as st
# import pandas as pd
# from PIL import Image
# from function import load_data1
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn import preprocessing
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn import metrics
# from sklearn.metrics import confusion_matrix
# from mlxtend.plotting import plot_decision_regions

# def app():
#     st.write("""
#              # Web Klasifikasi Bunga Iris
#              Aplikasi untuk melakukan klasifikasi terhadap **bunga Iris**
#              """)
    
#     img = Image.open('irisimg.png')
#     st.image(img)
    
#     st.text("")
#     st.text("")
#     st.markdown("**:green[Dataset]** yang digunakan untuk melakukan **Train dan Testing Model**")
#     data = load_data1()
#     st.dataframe(data)
#     dataset_link = "https://github.com/achmatim/data-mining/blob/main/Dataset/iris.csv"
#     st.markdown(f"Akses Dataset: [Klik Disini]({dataset_link})")
#     st.text("")
    
#     data0 = data[:50]
#     data1 = data[50:100]
#     data2 = data[100:]
    
#     st.header("Sebaran Data Iris")
#     plt.xlabel('Sepal Length')
#     plt.ylabel('Sepal Width')
#     plt.scatter(data0['SepalLengthCm'], data0['SepalWidthCm'], color="green", marker='+')
#     plt.scatter(data1['SepalLengthCm'], data1['SepalWidthCm'], color="blue", marker='.')
#     plt.scatter(data2['SepalLengthCm'], data2['SepalWidthCm'], color="red", marker='*')
#     st.pyplot(plt)
    
#     plt.xlabel('Petal Length')
#     plt.ylabel('Petal Width')
#     plt.scatter(data0['PetalLengthCm'], data0['PetalWidthCm'], color="green", marker='+')
#     plt.scatter(data1['PetalLengthCm'], data1['PetalWidthCm'], color="blue", marker='.')
#     plt.scatter(data2['PetalLengthCm'], data2['PetalWidthCm'], color="red", marker='*')
#     st.pyplot(plt)
    
#     st.text("")
#     st.text("")
#     st.header("Correlation Heatmap")
#     # menentukan ukuran plot
#     plt.figure(figsize=(15, 10))
#     # membuat korelasi heatmap
#     heatmap = sns.heatmap(data.corr(), vmin=-1, vmax=1, annot=True)
#     # menentukan format plot (judul, ukuran teks)
#     heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 12}, pad=12)
#     st.pyplot(plt)
    
#     X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values 
#     data["Label"] = data["Label"].replace({"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2})
#     y = data['Label'].values
    
#     X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
    
    # k = 4
    # # Train Model and Predict
    # knn = KNeighborsClassifier(n_neighbors=k)
    # knn.fit(X_train, y_train)
    # yhat = knn.predict(X_test)
    # plt.figure(figsize=(7, 5))
    
    # plot_decision_regions(X_train, y_train, clf=knn, legend=2)
    # plt.xlabel('Sepal Length (cm)')
    # plt.ylabel('Sepal Width (cm)')
    # plt.title('KNN Decision Regions')
    # st.pyplot(plt)
    
    # st.text("Akurasi dari model KNN dengan menggunakan K = 4 yaitu " + str(metrics.accuracy_score(y_test, yhat) * 100) + "%")
    
    # y_pred = knn.predict(X_test)
    # cm = confusion_matrix(y_test, y_pred)
    
    # plt.figure(figsize=(7, 5))
    # sns.heatmap(cm, annot=True)
    # plt.xlabel('Predicted')
    # plt.ylabel('Truth')
    # st.pyplot(plt)