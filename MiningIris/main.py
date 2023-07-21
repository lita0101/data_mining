# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 14:12:10 2023

@author: USER
"""

import streamlit as st
# from function import load_data

#melakukan import tab Home dan Prediksi
from Tabs import Home, Predict

Tabs = {
        "Beranda" : Home,
        "Prediksi" : Predict,
        }

#sidebar
st.sidebar.title("Navigasi")

#option/membuat option
page = st.sidebar.radio("Halaman", list(Tabs.keys()))

#load data
# data,x,y = load_data()

#call function
# if page in ["Prediction", "Visualisation"]:
#     Tabs[page].app(data,x,y)
# else:
#     Tabs[page].app()


#inisialisasi tab
Tabs[page].app()