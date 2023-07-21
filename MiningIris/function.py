# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 14:22:30 2023

@author: USER
"""
import pandas as pd
import streamlit as st

#fungsi untuk melakukan load dataset
@st.cache_data()
def load_data1():
    data = pd.read_csv('iris.csv')
    return data