import numpy as np
import pickle
import pandas as pd
import streamlit as st
import joblib
import time

import xgboost as xgb

st.title("Vehicle Pricing Prediction App")


# Load model
XGBRegressor_model = xgb.XGBRegressor()

# load params
with open('xgb_best_params.pkl', 'rb') as f:
    xgb_best_params = pickle.load(f)

# assign best params to model
XGBRegressor_model = xgb.XGBRegressor(**xgb_best_params)

st.header(XGBRegressor_model)

