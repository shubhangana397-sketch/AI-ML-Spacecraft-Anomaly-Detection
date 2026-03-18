import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Space Mission Anomaly Dashboard", layout="wide")
st.title("AI-Based Anomaly Detection - Deep Space Mission Dashboard")

col_names = ["unit_id","cycle","op1","op2","op3"] + [f"s{i}" for i in range(1,22)]
train = pd.read_csv(r"D:\CMAPSSData\train_FD001.txt", sep=r"\s+", header=None, names=col_names)

max_cycles = train.groupby("unit_id")["cycle"].max().reset_index()
max_cycles.columns = ["unit_id", "max_cycle"]
train = train.merge(max_cycles, on="unit_id")
train["RUL"] = train["max_cycle"] - train["cycle"]
train.drop(columns=["max_cycle"], inplace=True)

useful_sensors = ["s2","s3","s4","s7","s9","s11","s12","s14","s17","s20","s21"]
scaler = MinMaxScaler()
train[useful_sensors] = scaler.fit_transform(train[useful_sensors])
train["anomaly"] = (train["RUL"] <= 30).astype(int)

iso = IsolationForest(n_estimators=100, contamination=0.15, random_state=42)
iso.fit(train[useful_sensors])
train["iso_pred"] = (iso.predict(train[useful_sensors]) == -1).astype(int)

def decision(rul, pred):
    if pred == 0: return "NORMAL"
    elif rul > 100: return "WATCH"
    elif rul > 50: return "WARNING"
    elif rul > 30: return "CRITICAL"
    else: return "EMERGENCY"

train["decision"] = train.apply(lambda r: decision(r["RUL"], r["iso_pred"]), axis=1)

st.sidebar.header("Select Engine")
engine_id = st.sidebar.selectbox("Engine ID", sorted(train["unit_id"].unique()))
engine_df = train[train["unit_id"] == engine_id]

col1, col2, col3, col4 = st.columns(4)
last = engine_df.iloc[-1]
col1.metric("Engine ID", int(engine_id))
col2.metric("Total Cycles", int(engine_df["cycle"].max()))
col3.metric("Remaining Useful Life", int(last["RUL"]))
col4.metric("Status", last["decision"])

st.markdown("---")

st.subheader("Sensor Readings Over Time")
sensor = st.selectbox("Select Sensor", useful_sensors)
fig = px.line(engine_df, x="cycle", y=sensor, title=f"Engine {engine_id} - {sensor}",
              color_discrete_sequence=["tomato"])
st.plotly_chart(fig, use_container_width=True)

st.subheader("Anomaly Detection Timeline")
fig2 = px.scatter(engine_df, x="cycle", y="RUL", color="iso_pred",
                  color_discrete_map={0:"green", 1:"red"},
                  title=f"Engine {engine_id} - Anomaly Timeline (Red = Anomaly)")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Fleet-Wide Decision Distribution")
dist = train["decision"].value_counts().reset_index()
dist.columns = ["Decision", "Count"]
fig3 = px.bar(dist, x="Decision", y="Count", color="Decision",
              color_discrete_map={"NORMAL":"green","WATCH":"blue",
                                  "WARNING":"orange","CRITICAL":"red","EMERGENCY":"darkred"})
st.plotly_chart(fig3, use_container_width=True)
