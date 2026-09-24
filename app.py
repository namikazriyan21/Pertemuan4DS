import streamlit as st
import pandas as pd
import joblib

# Load artefak model
scaler_deploy = joblib.load("scaler_retail.joblib")
kmeans_deploy = joblib.load("kmeans_retail.joblib")

st.title("Retail Customer Segmentation AI")
st.write("Masukkan profil pendapatan dan kebiasaan belanja pelanggan di bawah ini.")

# Form Input
income = st.number_input("Pendapatan Tahunan ($)", min_value=0, value=45000)
mnt_meat = st.number_input("Total Belanja Produk Daging ($)", min_value=0, value=120)
mnt_wines = st.number_input("Total Belanja Minuman Anggur ($)", min_value=0, value=250)

# Tombol Prediksi
if st.button("Prediksi Segmen"):
    input_df = pd.DataFrame(
        [[income, mnt_meat, mnt_wines]], 
        columns=["Income", "MntMeatProducts", "MntWines"]
    )
    
    scaled_input = scaler_deploy.transform(input_df)
    cluster_result = kmeans_deploy.predict(scaled_input)[0]
    
    profil = ""
    if cluster_result == 0:
        profil = "Pelanggan Ekonomis / Fokus Barang Kebutuhan Dasar"
    elif cluster_result == 1:
        profil = "Pelanggan Kelas Menengah / Belanja Fleksibel"
    else:
        profil = "Pelanggan Premium / Pengeluaran Besar"
        
    st.success(f"Pelanggan ini masuk ke dalam **Cluster {cluster_result}** ({profil})")