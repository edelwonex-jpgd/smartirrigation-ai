import io
import pandas as pd
import streamlit as st
from model import (FEATURES, STAGES, CATEGORIES, load_dataset, train_model,
                   predict_one, validate_input, encode_features, estimate_water_deficit_mm)

st.set_page_config(page_title='SmartIrrigation AI', page_icon='🌱', layout='wide')
st.markdown('''<style>
.block-container {padding-top:2rem;max-width:1150px}
[data-testid="stMetricValue"] {font-size:1.7rem}
</style>''',unsafe_allow_html=True)

@st.cache_resource(show_spinner='Melatih model AI pada dataset simulasi...')
def get_model():
    dataset = load_dataset()
    model, stats = train_model(dataset)
    return model, stats

model, stats = get_model()
dataset=load_dataset()
st.title('🌱 SmartIrrigation AI')
st.caption('Aplikasi edukasi agroekoteknologi • Random Forest Classifier • Tanaman cabai')
st.warning('**PROTOTIPE AKADEMIK.** Model dilatih dengan **data simulasi**, bukan pengamatan lapangan. Output bukan dasar tunggal untuk menentukan irigasi riil. Input kelembapan harus berupa **kadar air volumetrik (% VWC)**; nilai sensor mentah perlu kalibrasi terlebih dahulu.')

tab1,tab2,tab3,tab4 = st.tabs(['💧 Analisis 1 lahan','📂 Analisis CSV','📊 Evaluasi model','📘 Metodologi'])
with tab1:
    st.subheader('Data lingkungan dan tanah')
    with st.form('irrigation_form'):
        a,b=st.columns(2)
        with a:
            soil=st.number_input('Kelembapan tanah (% VWC)',0.0,70.0,23.0,1.0)
            fc=st.number_input('Kapasitas lapang (% VWC)',1.0,60.0,35.0,1.0)
            wp=st.number_input('Titik layu permanen (% VWC)',0.0,59.0,15.0,1.0)
            root=st.number_input('Kedalaman zona perakaran (cm)',5.0,100.0,30.0,1.0)
        with b:
            temperature=st.number_input('Suhu udara (°C)',10.0,45.0,30.0,0.5)
            rain=st.number_input('Prakiraan hujan (mm)',0.0,100.0,5.0,1.0)
            stage=st.selectbox('Fase pertumbuhan',STAGES,index=1)
        submitted=st.form_submit_button('🔎 Analisis kebutuhan irigasi',type='primary',use_container_width=True)
    if submitted:
        row=dict(soil_moisture_pct=soil,field_capacity_pct=fc,wilting_point_pct=wp,
                 temperature_c=temperature,forecast_rain_mm=rain,root_depth_cm=root,crop_stage=stage)
        try:
            result=predict_one(model,row)
            st.subheader('Hasil analisis')
            x,y,z=st.columns(3)
            x.metric('Kategori model',result['status'])
            y.metric('Skor kelas tertinggi',f"{result['model_score']}%")
            z.metric('Defisit menuju kapasitas lapang',f"{result['net_deficit_mm']} mm")
            st.info(f"Perkiraan kedalaman air bruto: **{result['gross_depth_mm']} mm** (asumsi efisiensi 85%). Ini merupakan estimasi selisih simpanan air untuk zona akar, **bukan** rekomendasi dosis atau jadwal aplikasi.")
            if result['status']=='Perlu irigasi':
                st.error('Klasifikasi: perlu pemeriksaan kemungkinan irigasi. Verifikasi kondisi tanah, tanaman, hujan, dan kondisi setempat.')
            elif result['status']=='Pantau':
                st.warning('Klasifikasi: pantau kelembapan tanah dan prakiraan cuaca, lalu evaluasi ulang.')
            else:
                st.success('Klasifikasi: belum perlu irigasi menurut model simulasi. Tetap lakukan pemantauan.')
            st.caption('Skor kelas tertinggi adalah keluaran probabilitas Random Forest, **bukan** probabilitas keberhasilan nyata atau nilai kepercayaan terkalibrasi.')
            st.download_button('⬇️ Unduh hasil CSV',pd.DataFrame([{**row,**result}]).to_csv(index=False).encode('utf-8-sig'),file_name='hasil_smartirrigation.csv',mime='text/csv')
        except ValueError as e:
            st.error(str(e))

with tab2:
    st.subheader('Prediksi beberapa lahan sekaligus')
    example=dataset[FEATURES].head(8)
    st.download_button('⬇️ Unduh template CSV',example.to_csv(index=False).encode('utf-8-sig'),file_name='template_input.csv',mime='text/csv')
    upload=st.file_uploader('Unggah CSV dengan kolom sesuai template',type=['csv'])
    if upload is not None:
        try:
            incoming=pd.read_csv(upload)
            if len(incoming)>5000:
                raise ValueError('Maksimal 5.000 baris dalam satu analisis.')
            if len(incoming)==0:
                raise ValueError('CSV kosong.')
            missing=[c for c in FEATURES if c not in incoming.columns]
            if missing:
                raise ValueError('Kolom tidak ditemukan: '+', '.join(missing))
            for idx, row in incoming.iterrows():
                try: validate_input(row)
                except (ValueError,TypeError,KeyError) as e: raise ValueError(f'Baris {idx+2}: {e}') from e
            output=incoming.copy()
            output['status_ai']=model.predict(encode_features(output))
            scores=model.predict_proba(encode_features(output))
            output['skor_kelas_tertinggi_pct']=(scores.max(axis=1)*100).round(1)
            output['defisit_air_net_mm']=[estimate_water_deficit_mm(r)[0] for _,r in output.iterrows()]
            st.success(f'{len(output)} baris berhasil diproses.')
            st.dataframe(output,use_container_width=True)
            st.download_button('⬇️ Unduh seluruh prediksi',output.to_csv(index=False).encode('utf-8-sig'),file_name='prediksi_batch.csv',mime='text/csv')
        except (ValueError, KeyError, pd.errors.ParserError, UnicodeDecodeError) as e:
            st.error(f'Gagal memproses CSV: {e}')

with tab3:
    st.subheader('Evaluasi model dengan data uji simulasi')
    x,y,z=st.columns(3)
    x.metric('Akurasi holdout',f"{stats['accuracy']:.1%}")
    y.metric('Macro F1',f"{stats['macro_f1']:.3f}")
    z.metric('Data latih / uji',f"{stats['train_n']} / {stats['test_n']}")
    st.caption('Pembagian 80:20 dilakukan secara stratified. Nilai di atas hanya menunjukkan kecocokan pada label yang dihasilkan dari aturan simulasi, BUKAN akurasi lapangan.')
    matrix=pd.DataFrame(stats['confusion'], index=[f'Aktual: {c}' for c in CATEGORIES],columns=[f'Prediksi: {c}' for c in CATEGORIES])
    st.write('**Confusion matrix**')
    st.dataframe(matrix,use_container_width=True)
    importance=pd.Series(model.feature_importances_,index=FEATURES).sort_values(ascending=False)
    st.write('**Feature importance (model simulasi)**')
    st.bar_chart(importance)
    with st.expander('Lihat contoh dataset sintetis'):
        st.dataframe(dataset.head(30),use_container_width=True)
        st.download_button('⬇️ Unduh dataset lengkap',dataset.to_csv(index=False).encode('utf-8-sig'),file_name='dataset_simulasi.csv',mime='text/csv')

with tab4:
    st.markdown('''### Bagaimana AI bekerja?
1. Dataset sintetis berisi 2.400 contoh kondisi tanah, cuaca, dan pertumbuhan cabai yang dihasilkan dengan seed tetap.
2. **Label kelas** (Belum perlu, Pantau, Perlu irigasi) dibentuk oleh aturan hipotetis dari fraksi air tersedia, suhu, hujan, dan fase pertumbuhan, serta gangguan acak. Label **bukan** hasil pengamatan agronomis.
3. Data dibagi menjadi 80% pelatihan dan 20% pengujian; `RandomForestClassifier` mempelajari hubungan fitur dan label.
4. Input baru diklasifikasikan oleh model terlatih; metrik pengujian serta confusion matrix ditampilkan.
5. Kedalaman defisit air dihitung terpisah dari model AI dengan persamaan sederhana berikut.

**Defisit air net (mm) = max(0, kapasitas lapang − kadar air tanah) × kedalaman akar (cm) / 10**

**Kedalaman bruto (mm) = defisit net / 0,85**

Rumus mengasumsikan semua persen merupakan kadar air volumetrik, kondisi air homogen sepanjang zona akar, dan efisiensi aplikasi 85%. Tidak memperhitungkan evapotranspirasi, infiltrasi, sifat hidraulik, pembatasan air tersedia, curah hujan efektif, atau jadwal irigasi.

### Keterbatasan penting
- Belum terhubung ke sensor IoT dan belum memiliki data nyata yang divalidasi.
- Angka akurasi hanyalah akurasi terhadap aturan simulasi.
- Untuk penelitian atau penerapan nyata, kumpulkan data lapangan terkalibrasi, tentukan label oleh ahli, lalu lakukan validasi lintas musim/lokasi.
- Aplikasi ini tidak memicu pompa secara otomatis.

**Teknologi:** Python, Streamlit, Pandas, NumPy, Scikit-learn / Random Forest.
''')

st.divider()
st.caption('SmartIrrigation AI • Prototipe pembelajaran • Bukan sistem kendali irigasi operasional')
