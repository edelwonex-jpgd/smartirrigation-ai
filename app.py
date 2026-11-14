"""SmartIrrigation AI — Future Farming UI. Educational prototype, synthetic-data classifier."""
from html import escape
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from model import (FEATURES, STAGES, CATEGORIES, load_dataset, train_model,
                   predict_one, validate_input, encode_features, estimate_water_deficit_mm)

st.set_page_config(page_title='SmartIrrigation AI | Future Farming', page_icon='💠', layout='wide', initial_sidebar_state='expanded')

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root { --a:#9b72ff; --b:#42e3d5; --muted:#9aabc6; }
html, body, [class*="css"], [data-testid="stApp"] {font-family:'DM Sans',sans-serif;}
[data-testid="stAppViewContainer"] {background: radial-gradient(ellipse at 85% -15%,rgba(113,65,221,.22),transparent 36%),radial-gradient(ellipse at -10% 25%,rgba(24,139,177,.13),transparent 34%),#080d1d;}
[data-testid="stHeader"] {background:rgba(8,13,29,.65);backdrop-filter:blur(12px);}
[data-testid="stSidebar"] {background:linear-gradient(175deg,#141b35,#0b1125) !important;border-right:1px solid rgba(167,182,255,.13);}
[data-testid="stSidebar"] * {color:#dce6ff;}
.block-container {max-width:1450px;padding-top:1.7rem;padding-bottom:3rem;}
h1,h2,h3,h4 {font-family:'Space Grotesk',sans-serif !important;letter-spacing:-.025em;}
h1 {font-weight:700 !important;} h3 {font-size:1.3rem !important;}
p,li {color:#cbd6ec;}
[data-testid="stCaptionContainer"] p,small {color:#9aabc6 !important;}
.hero {border:1px solid rgba(169,165,255,.23);border-radius:24px;padding:31px 35px;margin-bottom:22px;position:relative;overflow:hidden;background:radial-gradient(circle at 93% 20%,rgba(66,227,213,.16),transparent 29%),linear-gradient(108deg,rgba(73,47,135,.46),rgba(25,36,66,.70) 70%);box-shadow:0 20px 70px rgba(0,0,0,.17);}
.hero:after {content:'';position:absolute;width:230px;height:230px;border-radius:50%;right:-55px;bottom:-135px;background:rgba(115,85,244,.18);filter:blur(12px);pointer-events:none;}
.eyebrow {color:#55e0d1;font-size:.78rem;font-weight:700;letter-spacing:.17em;text-transform:uppercase;}
.hero h1 {font-size:clamp(2rem,4vw,3.2rem);margin:.45rem 0 .4rem;line-height:1.1;color:#f4f7ff;}
.gradient-text {background:linear-gradient(95deg,#a992ff,#62e7d5);-webkit-background-clip:text;background-clip:text;color:transparent;}
.hero p {color:#bbcae7;max-width:650px;margin:0;font-size:1rem;line-height:1.65;}
.pill {display:inline-block;border:1px solid rgba(86,226,207,.32);background:rgba(31,140,141,.12);color:#8cf4e6;border-radius:999px;padding:6px 12px;font-size:.75rem;margin-top:17px;}
.kpi {border:1px solid rgba(167,188,247,.14);background:linear-gradient(145deg,rgba(29,39,69,.9),rgba(18,26,49,.76));border-radius:19px;padding:18px 20px;min-height:125px;box-shadow:0 10px 28px rgba(0,0,0,.11);}
.kpi .klabel {color:#9fb0ce;font-size:.79rem;letter-spacing:.035em;}
.kpi .knum {font:700 clamp(1.3rem,2.5vw,2.05rem) 'Space Grotesk',sans-serif;color:#f7f8ff;margin-top:10px;line-height:1.25;overflow-wrap:anywhere;}
.kpi .kfoot {color:#58e0d6;font-size:.73rem;margin-top:8px;}
.section-label {font:700 .84rem 'Space Grotesk',sans-serif;text-transform:uppercase;letter-spacing:.13em;color:#aab9da;margin:8px 0 14px;}
.glass {border:1px solid rgba(161,181,232,.16);background:linear-gradient(130deg,rgba(32,42,75,.76),rgba(17,24,47,.75));border-radius:18px;padding:19px 22px;margin:10px 0 18px;}
.glass p {margin:0;color:#bbc8e3;}
.callout {border:1px solid rgba(234,187,87,.28);background:rgba(159,111,35,.11);color:#f6deac;border-radius:13px;padding:12px 16px;margin:10px 0 22px;font-size:.88rem;line-height:1.6;}
.statusgood {color:#4fe3c5 !important;} .statusmid {color:#ffcb77 !important;} .statusbad {color:#ff8294 !important;}
[data-testid="stTabs"] [data-baseweb="tab-list"] {gap:7px;flex-wrap:wrap;}
[data-testid="stTabs"] button[data-baseweb="tab"] {border:1px solid rgba(155,172,230,.16);border-radius:12px;background:rgba(31,41,76,.65);padding:9px 17px;color:#d9e5ff;}
[data-testid="stTabs"] button[aria-selected="true"] {background:rgba(114,91,223,.23);border-color:rgba(155,121,255,.55);}
[data-testid="stTabs"] [data-baseweb="tab-highlight"], [data-testid="stTabs"] [data-baseweb="tab-border"] {display:none;}
[data-testid="stForm"], [data-testid="stVerticalBlockBorderWrapper"] {border-color:rgba(158,179,238,.17) !important;border-radius:18px !important;}
[data-testid="stForm"] {background:rgba(23,32,60,.4);padding:14px;}
[data-testid="stWidgetLabel"] p {color:#c4d0ea;font-weight:600;font-size:.87rem;}
[data-baseweb="input"] > div, [data-baseweb="select"] > div {background:#151e39 !important;border-color:rgba(170,190,255,.18) !important;border-radius:11px !important;}
.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] button {background:linear-gradient(110deg,#805bf4,#469fd6) !important;border:0 !important;border-radius:11px !important;color:white !important;font-weight:700;box-shadow:0 5px 21px rgba(108,92,241,.27);}
.stDownloadButton button {border:1px solid rgba(109,227,220,.32) !important;color:#a6f5ea !important;background:rgba(36,130,140,.1) !important;border-radius:11px !important;}
[data-testid="stFileUploader"] {background:rgba(24,34,63,.44);border-radius:12px;}
[data-testid="stDataFrame"] {border-radius:13px;overflow:hidden;}
.footer {text-align:center;color:#8596b7;font-size:.79rem;padding:18px 0 7px;border-top:1px solid rgba(150,170,224,.12);margin-top:32px;}
@media(max-width:700px){.block-container{padding:1rem .85rem 2rem;}.hero{padding:22px 19px;border-radius:18px;}.hero h1{font-size:2rem}.kpi{min-height:110px;padding:14px}.glass{padding:15px}.callout{font-size:.8rem}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

@st.cache_resource(show_spinner='Mempersiapkan model Random Forest...')
def get_model():
    return train_model(load_dataset())

@st.cache_data
def get_data():
    return load_dataset()

try:
    dataset = get_data()
    model, stats = get_model()
except Exception as err:
    st.error(f'Model atau dataset gagal dimuat: {err}')
    st.info('Pastikan dataset ada di data/synthetic_irrigation.csv atau synthetic_irrigation.csv pada root repository.')
    st.stop()


def kpi(label, value, footer='● Dataset simulasi', color=''):
    st.markdown(f'<div class="kpi"><div class="klabel">{escape(str(label))}</div><div class="knum {color}">{escape(str(value))}</div><div class="kfoot">{escape(str(footer))}</div></div>', unsafe_allow_html=True)


def plot_style(fig, height=330):
    fig.update_layout(height=height,margin=dict(l=14,r=14,t=26,b=12),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font=dict(color='#c1d0e9',family='DM Sans'),legend=dict(orientation='h',y=-.2),hoverlabel=dict(bgcolor='#1b2746',font_color='#ffffff'))
    fig.update_xaxes(gridcolor='rgba(190,205,250,.07)',zeroline=False)
    fig.update_yaxes(gridcolor='rgba(190,205,250,.07)',zeroline=False)
    return fig

with st.sidebar:
    st.markdown('### 💠 FUTURE FARMING')
    st.caption('SMART AGRICULTURE INTELLIGENCE')
    st.divider()
    st.markdown('**⚡ Sistem analisis**')
    st.write('💧 Irrigation intelligence')
    st.write('🧠 Random Forest ML')
    st.write('📈 Predictive analytics')
    st.write('🌱 Tanaman cabai')
    st.divider()
    st.markdown('**Status sistem**')
    st.success('● AI model siap digunakan')
    st.caption(f"{len(dataset):,} sampel sintetis · {stats['train_n']:,} latih · {stats['test_n']:,} uji")
    st.divider()
    st.caption('Versi 2.0 · Academic demonstration')
    st.caption('Tidak terhubung dengan sensor atau pompa. Hasil wajib divalidasi sebelum penggunaan nyata.')

st.markdown('''<div class="hero">
<div class="eyebrow">◈ PRECISION AGRICULTURE · AI POWERED</div>
<h1>SmartIrrigation <span class="gradient-text">AI</span></h1>
<p>Explore the future of intelligent farming. Analisis kebutuhan irigasi tanaman cabai dengan machine learning, pemantauan parameter, dan visualisasi data yang intuitif.</p>
<span class="pill">● FUTURE FARMING EDITION &nbsp; / &nbsp; MACHINE LEARNING READY</span>
</div>''',unsafe_allow_html=True)

st.markdown('''<div class="callout"><b>⚠️ Prototipe akademik:</b> Model menggunakan label dari dataset <b>sintetis</b>, bukan pengamatan lapangan. Skor model bukan tingkat keyakinan yang terkalibrasi. Kelembapan tanah dan kapasitas lapang harus menggunakan satuan <b>kadar air volumetrik (% VWC)</b>, bukan angka mentah sensor.</div>''',unsafe_allow_html=True)

st.markdown('<div class="section-label">✦ Model overview</div>',unsafe_allow_html=True)
a,b,c,d = st.columns(4)
with a: kpi('MODEL ENGINE','Random Forest','● Classifier aktif')
with b: kpi('AKURASI UJI',f"{stats['accuracy']*100:.2f}%",'● Pada label sintetis')
with c: kpi('MACRO F1',f"{stats['macro_f1']:.3f}",'● Holdout evaluation')
with d: kpi('DATASET',f"{len(dataset):,}",'● Synthetic records')

st.write('')
tab_predict, tab_batch, tab_analytics, tab_about = st.tabs(['💧 Irrigation Lab','📂 Batch Intelligence','📊 AI Analytics','📖 Model & Science'])

with tab_predict:
    st.markdown('### Intelligent Irrigation Assessment')
    st.caption('Masukkan kondisi lahan untuk mendapatkan hasil klasifikasi model dan estimasi selisih simpanan air.')
    with st.form('assessment'):
        left,right=st.columns(2,gap='large')
        with left:
            st.markdown('**🌱 Soil parameters**')
            soil=st.number_input('Kelembapan tanah (% VWC)',min_value=0.0,max_value=70.0,value=23.0,step=1.0)
            fc=st.number_input('Kapasitas lapang (% VWC)',min_value=1.0,max_value=60.0,value=35.0,step=1.0)
            wp=st.number_input('Titik layu permanen (% VWC)',min_value=0.0,max_value=59.0,value=15.0,step=1.0)
            root=st.number_input('Kedalaman zona akar (cm)',min_value=5.0,max_value=100.0,value=30.0,step=1.0)
        with right:
            st.markdown('**☀️ Environmental parameters**')
            temp=st.number_input('Suhu udara (°C)',min_value=10.0,max_value=45.0,value=30.0,step=.5)
            rain=st.number_input('Prakiraan hujan (mm)',min_value=0.0,max_value=100.0,value=5.0,step=1.0)
            stage=st.selectbox('Fase pertumbuhan',STAGES,index=1)
            st.caption('Tidak ada koneksi cuaca langsung. Semua nilai merupakan input manual.')
        submitted=st.form_submit_button('✦ Jalankan AI Assessment',type='primary',use_container_width=True)
    if submitted:
        row=dict(soil_moisture_pct=soil,field_capacity_pct=fc,wilting_point_pct=wp,temperature_c=temp,forecast_rain_mm=rain,root_depth_cm=root,crop_stage=stage)
        try:
            result=predict_one(model,row)
            st.markdown('### ✧ Assessment Result')
            xa,xb,xc=st.columns(3)
            status_color={'Belum perlu':'statusgood','Pantau':'statusmid','Perlu irigasi':'statusbad'}[result['status']]
            with xa: kpi('STATUS AI',result['status'],'● Kategori simulasi',status_color)
            with xb: kpi('SKOR KELAS',f"{result['model_score']}%",'● Tidak terkalibrasi')
            with xc: kpi('DEFISIT AIR NET',f"{result['net_deficit_mm']} mm",'● Estimasi teoritis')
            pc,gc=st.columns(2,gap='large')
            with pc:
                st.markdown('**Distribusi skor kelas model**')
                prob=model.predict_proba(encode_features(pd.DataFrame([row])))[0]
                order=[list(model.classes_).index(name) for name in CATEGORIES]
                fig=go.Figure(go.Bar(x=[float(prob[i])*100 for i in order],y=CATEGORIES,orientation='h',marker_color=['#4fe3c5','#ffcb77','#a992ff'],text=[f'{prob[i]*100:.1f}%' for i in order],textposition='auto'))
                fig.update_xaxes(range=[0,105],title='Skor kelas (%)')
                st.plotly_chart(plot_style(fig,260),use_container_width=True,config={'displayModeBar':False})
            with gc:
                st.markdown('**Profil ketersediaan air**')
                fraction=(soil-wp)/(fc-wp)*100
                fig=go.Figure(go.Indicator(mode='gauge+number',value=max(0,min(100,fraction)),number={'suffix':'%'},title={'text':'Fraksi air tersedia (dibatasi 0–100%)','font':{'size':13}},gauge={'axis':{'range':[0,100]},'bar':{'color':'#61e4d9'},'bgcolor':'#172441','steps':[{'range':[0,35],'color':'#453153'},{'range':[35,70],'color':'#293660'},{'range':[70,100],'color':'#164a55'}]}))
                st.plotly_chart(plot_style(fig,260),use_container_width=True,config={'displayModeBar':False})
            st.markdown(f'<div class="glass"><b>💠 Estimasi air bruto</b><p style="font-size:1.4rem;color:#7ceee0;font-weight:700;margin:8px 0">{result["gross_depth_mm"]} mm</p><p>Estimasi untuk mengisi defisit menuju kapasitas lapang dengan efisiensi aplikasi asumsi 85%. <b>Bukan dosis atau jadwal irigasi operasional.</b></p></div>',unsafe_allow_html=True)
            if result['status']=='Perlu irigasi':
                st.error('Model mengklasifikasikan “Perlu irigasi”. Periksa kelembapan aktual, cuaca, dan kondisi tanaman sebelum tindakan.')
            elif result['status']=='Pantau':
                st.warning('Model mengklasifikasikan “Pantau”. Lanjutkan pemantauan dan validasi pengukuran.')
            else:
                st.success('Model mengklasifikasikan “Belum perlu”. Tetap pantau kondisi lapangan.')
            st.download_button('↓ Download assessment (.csv)',pd.DataFrame([{**row,**result}]).to_csv(index=False).encode('utf-8-sig'),file_name='future_farming_assessment.csv',mime='text/csv')
        except (ValueError,TypeError) as exc:
            st.error(str(exc))
    else:
        st.markdown('<div class="glass"><b>◇ Ready for analysis</b><p>Isi parameter lahan lalu jalankan AI Assessment untuk melihat klasifikasi, grafik skor, serta perkiraan defisit air.</p></div>',unsafe_allow_html=True)

with tab_batch:
    st.markdown('### Batch Intelligence')
    st.caption('Unggah data hingga 5.000 baris dan unduh hasil klasifikasi untuk setiap sampel.')
    st.download_button('↓ Download CSV template',dataset[FEATURES].head(10).to_csv(index=False).encode('utf-8-sig'),file_name='future_farming_template.csv',mime='text/csv')
    uploaded=st.file_uploader('Drag & drop CSV dataset',type=['csv'])
    if uploaded is not None:
        try:
            df=pd.read_csv(uploaded)
            if not 1<=len(df)<=5000: raise ValueError('CSV harus berisi 1–5.000 baris.')
            missing=[f for f in FEATURES if f not in df]
            if missing: raise ValueError('Kolom yang belum ada: '+', '.join(missing))
            for index,raw in df.iterrows():
                try: validate_input(raw)
                except (KeyError,TypeError,ValueError) as e: raise ValueError(f'Baris {index+2}: {e}') from e
            result=df.copy()
            result['status_ai']=model.predict(encode_features(df))
            result['skor_kelas_pct']=(model.predict_proba(encode_features(df)).max(axis=1)*100).round(1)
            result['defisit_net_mm']=[estimate_water_deficit_mm(r)[0] for _,r in df.iterrows()]
            st.success(f'{len(result)} data berhasil dianalisis.')
            counts=result['status_ai'].value_counts().reindex(CATEGORIES,fill_value=0)
            fig=go.Figure(go.Pie(labels=counts.index.tolist(),values=counts.values.tolist(),hole=.66,marker={'colors':['#49dacd','#efc174','#9b77f7']}))
            fig.update_layout(showlegend=True)
            st.plotly_chart(plot_style(fig,310),use_container_width=True,config={'displayModeBar':False})
            st.dataframe(result,use_container_width=True)
            st.download_button('↓ Export predictions (.csv)',result.to_csv(index=False).encode('utf-8-sig'),file_name='future_farming_predictions.csv',mime='text/csv')
        except (ValueError,KeyError,TypeError,pd.errors.ParserError,UnicodeDecodeError) as exc:
            st.error(f'Gagal memproses CSV: {exc}')

with tab_analytics:
    st.markdown('### Model Performance Analytics')
    st.caption('Hasil ini mengukur seberapa baik model meniru label dari aturan simulasi, bukan validitas di lahan nyata.')
    la,lb=st.columns(2,gap='large')
    with la:
        st.markdown('**Feature importance**')
        ranks=pd.Series(model.feature_importances_,index=FEATURES).sort_values()
        fig=go.Figure(go.Bar(x=ranks.values,y=ranks.index,orientation='h',marker={'color':ranks.values,'colorscale':[[0,'#4ad7d3'],[1,'#9d71f7']]}))
        st.plotly_chart(plot_style(fig,370),use_container_width=True,config={'displayModeBar':False})
    with lb:
        st.markdown('**Confusion matrix · test split**')
        fig=go.Figure(go.Heatmap(z=stats['confusion'],x=CATEGORIES,y=CATEGORIES,colorscale=[[0,'#18233f'],[.5,'#5a55b4'],[1,'#4de0d0']],text=stats['confusion'],texttemplate='%{text}',showscale=False))
        fig.update_xaxes(title='Prediksi')
        fig.update_yaxes(title='Aktual',autorange='reversed')
        st.plotly_chart(plot_style(fig,370),use_container_width=True,config={'displayModeBar':False})
    st.markdown('**Distribusi kelas dataset sintetis**')
    counts=dataset['target'].value_counts().reindex(CATEGORIES,fill_value=0)
    fig=go.Figure(go.Bar(x=counts.index,y=counts.values,marker_color=['#4de0d0','#ffc87b','#a78bfa'],text=counts.values,textposition='outside'))
    st.plotly_chart(plot_style(fig,275),use_container_width=True,config={'displayModeBar':False})
    with st.expander('Lihat data sintetis'):
        st.dataframe(dataset.head(40),use_container_width=True)
        st.download_button('↓ Download full synthetic dataset',dataset.to_csv(index=False).encode('utf-8-sig'),file_name='future_farming_synthetic_dataset.csv',mime='text/csv')

with tab_about:
    st.markdown('### About the AI system')
    st.markdown('''<div class="glass"><b>01 / Data Collection</b><p>Dataset berisi 2.400 contoh sintetis dengan variabel kelembapan tanah, kapasitas lapang, titik layu permanen, suhu, prakiraan hujan, kedalaman akar dan fase tanaman.</p></div>
<div class="glass"><b>02 / Model Training</b><p>Random Forest Classifier mempelajari label yang dihasilkan dari aturan simulasi. Dataset dibagi secara stratified: 80% latih dan 20% uji.</p></div>
<div class="glass"><b>03 / AI Prediction</b><p>Model mengklasifikasikan sampel baru menjadi Belum perlu, Pantau, atau Perlu irigasi. Skor kelas bukan probabilitas valid secara klinis/agronomis.</p></div>
<div class="glass"><b>04 / Water Balance</b><p>Defisit net (mm) = max(0, kapasitas lapang − kelembapan tanah) × kedalaman zona akar (cm) / 10. Defisit bruto = defisit net / 0,85. Perhitungan ini terpisah dari model machine learning.</p></div>''',unsafe_allow_html=True)
    st.markdown('**Limitasi penting**')
    st.write('Model belum divalidasi menggunakan sensor terkalibrasi atau pengamatan lapangan; tidak menghitung evapotranspirasi, infiltrasi, curah hujan efektif, maupun waktu irigasi. Tidak boleh dipakai sebagai sistem kendali pompa otomatis tanpa validasi independen.')
    st.write('**Tech stack:** Python · Streamlit · Scikit-learn · Pandas · NumPy · Plotly')

st.markdown('<div class="footer">SMARTIRRIGATION AI · FUTURE FARMING / v2.0 &nbsp; ✦ &nbsp; Educational prototype · Synthetic training data</div>',unsafe_allow_html=True)
