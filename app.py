import streamlit as st
import pandas as pd
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder


def converter_mileage(x, fuel):
  if pd.isna(x):
    return x

  val, units = x.split()
  if units == 'kmpl':
    return float(val)
  elif units == 'km/kg':
    if fuel == 'LPG':
      return float(val) * (1/520)
    elif fuel == 'CNG':
      return float(val) * (1/190)
    return float('nan')

def parse(s, col=0):
  if pd.isna(s):
    return s
  s = s.lower()
  words = []
  w = ""
  g = 'd'
  for i in s:
    if i in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '.', '~']:
      if g != 'd' and w != "":
          words.append(w)
          w =""
      g = 'd'
      w += i
    elif i.isalpha():
      if g != 'a' and w != "":
          words.append(w)
          w = ""
      g = 'a'
      w += i
    elif i in [' ', '@']:
      if w != "":
        words.append(w)
      w = ""
    elif i == ',':
      pass
    else:
      if w != "":
        words.append(w)
      w = ""
  if w != "":
      words.append(w)
  if 'at' in words:
    words.remove('at')

  torque = float(words[0])
  if 'kgm' in words:
    torque *= 9.80665
  rpm = float('nan')
  if any(c.isdigit() for c in words[1]):
    rpm = words[1]
  elif len(words) >= 3 and any(c.isdigit() for c in words[2]):
    rpm = words[2]
  else:
    return [torque, rpm][col]
  if '+/-' in s:
    w = ""
    for i in range(s.find('+/-'), len(s)):
      if s[i].isdigit():
        w += s[i]
    rpm = float(rpm)
    rpm = rpm + float(w)
  elif '-' in rpm:
    rpm = max(float(rpm.split('-')[0]), float(rpm.split('-')[1]))
  elif '~' in rpm:
    rpm = max(float(rpm.split('~')[0]), float(rpm.split('~')[1]))
  else:
    rpm = float(rpm)
  return [torque, rpm][col]

def GenFeatures(df):
  df['year2'] = df['year']**2
  df['engine2'] = df['engine']**2
  df['torque_log'] = np.log(df['torque'])

  df['age'] = 2025 - df['year']
  df['age2'] = df['age']**2

  df['power_per_volue'] = df['max_power'] / df['engine']
  df['torque_per_volue'] = df['torque'] / df['engine']
  df['torque+power'] = df['torque'] * df['max_power']

def AddNewFeatures(df):
  # этим нужно было заниматься раньше. Сейчас данные находятся в не очень удачном формате
  df['good_choice'] = ((df['seller_type_Individual'] == 1) & (df['owner_Fourth & Above Owner'] == 0)
      & (df['owner_Test Drive Car'] == 0) & (df['owner_Third Owner'] == 0))

def prepare(df, brands, models, enc):
  df['engine'] = df['engine'].str.replace(' CC', '').astype(float)
  df['max_power'] = df['max_power'].str.split().str[0]
  df['max_power'] = pd.to_numeric(df['max_power'], errors='coerce')
  df['mileage'] = df.apply(lambda row: converter_mileage(row['mileage'], row['fuel']), axis=1)
  df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')
  df['max_torque_rpm'] = df['torque'].apply(lambda x: parse(x, 1))
  df['torque'] = df['torque'].apply(lambda x: parse(x, 0))

  df['mileage'] = df['mileage'].fillna(df['mileage'].median())
  df['engine'] = df['engine'].fillna(df['engine'].median())
  df['max_power'] = df['max_power'].fillna(df['max_power'].median())
  df['torque'] = df['torque'].fillna(df['torque'].median())
  df['seats'] = df['seats'].fillna(df['seats'].median())
  df['max_torque_rpm'] = df['max_torque_rpm'].fillna(df['max_torque_rpm'].median())

  df['seats'] = df['seats'].astype(int)
  df['engine'] = df['engine'].astype(int)

  df['model'] = df['name'].str.split().str[1]
  df['brand'] = df['name'].str.split().str[0]
  df = df.drop('name', axis=1)
  df = df[df['brand'].isin(brands) & df['model'].isin(models)]
  df = df.reset_index(drop=True)

  columns = ['model', 'brand', 'owner', 'transmission', 'seller_type', 'fuel', 'seats']
  coded = enc.transform(df[columns])
  df = pd.concat([df.drop(columns=columns, axis=1), pd.DataFrame(coded, columns=enc.get_feature_names_out(columns))], axis=1)

  GenFeatures(df)
  AddNewFeatures(df)

  df = df[df['year'] > 1995]
  df = df[(df['max_power'] > 25) & (df['max_power'] < 300)]
  df = df[df['km_driven'] < 0.5 * 1e6]
  df = df[df['torque'] < 800]

  y = df['selling_price']
  df = df.drop(columns=['mileage', 'max_torque_rpm', 'selling_price'], axis=1)

  return (df, y)





st.title("Предсказание стоимости автомобиля")
uploaded_file = st.file_uploader("Выберите csv файл с данными для прогнозирования", type=['csv'])

with open('to_export.pkl', 'rb') as f:
    import_pickle = pickle.load(f)
scaler = import_pickle['scaler']
model = import_pickle['model']
known_brands = import_pickle['known_brands']
known_models = import_pickle['known_models']
encoder = import_pickle['encode']

X = pd.DataFrame()
y = pd.DataFrame()
  
tab_csv, tab_eda, tab_model_w, tab_model = st.tabs(["Предпросмотр данных", "EDA", "Веса модели", "Прогноз модели"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    X, y = prepare(df, known_brands, known_models, encoder)
    X = pd.DataFrame(scaler.transform(X), columns=X.columns)
    
    with tab_csv:
        st.write("Данные предобработаны. Обратонаны категориальные признаки и применен скалер.")
        st.dataframe(X)

    with tab_eda:
        st.write("Графики построены для скалированных данных")
        
        st.subheader("Heatmap")
        df_to_corr = X.copy()
        df_to_corr['selling_price'] = y
        corr = df_to_corr[['selling_price', 'year', 'km_driven', 'engine', 'max_power', 'torque']].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, ax=ax, annot=True)
        ax.set_title("Матрица корреляции")
        st.pyplot(fig)

        st.subheader("Pairplot")
        pdf = df_to_corr[['selling_price', 'year', 'km_driven', 'engine', 'max_power', 'torque']]
        pairplot = sns.pairplot(pdf)
        st.pyplot(pairplot.figure)

    with tab_model_w:
        st.subheader("График весов модели")
        fig, ax = plt.subplots(figsize=(8, 48))
        sns.barplot(x=model.coef_, y=X.columns, ax=ax)
        ax.set_ylabel('')
        ax.set_title("Веса модели")
        st.pyplot(fig)

    with tab_model:
        st.subheader("Lineplot")
        y_pred = model.predict(X)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.lineplot(y_pred, ax=ax, label="предсказание")
        sns.lineplot(y, ax=ax, label="целевое")
        ax.set_ylabel("Значение целевой переменной")
        ax.set_title("График целевой переменной")
        st.pyplot(fig)

        st.subheader("KDEplot")
        fig, ax =  plt.subplots(figsize=(8, 6))
        sns.kdeplot(y_pred, ax=ax, label="предсказание")
        sns.kdeplot(y, ax=ax, label="целевое")
        ax.set_title("Распределение данных")
        plt.legend()
        st.pyplot(fig)

        resudials = y_pred - y
        st.subheader("График остатков")
        percent = (y_pred - y) / y * 100
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(resudials, ax=ax)
        ax.set_title("Остатки")
        st.pyplot(fig)

        st.subheader("Процентное отклонение")
        percent = resudials / y * 100
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(percent, ax=ax)
        ax.set_title("Процентное отклонение")
        st.pyplot(fig)

        st.subheader("Значения")
        ydf = pd.DataFrame({
            "Предсказанное значение": y_pred,
            "Целевое значение": y,
            "Процентное отклонение": percent,
            "Остаток": resudials,
        })
        st.dataframe(ydf)

