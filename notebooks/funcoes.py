import os

import mysql.connector
from sqlalchemy import create_engine

import pandas as pd
import numpy as np

import datetime
import holidays

import pickle

from sklearn.preprocessing import LabelEncoder

def baixar_arquivo(user, password, host, database, port, data_minima, data_maxima, arquivo_bruto):

    USER = user
    PASSWORD = password
    HOST = host
    DATABASE = database
    PORT = port

    string_conector = f"mysql+mysqlconnector://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}"

    motor = create_engine(string_conector)

    query = f"""
WITH dados_unificados AS (
    	SELECT appId, date FROM daumau
    	UNION
    	SELECT appId, DATE(date) AS date from desinstalacoes
    	UNION
    	SELECT appId AS appid, DATE(date) AS date from installs
    	UNION
    	SELECT appid AS appid, date from ratings_reviews
)

SELECT
    du.appId,
    du.date,

    dau.dauReal,
    dau.mauReal,

    des.country,
    des.lang,
    des.predictionLoss,

    ins.newinstalls,

    rat.category,
    rat.ratings,
    rat.daily_ratings,
    rat.reviews,
    rat.daily_reviews
    
FROM dados_unificados AS du
    LEFT JOIN daumau AS dau
        ON dau.appId = du.appId AND dau.date = du.date
    LEFT JOIN desinstalacoes AS des
        ON des.appId = du.appId AND DATE(des.date) = du.date
    LEFT JOIN installs AS ins
        ON ins.appid = du.appId AND DATE(ins.date) = du.date
    LEFT JOIN ratings_reviews as rat
        ON rat.appid = du.appId AND rat.date = du.date

WHERE du.date >= "{data_minima}" AND du.date <= "{data_maxima}"
"""

    df = pd.read_sql(query, motor)
    motor.dispose()
    
    pasta_atual = os.getcwd()
    pasta_pai = os.path.dirname(pasta_atual)
    pasta_data = os.path.join(pasta_pai, "data")

    df.to_csv(f"{pasta_data}\\{arquivo_bruto}", index=False)
    print(f"Arquivo gerado: {arquivo_bruto}")


def limpar_dados(arquivo_bruto, arquivo_limpo):
    # agora que temos os dados armazenados, podemos criar o nosso dataframe e seguir com as análises e limpeza
    pasta_atual = os.getcwd()
    pasta_pai = os.path.dirname(pasta_atual)
    pasta_data = os.path.join(pasta_pai, "data")

    df = pd.read_csv(f"{pasta_data}\\{arquivo_bruto}", parse_dates=["date"])\
        .sort_values(by=["date", "appId"])\
        .reset_index(drop=True)

    df = df.loc[~df["appId"].isna()]

    df = df.loc[df["date"].dt.year == 2024]

    df = df.drop_duplicates()

    periodo_completo = pd.date_range(start=df["date"].min(), end=df["date"].max())

    app_ids = df["appId"].unique()

    todas_combinacoes = pd.MultiIndex.from_product([app_ids, periodo_completo], names=["appId", "date"])

    df = df.set_index(["appId", "date"]).reindex(todas_combinacoes).reset_index()

    df = df.drop(["country", "lang"], axis=1)

    mapa = df.dropna(subset=["category"]).drop_duplicates("appId").set_index("appId")["category"].to_dict()

    df["category"] = df.apply(
        lambda linha: mapa.get(linha["appId"], "NO_CATEGORY") if pd.isna(linha["category"]) else linha["category"],
        axis=1
    )

    df[["ratings", "reviews"]] = df.groupby("appId")[["ratings", "reviews"]]\
        .transform(lambda x: x.ffill())

    df[["ratings", "reviews"]] = df[["ratings", "reviews"]].fillna(0)

    df["daily_ratings"] = df.groupby("appId")["ratings"].shift(-1) - df["ratings"] 
    df["daily_reviews"] = df.groupby("appId")["reviews"].shift(-1) - df["reviews"]

    df[["daily_ratings", "daily_reviews"]] = df.groupby("appId")[["daily_ratings", "daily_reviews"]]\
        .transform(lambda x: x.ffill())

    df[["mauReal", "predictionLoss"]] = df[["mauReal", "predictionLoss"]].interpolate(method="linear")

    df["mauReal"] = df.groupby("appId")["mauReal"]\
        .transform(lambda x: x.ffill())

    df["mauReal"] = df.groupby("appId")["mauReal"]\
        .transform(lambda x: x.bfill())

    df["newinstalls"] = df["newinstalls"].fillna(0)

    filtro = df["mauReal"] > 0
    df.loc[filtro, "dauReal"] = df["dauReal"].where(filtro).interpolate()

    df["dauReal"] = df.groupby("appId")["dauReal"]\
        .transform(lambda x: x.bfill())

    df.to_csv(f"{pasta_data}\\{arquivo_limpo}", index=False)
    print(f"Arquivo gerado: {arquivo_limpo}")


def extracao_de_caracteristicas(arquivo_limpo, arquivo_tratado):

    pasta_atual = os.getcwd()
    pasta_pai = os.path.dirname(pasta_atual)
    pasta_data = os.path.join(pasta_pai, "data")

    df = pd.read_csv(f"{pasta_data}\\{arquivo_limpo}", parse_dates=["date"])\
        .sort_values(by=["date", "appId"])\
        .reset_index(drop=True)

    feriados_brasil = holidays.Brazil()

    def validar_se_dia_trabalho(data):
        ser_dia_semana = data.weekday() < 5
        ser_feriado = data in feriados_brasil

        return ser_dia_semana and not ser_feriado

    df["dia_de_trabalho"] = df["date"].apply(validar_se_dia_trabalho)

    def codificacao_ciclica(aux_df):
        df = aux_df.copy()

        timestamp = df["date"].map(pd.Timestamp.timestamp)

        dia = 24 * 60 * 60
        semana = dia * 7
        ano = 365.2425 * dia
        trimestre = ano / 4
        meia_decada = ano * 5

        df["sin_semana"] = np.sin(timestamp * (2 * np.pi / semana))
        df["cos_semana"] = np.cos(timestamp * (2 * np.pi / semana))

        df["sin_trimestre"] = np.sin(timestamp * (2 * np.pi / trimestre))
        df["cos_trimestre"] = np.cos(timestamp * (2 * np.pi / trimestre))

        df["sin_ano"] = np.sin(timestamp * (2 * np.pi / ano))
        df["cos_ano"] = np.cos(timestamp * (2 * np.pi / ano))

        df["sin_meia_decada"] = np.sin(timestamp * (2 * np.pi / meia_decada))
        df["cos_meia_decada"] = np.cos(timestamp * (2 * np.pi / meia_decada))

        return df

    df = codificacao_ciclica(df)

    def adicao_dados_moveis(aux_df, colunas, dias_janela):
        df = aux_df.copy()

        for coluna in colunas:
            df[f"{coluna}_media_{dias_janela}"] = df.groupby("appId")[coluna]\
                .rolling(window = dias_janela, min_periods = 1)\
                .mean()\
                .reset_index(level=0, drop=True)

            df[f"{coluna}_dp_{dias_janela}"] = df.groupby("appId")[coluna]\
                .rolling(window = dias_janela, min_periods = 1)\
                .std()\
                .reset_index(level=0, drop=True)\
                .fillna(0)

        return df

    colunas_dados_moveis = ["dauReal", "mauReal", "newinstalls", "ratings", "reviews"]

    df = adicao_dados_moveis(df, colunas = colunas_dados_moveis, dias_janela = 3)
    df = adicao_dados_moveis(df, colunas = colunas_dados_moveis, dias_janela = 7)

    df["anterior_dauReal"] = df.groupby("appId")["dauReal"].shift(+1)
    df["proximo_dauReal"] = df.groupby("appId")["dauReal"].shift(-1)

    df = pd.get_dummies(df, columns=['category'])

    pasta_atual = os.getcwd()
    pasta_pai = os.path.dirname(pasta_atual)
    pasta_data = os.path.join(pasta_pai, "data")

    with open(f"{pasta_data}\\label_encoder.pkl", 'rb') as f:
        le = pickle.load(f)
        df["cod_appId"] = le.fit_transform(df["appId"])

    df = df.dropna(subset=["proximo_dauReal", "anterior_dauReal"])

    df.to_csv(f"{pasta_data}\\{arquivo_tratado}", index=False)
    print(f"Arquivo gerado: {arquivo_tratado}")
