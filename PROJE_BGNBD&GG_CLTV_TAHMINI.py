##############################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
##############################################################

# 1. Verinin Hazırlanması (Data Preperation)
# 2. BG-NBD Modeli ile Expected Sales Forecasting
# 3. Gamma-Gamma Modeli ile Expected Average Profit
# 4. BG-NBD ve Gamma-Gamma Modeli ile CLTV'nin Hesaplanması
# 5. CLTV'ye Göre Segmentlerin Oluşturulması
# 6. Çalışmanın fonksiyonlaştırılması
# 7. Sonuçların Veri Tabanına Gönderilmesi

# CLTV = (Customer_Value / Churn_Rate) x Profit_margin.
# Customer_Value = Average_Order_Value * Purchase_Frequency

##############################################################
# 1. Verinin Hazırlanması (Data Preperation)
##############################################################

# Gerekli Kütüphane ve Fonksiyonlar

pip install lifetimes
pip install sqlalchemy
pip install mysql-connector-python
conda install -c conda-forge mysql


from sqlalchemy import create_engine
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# Verinin Excel'den Okunması

df_ = pd.read_excel("pythonProject/datasets/online_retail_II.xlsx", sheet_name="Year 2009-2010")
df = df_.copy()

df.shape

# Veri Ön İşleme
# Ön İşleme Öncesi
df.describe().T
df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

df.describe().T

df["TotalPrice"] = df["Quantity"] * df["Price"]

today_date = dt.datetime(2011, 12, 11)

# Lifetime Veri Yapısının Hazırlanması

# recency: Son satın alma üzerinden geçen zaman. Haftalık. (daha önce analiz gününe göre, burada kullanıcı özelinde)
# T: Analiz tarihinden ne kadar süre önce ilk satın alma yapılmış. Haftalık.
# frequency: tekrar eden toplam satın alma sayısı (frequency>1)
# monetary_value: satın alma başına ortalama kazanç


cltv_df = df.groupby('Customer ID').agg({'InvoiceDate': [lambda date: (date.max() - date.min()).days,
                                                         lambda date: (today_date - date.min()).days],
                                         'Invoice': lambda num: num.nunique(),
                                         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']


cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

cltv_df = cltv_df[(cltv_df['frequency'] > 1)

cltv_df["recency"] = cltv_df["recency"] / 7

cltv_df["T"] = cltv_df["T"] / 7

# 2. BG-NBD Modelinin Kurulması

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])

# 1 Hafta içinde tüm Şirketin Beklenen Satış Sayısı

cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])

# 1 Ay içinde tüm Şirketin Beklenen Satış Sayısı

cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])

cltv_df.head()

# 3. GAMMA-GAMMA Modelinin Kurulması

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary'])


ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).head(10)


ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).sort_values(ascending=False).head(10)

cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df['monetary'])


# 4. BG-NBD ve GG modeli ile CLTV'nin hesaplanması.
# Görev - 1
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=6,  # 6 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

cltv.head()


cltv = cltv.reset_index()
cltv.sort_values(by="clv", ascending=False).head(50)

cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")

cltv_final.sort_values(by="clv", ascending=False).head(10)

#0-1 arası Transform
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(cltv_final[["clv"]])
cltv_final["scaled_clv"] = scaler.transform(cltv_final[["clv"]])


cltv_final.sort_values(by="scaled_clv", ascending=False).head()

# GÖREV 2

# 1. 2010-2011 UK müşterileri için 1 aylık ve 12 aylık CLTV hesaplayınız.

cltv1 = ggf.customer_lifetime_value(bgf,
                                    cltv_df['frequency'],
                                    cltv_df['recency'],
                                    cltv_df['T'],
                                    cltv_df['monetary'],
                                    time=1,  # months
                                    freq="W",  # T haftalık
                                    discount_rate=0.01)

rfm_cltv1_final = cltv_df.merge(cltv1, on="Customer ID", how="left")
rfm_cltv1_final.sort_values(by="clv", ascending=False).head(10)


cltv12 = ggf.customer_lifetime_value(bgf,
                                     cltv_df['frequency'],
                                     cltv_df['recency'],
                                     cltv_df['T'],
                                     cltv_df['monetary'],
                                     time=12,  # months
                                     freq="W",  # T haftalık
                                     discount_rate=0.01)

rfm_cltv12_final = cltv_df.merge(cltv12, on="Customer ID", how="left")
rfm_cltv12_final.sort_values(by="clv", ascending=False).head(10)

rfm_cltv12_final.head()

# 2. 1 aylık CLTV'de en yüksek olan 10 kişi ile 12 aylık'taki en yüksek 10 kişiyi analiz ediniz. Fark var mı? Varsa sizce neden olabilir?

rfm_cltv1_final.sort_values("clv", ascending=False).head(10)
rfm_cltv12_final.sort_values("clv", ascending=False).head(10)

# GÖREV 3

# 1. 2010-2011 UK müşterileri için 6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz.

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=6,  # months
                                   freq="W",  # T haftalık
                                   discount_rate=0.01)

cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
cltv_final.head()

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(cltv_final[["clv"]])
cltv_final["SCALED_CLTV"] = scaler.transform(cltv_final[["clv"]])

cltv_final["cltv_segment"] = pd.qcut(cltv_final["SCALED_CLTV"], 4, labels=["D", "C", "B", "A"])
cltv_final["cltv_segment"].value_counts()
cltv_final.head()


cltv_final.groupby("cltv_segment")[["expected_purc_1_month", "expected_average_profit", "clv", "SCALED_CLTV"]].agg(
    {"count", "mean", "sum"})

cltv_final.groupby("cltv_segment").agg({"mean"})

# Segment A: Bu segmentteki müsteriler en büyük CLTV degerlerine sahiptirler.
# Buna paralel olarak toplam harcamalari, islem sayisi ve alisveris sayilari en yüksektir.
# 391391.92521  beklenen ortalama karlılık  olarak tarafımıza iletilirken, 643 adet işlem yapılmıştır.
# Burada işlem başına yaklaşık 608 birim para karlılık gösteriyor.

# GÖREV 4

# Aşağıdaki değişkenlerden oluşacak final tablosunu veri tabanına gönderiniz.
# tablonun adını isim_soyisim şeklinde oluşturunuz.
# Tablo ismi ilgili fonksiyonda "name" bölümüne girilmelidir.

# Customer ID, recency, T, frequency, monetary, expected_purc_1_week, expected_purc_1_month, expected_average_profit
# clv, scaled_clv, segment

cltv_final = cltv_final.reset_index()
cltv_final["Customer ID"] = cltv_final["Customer ID"].astype(int)