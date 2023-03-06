import pandas as pd
if bbg:
    import pdblp
from typing import Optional
import os
from datetime import datetime, timedelta, date
import sgs
import numpy as np
from tqdm import tqdm
import json
import requests
import urllib.request

class OneDownloadertoDownloadThemAll:

    def __init__(self, tickers_path,start_date="19000101", end_date=(datetime.now()+timedelta(days=90)).strftime("%Y%m%d"),bbg=False,sgs=True,sidra=True,external_sector=True):
        self.tickers_path = tickers_path
        self.start_date = start_date
        self.end_date = end_date
        self.bbg = bbg
        self.sgs = sgs
        self.sidra = sidra
        self.external_sector = external_sector

    def read_series_list(self,):
        tickers_path = self.tickers_path
        series = pd.read_excel(tickers_path)
        series_sgs = series[series["Source"]=="SGS"]
        series_sgs_m = series_sgs[series_sgs["Freq"]=="M"]
        series_sgs_q = series_sgs[series_sgs["Freq"]=="Q"]
        series_bbg = series[series["Source"]=="BBG"]
        series_bbg_m = series_bbg[series_bbg["Freq"]=="M"]
        series_bbg_q = series_bbg[series_bbg["Freq"]=="Q"]
        series_bbg_d = series_bbg[series_bbg["Freq"]=="Intraday"]
        
        return series_sgs_m, series_sgs_q, series_bbg_m, series_bbg_q, series_bbg_d

    def _downloader_sgs_m(self, transform=[]):
        series_sgs_m, series_sgs_q, series_bbg_m, series_bbg_q, series_bbg_d = self.read_series_list()
        start_date, end_date = self.start_date, self.end_date

        tickers = list(series_sgs_m["Code"])
        names = list(series_sgs_m["Name"])
        df_sgs_m = pd.DataFrame()
        for ticker,name in tqdm(zip(tickers,names),total=len(tickers)):
            try:
                df_temp = sgs.dataframe(ticker,start_date,end_date)
                df_temp.columns = [str(name)]
                #print(df_temp)
                df_sgs_m = pd.concat([df_sgs_m,df_temp],axis=1)
            except:
                print("Erro para baixar a série: "+str(ticker)+str(name))
        return df_sgs_m

    def _downloader_sgs_q(self, transform=[]):
        series_sgs_m, series_sgs_q, series_bbg_m, series_bbg_q, series_bbg_d = self.read_series_list()
        start_date, end_date = self.start_date, self.end_date

        tickers = list(series_sgs_q["Code"])
        names = list(series_sgs_q["Name"])
        df_sgs_q = pd.DataFrame()
        for ticker,name in tqdm(zip(tickers,names),total=len(tickers)):
            try:
                df_temp = sgs.dataframe(ticker,start_date,end_date)
                df_temp.columns = [str(name)]
                #print(df_temp)
                df_sgs_q = pd.concat([df_sgs_q,df_temp],axis=1)
            except:
                print("Erro para baixar a série: "+str(ticker)+str(name))
        return df_sgs_q

    def _downloader_bbg_m(self, transform=[]):
        series_sgs_m, series_sgs_q, series_bbg_m, series_bbg_q, series_bbg_d = self.read_series_list()
        start_date, end_date = self.start_date, self.end_date

        tickers = list(series_bbg_m["Code"])
        client = pdblp.BCon(timeout=5000000, debug = False)
        client.start()
        field = ["PX_LAST"]
        df_bbg_m = client.bdh(tickers, flds = field,start_date = start_date,end_date = end_date)
        df_bbg_m.columns = [col[0] for col in df_bbg_m.columns]
        df_bbg_m.columns = df_bbg_m.columns.map(dict(zip(series_bbg_m["Code"],series_bbg_m["Name"])))
        return df_bbg_m

    def _downloader_bbg_q(self, transform=[]):
        series_sgs_m, series_sgs_q, series_bbg_m, series_bbg_q, series_bbg_d = self.read_series_list()
        start_date, end_date = self.start_date, self.end_date

        tickers = list(series_bbg_q["Code"])
        client = pdblp.BCon(timeout=5000000, debug = False)
        client.start()
        field = ["PX_LAST"]
        df_bbg_q = client.bdh(tickers, flds = field,start_date = start_date,end_date = end_date)
        df_bbg_q.columns = [col[0] for col in df_bbg_q.columns]
        df_bbg_q.columns = df_bbg_q.columns.map(dict(zip(series_bbg_q["Code"],series_bbg_q["Name"])))
        return df_bbg_q

    def _downloader_bbg_d(self, transform=[]):
        series_sgs_m, series_sgs_q, series_bbg_m, series_bbg_q, series_bbg_d = self.read_series_list()
        start_date, end_date = self.start_date, self.end_date

        start_date = "20000101"
        tickers = list(series_bbg_d["Code"])
        client = pdblp.BCon(timeout=5000000, debug = False)
        client.start()
        field = ["PX_LAST"]
        df_bbg_d = client.bdh(tickers, flds = field,start_date = start_date,end_date = end_date)
        df_bbg_d.columns = [col[0] for col in df_bbg_d.columns]
        df_bbg_d.columns = df_bbg_d.columns.map(dict(zip(series_bbg_d["Code"],series_bbg_d["Name"])))
        return df_bbg_d

    def sidra_pms(self,full=False):
        if full:
            url = "https://apisidra.ibge.gov.br/values/t/8162/n1/all/v/all/p/all/c11046/all/c12355/all/d/v11621%202,v11622%202"#,v11623%201,v11624%201,v11625%201,v11626%201"
        else:
            url = "https://apisidra.ibge.gov.br/values/t/8161/n1/all/v/11621,11622/p/all/c11046/all/d/v11621%202,v11622%202"

        response = requests.get(url)
        json_content = json.loads(response.content)

        df = pd.DataFrame(json_content)
        df.columns = df.iloc[0]
        df = df[1:]

        if full:
            df_pivot = pd.pivot_table(df, values="Valor",columns=["Variável","Tipos de índice","Atividades de serviços"],index="Mês (Código)",aggfunc="first")
        else:
            df_pivot = pd.pivot_table(df, values="Valor",columns=["Variável","Tipos de índice"],index="Mês (Código)",aggfunc="first")
        df_pivot = df_pivot.apply(pd.to_numeric, errors="coerce")
        df_pivot.index = pd.to_datetime(df_pivot.index,format='%Y%m')
        return df_pivot

    def sidra_pim(self):

        url = "https://apisidra.ibge.gov.br/values/t/8158/n1/all/v/11599,11600/p/all/c543/129278,129283,129300,129301,129305,129311/d/v11599%205,v11600%205"

        response = requests.get(url)
        json_content = json.loads(response.content)

        df = pd.DataFrame(json_content)
        df.columns = df.iloc[0]
        df = df[1:]

        df_pivot = pd.pivot_table(df, values="Valor",columns=["Variável","Grandes categorias econômicas"],index="Mês (Código)",aggfunc="first")#,"Atividades de serviços"]
        #df_pivot = pd.pivot_table(df, values="Valor",index=["Variável","Tipos de índice","Atividades de serviços"],columns="Mês (Código)",aggfunc="first")
        df_pivot = df_pivot.apply(pd.to_numeric, errors="coerce")
        df_pivot.index = pd.to_datetime(df_pivot.index,format='%Y%m')
        
        return df_pivot

    def sidra_pmc(self,ampliado=False):
        if ampliado:
            url = "https://apisidra.ibge.gov.br/values/t/8188/n1/all/v/11706,11707/p/all/c11046/all/c85/2759,90671,90672,90673,103155,103156,103157,103158/d/v11706%205,v11707%205"
        else:
            url = "https://apisidra.ibge.gov.br/values/t/8187/n1/all/v/11706,11707/p/all/c11046/all/c85/2759,90671,90672,90673,103155,103156,103157,103158/d/v11706%205,v11707%205"
        
        response = requests.get(url)
        json_content = json.loads(response.content)

        df = pd.DataFrame(json_content)
        df.columns = df.iloc[0]
        df = df[1:]

        df_pivot = pd.pivot_table(df, values="Valor",columns=["Variável","Tipos de índice","Atividades"],index="Mês (Código)",aggfunc="first")#,"Atividades de serviços"]
        df_pivot = df_pivot.apply(pd.to_numeric, errors="coerce")
        df_pivot.index = pd.to_datetime(df_pivot.index,format='%Y%m')
        
        return df_pivot

    def sidra_ipca_mom(self,dessaz=True):
        if dessaz:
            url = "https://apisidra.ibge.gov.br/values/t/118/n1/all/v/all/p/all/d/v306%202"
        else:
            url = "https://apisidra.ibge.gov.br/values/t/1737/n1/all/v/63/p/all/d/v63%202"
        
        response = requests.get(url)
        json_content = json.loads(response.content)

        df = pd.DataFrame(json_content)
        df.columns = df.iloc[0]
        df = df[1:]

        df_pivot = pd.pivot_table(df, values="Valor",columns=["Variável"],index="Mês (Código)",aggfunc="first")#,"Atividades de serviços"]
        df_pivot = df_pivot.apply(pd.to_numeric, errors="coerce")
        df_pivot.index = pd.to_datetime(df_pivot.index,format='%Y%m')

        return df_pivot

    def sidra_pnad_trabalho(self):
        url = "https://apisidra.ibge.gov.br/values/t/5434/n1/all/v/all/p/all/c693/all/d/v4091%201,v4108%201,v4109%201"

        response = requests.get(url)
        json_content = json.loads(response.content)

        df = pd.DataFrame(json_content)
        df.columns = df.iloc[0]
        df = df[1:]

        df_pivot = pd.pivot_table(df, values="Valor",columns=["Grupamento de atividades no trabalho principal - PNADC","Variável"],index="Trimestre",aggfunc="first")#,"Atividades de serviços"]
        df_pivot = df_pivot.apply(pd.to_numeric, errors="coerce")
        df_pivot.index = [x.replace("1º trimestre ","03/31/").replace("2º trimestre ","06/30/").replace("3º trimestre ","09/30/").replace("4º trimestre ","12/31/") for x in df_pivot.index]
        df_pivot.index = pd.to_datetime(df_pivot.index)
        df_pivot.sort_index(inplace=True)

        return df_pivot

    def bot(self):
        '''
        Function to update the Balance of trade data directly from government source."
        '''
        cuci_data = "https://balanca.economia.gov.br/balanca/SH/CUCI.xlsx"
        save_as = 'data/aux_files/CUCI.xlsx'
        urllib.request.urlretrieve(cuci_data, filename = save_as)
        
        df = pd.read_excel(save_as, sheet_name = 'DADOS_SH_CUCI')
        df['reference_date'] = df[['CO_ANO', 'CO_MES']].astype(str).agg('-'.join, axis=1)
        df['reference_date'] = pd.to_datetime(df['reference_date'], format = '%Y-%m')
        df['reference_date'] = df['reference_date'] + pd.offsets.MonthEnd(0)
        df['Product'] = df[['NO_CUCI_SEC', 'TIPO']].astype(str).agg('_'.join, axis=1)
        df = df.groupby(['reference_date', 'Product']).agg('sum',numeric_only=True)[['US$ VL_FOB']]
        df_pivot = pd.pivot_table(df, values = 'US$ VL_FOB', index = 'reference_date', columns = 'Product')
        #with open(f'BOT.pickle', 'wb') as fp:
        #    pickle.dump(self.bot_data, fp)
        return df_pivot

    def tot(self):

        url = "https://balanca.economia.gov.br/balanca/IPQ/arquivos/PRECO_TERMOS_DE_TROCA.xlsx"
        save_as = 'data/aux_files/precotermosdetroca.xlsx'
        urllib.request.urlretrieve(url, filename = save_as)

        df = pd.read_excel(save_as)

        return df

    def saver(self):
        sgs = self.sgs
        bbg = self.bbg
        sidra = self.sidra
        external_sector = self.external_sector
        series_sgs_m, series_sgs_q, series_bbg_m, series_bbg_q, series_bbg_d = self.read_series_list()
        if sgs:
            print("Downloading "+str(len(series_sgs_m["Code"]))+" Monthly Time Series from Brazil Central Bank SGS")
            df1 = self._downloader_sgs_m()
            df1.to_pickle("data/pickle/sgsm.pkl")
            with pd.ExcelWriter("data/rawdata_excel.xlsx", engine='openpyxl', mode='a',if_sheet_exists="replace") as writer:  
                df1.to_excel(writer, sheet_name='sgs_monthly')

            print("Downloading "+str(len(series_sgs_q["Code"]))+" Quarterly Time Series from Brazil Central Bank SGS")
            df2 = self._downloader_sgs_q()
            df2.to_pickle("data/pickle/sgsq.pkl")
            with pd.ExcelWriter("data/rawdata_excel.xlsx", engine='openpyxl', mode='a',if_sheet_exists="replace") as writer:  
                df2.to_excel(writer, sheet_name='sgs_quarterly')

        if bbg:
            print("Downloading "+str(len(series_bbg_m["Code"]))+" Monthly Time Series from Bloomberg")
            df3 = self._downloader_bbg_m()
            df3.to_pickle("data/pickle/bbgm.pkl")
            with pd.ExcelWriter("data/rawdata_excel.xlsx", engine='openpyxl', mode='a',if_sheet_exists="replace") as writer:  
                df3.to_excel(writer, sheet_name='bbg_monthly')

            print("Downloading "+str(len(series_bbg_q["Code"]))+" Quarterly Time Series from Bloomberg")
            df4 = self._downloader_bbg_q()
            df4.to_pickle("data/pickle/bbgq.pkl")
            with pd.ExcelWriter("data/rawdata_excel.xlsx", engine='openpyxl', mode='a',if_sheet_exists="replace") as writer:  
                df4.to_excel(writer, sheet_name='bbg_quarterly')

            print("Downloading "+str(len(series_bbg_d["Code"]))+" Daily Time Series from Bloomberg")
            df5 = self._downloader_bbg_d()
            df5.to_pickle("data/pickle/bbgd.pkl")
            with pd.ExcelWriter("data/rawdata_excel.xlsx", engine='openpyxl', mode='a',if_sheet_exists="replace") as writer:  
                df5.to_excel(writer, sheet_name='bbg_daily')

        if sidra:
            print("Downloading IBGE data from SIDRA api")

            pms = self.sidra_pms()
            pms.to_pickle("data/pickle/pms.pkl")
            with pd.ExcelWriter("data/rawdata_excel.xlsx", engine='openpyxl', mode='a',if_sheet_exists="replace") as writer:  
                pms.to_excel(writer, sheet_name='sidra_pms')
            pms_full = self.sidra_pms(full=True)
            pms_full = pms_full["PMS - Número-índice com ajuste sazonal (2014=100)"]
            pms_full.to_pickle("data/pickle/pms_full.pkl")
            with pd.ExcelWriter("data/rawdata_excel.xlsx", engine='openpyxl', mode='a',if_sheet_exists="replace") as writer:  
                pms_full.to_excel(writer, sheet_name='sidra_pms_full')
            print("PMS Sidra Data Downloaded")

            pim = self.sidra_pim()
            pim.to_pickle("data/pickle/pim.pkl")
            with pd.ExcelWriter("data/rawdata_excel.xlsx", engine='openpyxl', mode='a',if_sheet_exists="replace") as writer:  
                pim.to_excel(writer, sheet_name='sidra_pim')
            print("PIM Sidra Data Downloaded")

            pmc = self.sidra_pmc()
            pmc.to_pickle("data/pickle/pmc.pkl")
            with pd.ExcelWriter("data/rawdata_excel.xlsx", engine='openpyxl', mode='a',if_sheet_exists="replace") as writer:  
                pmc.to_excel(writer, sheet_name='sidra_pmc')
            pmc_ampliado = self.sidra_pmc(ampliado=True)
            pmc_ampliado.to_pickle("data/pickle/pmc_ampliado.pkl")
            with pd.ExcelWriter("data/rawdata_excel.xlsx", engine='openpyxl', mode='a',if_sheet_exists="replace") as writer:  
                pmc_ampliado.to_excel(writer, sheet_name='sidra_pmc_ampliado')
            print("PMC Sidra Data Downloaded")

            ipca = self.sidra_ipca_mom()
            ipca.to_pickle("data/pickle/ipca.pkl")
            with pd.ExcelWriter("data/rawdata_excel.xlsx", engine='openpyxl', mode='a',if_sheet_exists="replace") as writer:  
                ipca.to_excel(writer, sheet_name='sidra_ipca')
            ipca_nsa = self.sidra_ipca_mom(dessaz=False)
            ipca_nsa.to_pickle("data/pickle/ipca_nsa.pkl")
            with pd.ExcelWriter("data/rawdata_excel.xlsx", engine='openpyxl', mode='a',if_sheet_exists="replace") as writer:  
                ipca_nsa.to_excel(writer, sheet_name='sidra_ipca_nsa')
            print("IPCA Sidra Data Downloaded")

            pnad = self.sidra_pnad_trabalho()
            pnad.to_pickle("data/pickle/pnad.pkl")
            with pd.ExcelWriter("data/rawdata_excel.xlsx", engine='openpyxl', mode='a',if_sheet_exists="replace") as writer:  
                pnad.to_excel(writer, sheet_name='sidra_pnad')
            print("PNAD Sidra Data Downloaded")

        if external_sector:
            #print("")
            print("\nDownloading External Sector Data from MDIC")
            bot = self.bot()
            bot.to_pickle("data/pickle/bot.pkl")
            with pd.ExcelWriter("data/rawdata_excel.xlsx", engine='openpyxl', mode='a',if_sheet_exists="replace") as writer:  
                bot.to_excel(writer, sheet_name='bot')
            tot = self.tot()
            tot.to_pickle("data/pickle/tot.pkl")
            with pd.ExcelWriter("data/rawdata_excel.xlsx", engine='openpyxl', mode='a',if_sheet_exists="replace") as writer:  
                tot.to_excel(writer, sheet_name='tot')
            

        #if bbg
        #return df1, df2, df3 if bbg==True else df1, df2
        #df.to_excel("bbg.xlsx")
        #, df3