import hana_ml 
from hana_ml import dataframe
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller
from hana_ml.algorithms.pal.tsa.vector_arima import VectorARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm


# conn = dataframe.ConnectionContext('e8bce757-9d85-4c9e-96b5-2a44a42dc6c0.hana.trial-us10.hanacloud.ondemand.com', '443', 'DBADMIN', 'TestPass1$')

def adfuller_test(series, sig=0.1, name=''):
    res = adfuller(series, autolag='AIC')    
    p_value = round(res[1], 3) 

    if p_value <= sig:
        print(f" {name} : P-Value = {p_value} => Stationary. ")
        return 1
    else:
        print(f" {name} : P-Value = {p_value} => Non-stationary.")
        return 0



def manual_arima(dist_df):
    dist_main_df=dist_df
    dist_df=dist_df[['Yield','Rain','N_TC','P_TC','K_TC']]
    #Granger Causality Test for 90%
    if dist==dist:
        try:
            variables=dist_df.columns  
            matrix = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
            for col in matrix.columns:
                for row in matrix.index:
                    test_result = grangercausalitytests(dist_df[[row, col]], maxlag=7, verbose=False)            
                    p_values = [round(test_result[i+1][0]['ssr_chi2test'][1],4) for i in range(7)]            
                    min_p_value = np.min(p_values)
                    matrix.loc[row, col] = min_p_value
            matrix.columns = [var + '_x' for var in variables]
            matrix.index = [var + '_y' for var in variables]
            print(matrix)

            for name, column in dist_df.iteritems():
                stationary_count =  adfuller_test(column, name=column.name)

            diff = 0
            stationary_count = 0
            data_diff = dist_df
            cmd = 1
            while (stationary_count!=5 and cmd==1) :
                stationary_count = 0
                diff += 1
                print(str(diff)+" Difference")
                data_diff = data_diff.diff().dropna()
                for name, column in data_diff.iteritems():
                    stationary_count += adfuller_test(column, name=column.name)
                if (stationary_count==5):
                    break
                # cmd = int(input("Go for next level differencing (1 for yes): "))
            print("D-value: "+str(diff))
            

                            
        except:
            print("Failed")
        for col in ['Yield','Rain','N_TC','P_TC','K_TC']:
            try:
                print("=================")
                print(col)
                print("=================")
                data_diff["ids"] = data_diff.index + 1
                # dist_main_df["id"] = dist_main_df.index + 1
                df = data_diff[['ids',col]]
                plt.rcParams.update({'figure.figsize':(15,3), 'figure.dpi':120})
                fig, axes = plt.subplots(1, 3, sharex=True)
                axes[0].plot(df[col]); axes[0].set_title(str(diff)+' Differencing for '+col)
                axes[1].set(ylim=(0,5))
                plot_pacf(df[col], ax=axes[1], method='ywm')
                plot_acf(df[col], ax=axes[2])
                plt.show() 
                p = int(input("P-value: "))
                q = int(input("Q-value: ")) 
                model = ARIMA(df[col].values, order=(p,diff,q))
                model_fit = model.fit()
                print(model_fit.summary())
            except:
                continue
            residuals = pd.DataFrame(model_fit.resid)
            fig, ax = plt.subplots(1,2)
            residuals.plot(title="Residuals", ax=ax[0])
            residuals.plot(kind='kde', title='Density', ax=ax[1])
            plt.show()
            # Forecast
            n_periods = 24
            fc, confint = model.predict(119, n_periods=n_periods, return_conf_int=True)
            index_of_fc = np.arange(len(dist_df[col].values), len(dist_df[col].values)+n_periods)

            # make series for plotting purpose
            fc_series = pd.Series(fc, index=index_of_fc)
            lower_series = pd.Series(confint[:, 0], index=index_of_fc)
            upper_series = pd.Series(confint[:, 1], index=index_of_fc)

            # Plot
            plt.plot(dist_df[col].values)
            plt.plot(fc_series, color='darkgreen')
            plt.fill_between(lower_series.index, 
                            lower_series, 
                            upper_series, 
                            color='k', alpha=.15)

            plt.title("Final Forecast of "+col)
            plt.show()

def auto_ARIMA(dist_df, dist):
    dist_main_df=dist_df
    dist_df=dist_df[['Yield','Rain','N_TC','P_TC','K_TC']]
    out_df=pd.DataFrame()
    for col in dist_df.columns:
        print("=================")
        print(col)
        print("=================")
        model = pm.auto_arima(dist_df[col].values, start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=5, max_q=5, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)
        print(model.summary())
        model.plot_diagnostics(figsize=(10,8))
        plt.show()
        # Forecast
        n_periods = 24
        fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
        index_of_fc = np.arange(len(dist_df[col].values), len(dist_df[col].values)+n_periods)

        # make series for plotting purpose
        fc_series = pd.Series(fc, index=index_of_fc)
        lower_series = pd.Series(confint[:, 0], index=index_of_fc)
        upper_series = pd.Series(confint[:, 1], index=index_of_fc)

        out_df[col] = fc_series

        # Plot
        plt.plot(dist_df[col].values)
        plt.plot(fc_series, color='darkgreen')
        plt.fill_between(lower_series.index, 
                        lower_series, 
                        upper_series, 
                        color='k', alpha=.15)

        plt.title("Final Forecast of "+col)
        plt.show()
    out_df.to_csv(os.getcwd()+"/Output/rice_forecast"+str(dist)+".csv")

main_df=pd.read_csv(os.getcwd()+"/Output/rice.csv")

for dist in main_df.Dist.unique():
    
    print(dist)
    dist_df=main_df[(main_df['Dist']==dist)]
    # dist_df[['Year','Yield']].plot(x='Year',y='Yield')
    # dist_df[['Year','Rain']].plot(x='Year',y='Rain')
    # dist_df[['Year','N_TC']].plot(x='Year',y='N_TC')
    # dist_df[['Year','P_TC']].plot(x='Year',y='P_TC')
    # dist_df[['Year','K_TC']].plot(x='Year',y='K_TC')
    cmd = int(input("Use manual ARIMA? (1 for yes): "))
    if (cmd==1):
        manual_arima(dist_df)
    else:
        auto_ARIMA(dist_df, dist)

    
        
            
    n=input("Check for next district (press enter):")
    if(n=="break"):
        break




