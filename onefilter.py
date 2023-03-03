import numpy as np 
import statsmodels.tsa.stattools as ts
import math 
import logging  
import pandas as pd
from sklearn import linear_model # apply linear model
from sklearn.preprocessing import PolynomialFeatures # apply quadratic model
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm # apply Baxter-King and Hodrick-Prescott filters
import statsmodels as sms # apply seasonalization of time series
import quantecon # apply Hamilton Filter
import datetime as dt
from decimal import Decimal # round rational numbers



def BoostedHP(x, lam = 1600, iter = True, stopping = "BIC", \
              sig_p = 0.050, Max_Iter = 100):
        
    x = np.array(x)
    ## generating trend operator matrix "S：        
    raw_x = x # save the raw data before HP
    n = len(x) # data size
    
    I_n = np.eye(n)
    D_temp = np.vstack((np.zeros([1,n]),np.eye(n-1,n)))
    D_temp= np.dot((I_n-D_temp),(I_n-D_temp))
    D = D_temp[2:n].T
    S = np.linalg.inv(I_n+lam*np.dot(D,D.T)) # Equation 4 in PJ
    mS = I_n - S
    
    ##########################################################################  
    
    ## the simple HP-filter
    if not iter:
        
        print("Original HP filter.")
        x_f = np.dot(S,x)
        x_c = x - x_f
        result = {"cycle": x_c, "trend_hist" : x_f, \
                   "stopping" : "nonstop", "trend" : x - x_c, "raw_data" : raw_x}
            
    ##########################################################################
            
    ## The Boosted HP-filter 
    if iter:
        ### ADF test as the stopping criterion
        if stopping == "adf":       
            print("Boosted HP-ADF.")
            r = 1
            stationary = False
            x_c = x
            x_f = np.zeros([n,Max_Iter])
            adf_p = np.zeros([Max_Iter,1])
            while (r <= Max_Iter) and (not stationary):
                x_c = np.dot(mS,x_c)
                x_f[:,[r-1]] = x-x_c
                adf_p_r = ts.adfuller(x_c, maxlag = math.floor(pow(n-1,1/3)), autolag=None, \
                                      regression = "ct")[1]
                # x_c is the residual after the mean and linear trend being removed by HP filter
                # we use the critical value for the ADF distribution with
                    # the intercept and linear trend specification
                adf_p[[r-1]] = adf_p_r
                stationary = adf_p_r <= sig_p
                # Truncate the storage matrix and vectors
                if stationary:
                    R = r
                    x_f = x_f[:,0:R]
                    adf_p = adf_p[0:R]
                    break
                r += 1
            
            if r > Max_Iter:
                R = Max_Iter
                logging.warning("The number of iterations exceeds Max_Iter. \
                The residual cycle remains non-stationary.")
                
            result = {"cycle" : x_c, "trend_hist" : x_f,  "stopping" : stopping,
                     "signif_p" : sig_p, "adf_p_hist" : adf_p, "iter_num" : R,
                    "trend" : x - x_c, "raw_data" : raw_x}

        else: # either BIC or nonstopping
            
            # assignment 
            r = 0
            x_c_r = x
            x_f = np.zeros([n,Max_Iter])
            IC = np.zeros([Max_Iter,1])
            # IC_decrease = True
            I_S_0 = I_n - S
            c_HP = np.dot(I_S_0, x)
            I_S_r = I_S_0
            
            while r < Max_Iter:
                r += 1
                x_c_r = np.dot(I_S_r, x)
                x_f[:,[r-1]] = x - x_c_r
                B_r = I_n - I_S_r 
                IC[[r-1]] =  np.var(x_c_r)/np.var(c_HP) + \
                    np.log(n)/(n-np.sum(np.diag(S))) * np.sum(np.diag(B_r))
                I_S_r = np.dot(I_S_0, I_S_r) # update for the next round
                if r >= 2 and stopping == "BIC":
                    if IC[[r-2]] < IC[[r-1]]:
                        break
            
            # final assignment
            R = r-1
            x_f = x_f[:, list(range(0,R))]
            x_c = x - x_f[:, [R-1]]
            #x_c = pd.Series(np.squeeze(x_c))
            
            if stopping == "BIC":
                print("Boosted HP-BIC.")
                # save the path of BIC till iter+1 times to keep the "turning point" of BIC history.
                result = {"cycle" : x_c, "trend_hist" : x_f,  "stopping" : stopping, 
                       "BIC_hist" : IC[0:(R+1)], "iter_num" : R, "trend" : x- x_c, "raw_data" : raw_x}
            
            if stopping == "nonstop":
                print('Boosted HP-BIC with stopping = "nonstop".')
                result = {"cycle" : x_c, "trend_hist" : x_f,  "stopping" : stopping, 
                       "BIC_hist" : IC, "iter_num" : Max_Iter - 1, "trend" : x- x_c, "raw_data" : raw_x}
            
    return result 

def cycles(var, period):
    
    if period == 'annual':
        bk_1 = 1.5
        bk_2 = 8
        bk_3 = 3
        hp_1 = 6.25
        h_1 = 2
        h_2 = 1
        
    if period == 'quarter':
        bk_1 = 6
        bk_2 = 32
        bk_3 = 12
        hp_1 = 1600
        h_1 = 8
        h_2 = 4

    if period == 'monthly' or period =="monthly_smoothed":
        bk_1 = 6
        bk_2 = 32
        bk_3 = 12
        hp_1 = 14400
        h_1 = 24
        h_2 = 12
    
    # loading the data to be evaluated
    
    if period == 'monthly_smoothed':
        var = var.rolling(window=3).mean()
    var = var.dropna() # removing null entries
    var = var["1975-01-01":] # setting the time series start
    X = var.index # defining the covariate 'X' as the variable time index
    X = X.map(dt.datetime.toordinal) # transforming 'X' to datetimeindex format
    X = np.array(X) # setting 'X' as an array
    X = X.reshape(-1, 1) # transposing 'X' to regressions
    
    # computing linear regression cycle component
    reg_lin = linear_model.LinearRegression(fit_intercept = True) # defining 'reg_lin' as the linear regression application
    reg_lin.fit(X, var) # applying the linear regression of 'var' over 'X'
    y_hat_int = reg_lin.predict(X) # defining 'y_hat_int' as the linear regression forecasting
    cycle_lin_var = var - y_hat_int # defining 'cycle_lin_var' as the residual of the linear regression forecasting
    
    # computing quadratic regression cycle component
    quad = PolynomialFeatures(degree=2) # defining 'quad' as the quadratic function
    reg_quad = linear_model.LinearRegression(fit_intercept = False) # defining 'reg_quad' as the linear regression application
    X_quad = quad.fit_transform(X) # defining 'X_quad' as the quadratic transformation of 'X'
    reg_quad.fit(X_quad, var) # applying the quadratic regression of 'var' over 'X_quad'
    y_hat_quad = reg_quad.predict(X_quad) # defining 'y_hat_quad' as the quadratic regression forecasting
    cycle_quad_var = var - y_hat_quad # defining 'cycle_quad_var' as the residual of the linear regression forecasting
    
    # computing Baxter-King filter cycle component
    cycle_bk_var = sm.tsa.filters.bkfilter(var, bk_1, bk_2, bk_3) # defining 'cycle_bk_var' as the BK filter cycle component
    
    # computing Hodrick-Prescott filter cycle component
    cycle_hp_var, trend_hp_var = sm.tsa.filters.hpfilter(var, lamb=hp_1) # defining 'cycle_hp_var' as the HP filter cycle component
    
    # computing Hamilton filter cycle component
    try:
        cycle_h_var, trend_h_var = quantecon.hamilton_filter(var, h_1, h_2) # defining 'cycle_h_var' as the Hamilton filter cycle component
        cycle_h_var = pd.DataFrame(cycle_h_var) # converting 'cycle_h_var' to a dataframe format
        cycle_h_var.index = var.index # setting 'cycle_h_var' index equal to the 'var' index
    except:
        cycle_h_var, trend_h_var = np.nan, np.nan
        pass

    # Christiano Fitzgerald
    cf_cycles, cf_trend = sm.tsa.filters.cffilter(var, bk_1, bk_2)

    # Boosted HP
    bx_HP = BoostedHP(var, lam = hp_1, iter = False)
    bx_cycle = pd.Series(np.squeeze(bx_HP["cycle"]))
    bx_cycle.index = var.index

    return cycle_lin_var, cycle_quad_var, cycle_bk_var, cycle_hp_var, cycle_h_var, cf_cycles, bx_cycle

def filter_var(var, var_name, period):

    var = var.dropna() # removing null entries
    var = var.loc[var.index >= dt.datetime(1996,1,1)] # setting the time series start  
    
    cycle_lin_var, cycle_quad_var, cycle_bk_var, cycle_hp_var, cycle_h_var, cf_cycles, bx_cycle = cycles(var, period)

    # generating 'x' values
    ### SMOOTHING 3m 
    if period == 'monthly_smoothed':
        var = var.rolling(window=3).mean()    
        var = var.dropna()

    graph_yrs_var = var.index
    graph_yrs_bk_var = pd.to_datetime(cycle_bk_var.index, format="%Y")  
    
    res_zeros = [0 for i in range(len(graph_yrs_var))]
    
    # start generating graphs, setting the figure
    fig, axs = plt.subplots(nrows= 4, ncols= 2, figsize=(16,18))
    fig.suptitle('Comparing '+var_name, y= .93, fontsize=16, fontweight='bold')
    
    # linear regression cycles, first graph
    #plt.style.use("seaborn")
    plt.subplot(4,2,1)
    axs[0,0].plot(graph_yrs_var, cycle_lin_var) #Interest variable
    axs[0,0].plot(graph_yrs_var, res_zeros) #Line at zero
    plt.title('Linear Filter', fontweight="bold")

    # quadratic regression cycles, second graph
    plt.subplot(4,2,2)
    axs[0,1].plot(graph_yrs_var, cycle_quad_var) #Interest variable
    axs[0,1].plot(graph_yrs_var, res_zeros) #Line at zero
    plt.title('Quadratic Filter', fontweight="bold")

    # Baxter-King cycles, third graph
    plt.subplot(4,2,3)
    axs[1,0].plot(graph_yrs_bk_var, cycle_bk_var) #Interest variable
    axs[1,0].plot(graph_yrs_var, res_zeros) #Line at zero
    plt.title('Baxter-King Filter', fontweight="bold")

    # Hodrick-Prescott cycles, fourth graph
    plt.subplot(4,2,4)
    axs[1,1].plot(graph_yrs_var, cycle_hp_var) #Interest variable
    axs[1,1].plot(graph_yrs_var, res_zeros) #Line at zero
    plt.title('Hodrick-Prescott Filter', fontweight="bold")

    # Hamilton cycles, fifth graph
    plt.subplot(4,2,5)
    try:
        axs[2,0].plot(graph_yrs_var, cycle_h_var[0]) #Interest variable
        axs[2,0].plot(graph_yrs_var, res_zeros) #Line at zero
    except:
        axs[2,0].plot(graph_yrs_var, res_zeros)
    plt.title('Hamilton Filter', fontweight="bold")

    # GDP and 'var' cycles, sixth graph
    plt.subplot(4,2,6)
    axs[2,1].plot(graph_yrs_var, var/var.iloc[0]) #Interest variable
    plt.title('log.Variable', fontweight="bold")

    # Boosted HP
    plt.subplot(4,2,7)
    axs[3,0].plot(graph_yrs_var, bx_cycle) #Interest variable
    axs[3,0].plot(graph_yrs_var, res_zeros) #Line at zero
    plt.title('Boosted HP Filter', fontweight="bold")

    # Christiano Fitzgerald
    plt.subplot(4,2,8)
    axs[3,1].plot(graph_yrs_var, cf_cycles) #Interest variable
    axs[3,1].plot(graph_yrs_var, res_zeros) #Line at zero
    plt.title('Christiano Fitzgerald Filter', fontweight="bold")

    plt.subplots_adjust(hspace=0.2)
  
    return plt.show() 




def filter_comparison(var, var_name, period):

    var = var.dropna() # removing null entries
    var = var.loc[var.index >= dt.datetime(1996,1,1)] # setting the time series start  
    
    # deciding if we compare 'var' to annual GDP ou quarter GDP according to 'var' periodicity
#    if period == 'quarter': 
#        output = df['GDP [Q]']
#    if period == 'annual':
#        output = df['GDP [Y]']
    
    # generating cycle component series of GDP and 'var' from the five techniques of interest
#    cycle_lin, cycle_quad, cycle_bk, cycle_hp, cycle_h = cycles(output, period)
    cycle_lin_var, cycle_quad_var, cycle_bk_var, cycle_hp_var, cycle_h_var, cf_cycles, bx_cycle = cycles(var, period)

    # generating 'x' values
    ### SMOOTHING 3m 
    if period == 'monthly_smoothed':
        var = var.rolling(window=3).mean()    
        var = var.dropna()

    graph_yrs_var = var.index
    graph_yrs_bk_var = pd.to_datetime(cycle_bk_var.index, format="%Y")  
    
    # resizing 'y' values
#    res_cycle_lin = cycle_lin.loc[cycle_lin.index >= graph_yrs_var[0]]
#    res_cycle_quad = cycle_quad.loc[cycle_quad.index >= graph_yrs_var[0]]
#    res_cycle_bk = cycle_bk.loc[cycle_bk.index >= graph_yrs_var[0]] 
#    res_cycle_hp = cycle_hp.loc[cycle_hp.index >= graph_yrs_var[0]]
#    res_cycle_h = cycle_h.loc[cycle_h.index >= graph_yrs_var[0]]
#    res_y = output.loc[output.index >= graph_yrs_var[0]]
#    res_y = res_y.dropna()
    res_zeros = [0 for i in range(len(graph_yrs_var))]
    
    # start generating graphs, setting the figure
    fig, axs = plt.subplots(nrows= 4, ncols= 2, figsize=(16,18))
    fig.suptitle('Comparing '+var_name, y= .93, fontsize=16, fontweight='bold')
    
    # linear regression cycles, first graph
    #plt.style.use("seaborn")
    plt.subplot(4,2,1)
#    axs[0,0].plot(res_cycle_lin) #GDP
    axs[0,0].plot(graph_yrs_var, cycle_lin_var) #Interest variable
    axs[0,0].plot(graph_yrs_var, res_zeros) #Line at zero
    plt.title('Linear Filter', fontweight="bold")
#    axs[0,0].legend(labels = ['GDP', var_name])
    #plt.text(10500, -.28, lin_stats_summary(var_name))
    
    # quadratic regression cycles, second graph
    plt.subplot(4,2,2)
#    axs[0,1].plot(res_cycle_quad) #GDP
    axs[0,1].plot(graph_yrs_var, cycle_quad_var) #Interest variable
    axs[0,1].plot(graph_yrs_var, res_zeros) #Line at zero
    plt.title('Quadratic Filter', fontweight="bold")
#    axs[0,1].legend(labels = ['GDP', var_name])
    #plt.text(10500, -.28, quad_stats_summary(var_name))
    
    # Baxter-King cycles, third graph
    plt.subplot(4,2,3)
#    axs[1,0].plot(res_cycle_bk) #GDP
    axs[1,0].plot(graph_yrs_bk_var, cycle_bk_var) #Interest variable
    axs[1,0].plot(graph_yrs_var, res_zeros) #Line at zero
    plt.title('Baxter-King Filter', fontweight="bold")
#    axs[1,0].legend(labels = ['GDP', var_name])
    #plt.text(10500, -.11, bk_stats_summary(var_name))
    
    # Hodrick-Prescott cycles, fourth graph
    plt.subplot(4,2,4)
#    axs[1,1].plot(res_cycle_hp) #GDP
    axs[1,1].plot(graph_yrs_var, cycle_hp_var) #Interest variable
    axs[1,1].plot(graph_yrs_var, res_zeros) #Line at zero
    plt.title('Hodrick-Prescott Filter', fontweight="bold")
#    axs[1,1].legend(labels = ['GDP', var_name])
#    axs[1,1].legend(labels = ['GDP', var_name])
    #plt.text(10500, -.1, hp_stats_summary(var_name))
    
    # Hamilton cycles, fifth graph
    plt.subplot(4,2,5)
#    axs[2,0].plot(res_cycle_h) #GDP
    try:
        axs[2,0].plot(graph_yrs_var, cycle_h_var[0]) #Interest variable
        axs[2,0].plot(graph_yrs_var, res_zeros) #Line at zero
    except:
        axs[2,0].plot(graph_yrs_var, res_zeros)
    plt.title('Hamilton Filter', fontweight="bold")
#    axs[2,0].legend(labels = ['GDP', var_name])
    #plt.text(10500, -.3, h_stats_summary(var_name))
    
    # GDP and 'var' cycles, sixth graph
    plt.subplot(4,2,6)
#    axs[2,1].plot(res_y/res_y.iloc[0]) #GDP
    axs[2,1].plot(graph_yrs_var, var/var.iloc[0]) #Interest variable
    plt.title('log.GDP and log.Variable', fontweight="bold")
#    axs[2,1].legend(labels = ['GDP', var_name])       

    # Boosted HP
    plt.subplot(4,2,7)
#    axs[1,1].plot(res_cycle_hp) #GDP
    axs[3,0].plot(graph_yrs_var, bx_cycle) #Interest variable
    axs[3,0].plot(graph_yrs_var, res_zeros) #Line at zero
    plt.title('Boosted HP Filter', fontweight="bold")

    # Christiano Fitzgerald
    plt.subplot(4,2,8)
#    axs[1,1].plot(res_cycle_hp) #GDP
    axs[3,1].plot(graph_yrs_var, cf_cycles) #Interest variable
    axs[3,1].plot(graph_yrs_var, res_zeros) #Line at zero
    plt.title('Christiano Fitzgerald Filter', fontweight="bold")

    plt.subplots_adjust(hspace=0.2)
  
    return plt.show() 