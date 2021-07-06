# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 15:44:41 2021

@author: hp1
"""
#Import needed modules/libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from scipy.optimize import fsolve, minimize
from scipy.integrate import trapz
from IPython.display import display, HTML

style.use("seaborn-dark-palette")

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

#Define needed functions here:
def VapourPressure(T):
    
    """ T in degree Celcius, P in mmHg"""
    A = 18.3036
    B = -3816.44
    C = -46.13
    lnP = A + B/(T+273.16+C)
    return np.exp(lnP)

def HumiditySat(T, Pt):
   
    """ T in degree Celcius, RH is percentage Relative Humidity"""
    VP = VapourPressure(T)
    Y =  (VP/(Pt - VP))*(18.02/28.97)
    return Y

def EnthalpySat(T, Pt):
   
    """ T in degree Celcius, H in kJ/mol"""
    Y = HumiditySat(T, Pt)
    H = 2500*Y + (1.005 + 1.88*Y)*T
    return H 

def Humidity(Tg, Ts, Pt):
    
    Ys = HumiditySat(Ts, Pt)
    def objective(Y):
        f = ((Tg-Ts)*((1.005 + 1.88*Y))/2500) - Ys + Y
        return f
    Y0 = 0.1*np.ones(Ys.size)
    Y = fsolve(objective, Y0)
    return Y

def Enthalpy(Y,T):
   
    H = 2500*Y + (1.005 + 1.88*Y)*T
    return H


def FindFlowrate(T_out, Pt, Ts, T_water_in):
    Q_T = T_out
    Q_H = Enthalpy(Humidity(T_out,Ts,Pt), T_out)
    def slopediff(T, Pt):
        h = 0.001
        x1 = Q_T
        y1 = Q_H
        x2 = T
        y2 = EnthalpySat(T,Pt)
        slope1 = (y2-y1)/(x2-x1)
        slope2 = (EnthalpySat(T+h, Pt)-EnthalpySat(T-h, Pt))/(2*h)
        error = (slope1 - slope2)**2
        return error
        
    
    T0 = T_water_in
    temp = minimize(slopediff, T0, args = (Pt))
    return temp
        
def FindSlope(T1,T2, Pt, Ts):
    x1 = T1
    y1 = Enthalpy(Humidity(T1,Ts, Pt), T1)
    x2 = T2
    y2 = EnthalpySat(T2, Pt)
    slope = (y2-y1)/(x2-x1)
    return slope
    
class CoolingTower:
    
    def __init__(self):
        self.Td_air_in = 31 #31 # degCelcius, Dry bulb Temperature of inlet Air
        self.Tw_air_in = 22 #22 # degCelcius, Wet bulb Temperature of Inlet Air
        self.T_water_out = 25 #30 #degCelcius, Temperature of water in Outlet
        self.T_water_in = 90 #45 #degCelcius, Temperature of water in Inlet
        self.Lwater = 6000 #6000 # kg/m^2hr, Mass flowrate of water
        self.Ratio = 1.4 #1.4   # Ratio between Air flowrate and Minimum needed ar flowrate
        self.cwl = 4.178 #4.178  # kJ/Kg.K , Specific heat of water
        self.kya = 6000 #6000    # kg/m^3.hr.delY', individual gas phase mass transfer coefficient
        self.Pt = 760 # mmHg
        self.Tg, self.Ts = self.Td_air_in, self.Tw_air_in
        
        
        
    def start_cooling(self):
        self.T_water_in = self.TCWout
        self.T_water_out = self.TCWin
        self.Lwater = self.coooling_water_flowrate
        self.Tg, self.Ts = self.Td_air_in, self.Tw_air_in
        self.Pt = 760 # mmHg
        
    def show_sat_curve(self):
        T = np.linspace(self.T_water_out*0.8,self.T_water_in*1.2,100)
        VP = VapourPressure(T)
        Y = HumiditySat(T, self.Pt)
        H = EnthalpySat(T, self.Pt)
        plt.figure(figsize = (7,7))
        plt.plot(T, H)
        plt.xlabel('Temperature, degC', fontsize = 15)
        plt.ylabel('Saturation Enthalpy, kJ/kg', fontsize = 15)
        plt.title('Saturation Enthalpy Curve', fontsize = 15)
        plt.grid('True')
        
    
    def show_tangent(self):
        temp = FindFlowrate(self.T_water_out, self.Pt, self.Ts, self.T_water_in).x
        m = FindSlope(self.T_water_out, temp, self.Pt, self.Ts)
        T1 = np.linspace(self.T_water_out,temp*1.3,100)
        plt.figure(figsize = (7,7))
        plt.plot(T1, EnthalpySat(T1, self.Pt))
        plt.plot(T1, Enthalpy(Humidity(self.T_water_out,self.Ts, self.Pt), self.T_water_out)+ m*(T1-T1[0]))
        plt.scatter([self.T_water_out], [Enthalpy(Humidity(self.T_water_out,self.Ts, self.Pt), self.T_water_out)], marker = "^")
        plt.plot([temp],EnthalpySat(temp, self.Pt) , 'x', markersize = 7, linewidth=5)
        plt.xlabel('Temperature, degC', fontsize = 15)
        plt.ylabel('Saturation Enthalpy, kJ/kg', fontsize = 15)
        plt.title('Saturation Enthalpy Curve', fontsize = 15)
        plt.grid('True')
        plt.annotate('Q', xy = (self.T_water_out,Enthalpy(Humidity(self.T_water_out,self.Ts, self.Pt), self.T_water_out)*1.1), fontsize = 15);
        plt.annotate('Tangent Line', xy = (35,100), fontsize = 15)
        plt.annotate('Saturation Enthalpy Curve', xy = (35,300), fontsize = 15);
        self.m = m
        
    def show_cool_opline(self):
        self.Gs_min = self.Lwater*self.cwl/self.m
        self.Gair = self.Gs_min*self.Ratio # Dry basis
        # New slope of operating line will be Ratio (1.4 this case)  times smaller than the slope of the tangent we drew earlier.
        m2 = self.m/self.Ratio 
        # operating line equation : 
        T2 = np.linspace(self.T_water_out, self.T_water_in, 100)
        self.Op = Enthalpy(Humidity(self.T_water_out,self.Ts, self.Pt), self.T_water_out)+ m2*(T2-T2[0])
        plt.figure(figsize = (7,7))
        plt.plot(T2, EnthalpySat(T2, self.Pt))
        plt.plot(T2, self.Op)
        plt.scatter([self.T_water_out], [Enthalpy(Humidity(self.T_water_out,self.Ts, self.Pt), self.T_water_out)], marker = "^", linewidth = 8, color = 'blue')
        plt.scatter([self.T_water_in], self.Op[-1], marker = "^", linewidth = 8, color = 'red')
        plt.xlabel('Temperature, degC', fontsize = 15)
        plt.ylabel('Enthalpy, kJ/kg', fontsize = 15)
        plt.title('Saturation Enthalpy and Operating Line', fontsize = 12)
        plt.grid('True')
        plt.annotate('Q', xy = (self.T_water_out,Enthalpy(Humidity(self.T_water_out,self.Ts, self.Pt), self.T_water_out)*1.1), fontsize = 15);
        plt.annotate('P', xy = (T2[-1], self.Op[-1]*1.05), fontsize = 15)
        plt.annotate('Operating Line', xy = (40,100), fontsize = 15)
        plt.annotate('Saturation Enthalpy Curve ', xy = (36,800), fontsize = 15);
        self.m2, self.T2 = m2, T2
        
    def show_tieline(self):
        hla = 0.059*np.power(self.Lwater, 0.51)*self.Gair # kcal/hr.m^3
        self.hla = hla*4.1858 # kJ/hr.m^3
        self.mtie = -self.hla/self.kya
        def OpLine(T):
            if (T > self.T_water_in) or (T < self.T_water_out):
                raise ValueError('Please enter value of T within temperature range')
            else:
                H = Enthalpy(Humidity(self.T_water_out,self.Ts, self.Pt), self.T_water_out)+ self.m2*(T-self.T2[0])
            return H
        
        def TieLine(T):
            x1,y1 = T, OpLine(T)
            def TieInt(X):
                x2, y2 = X[0], X[1]
                err1 = y2 - EnthalpySat(x2, self.Pt)
                err2 = y1 - y2 + self.mtie*(x2 - x1)
                error = err1,err2
                return error
            x20,y20 = x1*0.9, y1*1.1
            X = np.zeros(2); X[0] = x20; X[1] = y20
            sol = fsolve(TieInt, X)
            x2,y2 = sol[0], sol[1]
            return (x2,y2)
    
        def DrawTie(T):
            x1,y1 = T, OpLine(T);
            x2,y2 = TieLine(T);
            #plt.figure(figsize = (7,7))
            plt.plot(self.T2, EnthalpySat(self.T2, self.Pt), 'orange');
            plt.plot(np.linspace(25,30,100), EnthalpySat(np.linspace(25,30,100), self.Pt), 'orange');
            plt.plot(self.T2, self.Op);
            plt.plot([x1,x2], [y1,y2], '--k');
        
        TieDraw = np.vectorize(DrawTie)
        T3 = np.linspace(self.T_water_out, self.T_water_in, 10, dtype = float)
        plt.figure(figsize = (7,7));
        TieDraw(T3);
        plt.scatter([self.T_water_out], [Enthalpy(Humidity(self.T_water_out,self.Ts, self.Pt), self.T_water_out)], marker = "^", linewidth = 8, color = 'blue');
        plt.scatter([self.T_water_in], self.Op[-1], marker = "^", linewidth = 8, color = 'red');
        plt.xlabel('Temperature, degC', fontsize = 15);
        plt.ylabel('Enthalpy, kJ/kg', fontsize = 15);
        plt.title('Saturation Enthalpy Curve, Operating line and Tie Lines', fontsize = 12);
        plt.grid('True');
        plt.annotate('Q', xy = (self.T_water_out,Enthalpy(Humidity(self.T_water_out,self.Ts, self.Pt), self.T_water_out)*1.1), fontsize = 15);
        plt.annotate('P', xy = (self.T2[-1], self.Op[-1]*1.05), fontsize = 15)
        plt.annotate('Operating Line', xy = (40,100), fontsize = 15);
        plt.annotate('Saturation Enthalpy Curve', xy = (36,800), fontsize = 15);
        self.TieLine_vec = np.vectorize(TieLine)
        self.OpLine_vec = np.vectorize(OpLine)
        
    def calc_ntu_htu(self):
        x2, Hi = self.TieLine_vec(self.T2)
        H = self.OpLine_vec(self.T2)
        func = 1/(Hi - H)
        Integral = trapz(func)
        self.NTU = Integral 
        self.HTU = self.Gair/self.kya
        self.z = self.NTU*self.HTU
        
    def show_cooling_tower_chars(self):
        self.data_cool = {}
        self.data_cool['Design Parameter'] = ['Inlet Water Temperature (degC)','Outlet Water Temperature (degC)', 
                                    'Inlet Air Dry bulb Temperature (degC)', 'Inlet Air Wet bulb Temperature (degC)',
                                    'Inlet Water Flowrate (kg/m^2hr)', 'Inlet Air Flowrate (kg/m^2hr)',
                                        'NTU', 'HTU', 'Tower Height (m)' ]
        self.data_cool['Value']=[self.T_water_in ,self.T_water_out, self.Td_air_in, self.Tw_air_in, 
                                 self.Lwater, self.Gair,self.NTU, self.HTU, 
                                 self.z ]
        self.table_cool = pd.DataFrame(data = self.data_cool)
        display(HTML(self.table_cool.to_html()))
        
    
    