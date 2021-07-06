# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 20:16:54 2021

@author: hp1
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import scipy
import scipy.optimize as scopt
import scipy.integrate as scint
from IPython.display import display, HTML


from NRTL import NRTL
from PureComponentData import purecomponentdata
from AnimateConc import AnimateConc
from AnimateStage import AnimateStage
from CoolingTower import CoolingTower

style.use("seaborn-dark-palette")



import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

def get_residual_bubblepoint(T, dict_molfractions, nrtl, Ptotal,dict_compounds):
    dict_gammas = nrtl.get_gammas(T, dict_molfractions)
    P = 0.0
    for name in dict_molfractions:
        P += dict_gammas[name]*dict_molfractions[name]*dict_compounds[name].Pvap(T)
    return Ptotal - P

def get_bubble_point(Ptotal, dict_molfractions, nrtl, dict_compounds):    
    Tbubble = scopt.newton(get_residual_bubblepoint, 300.0, args = (dict_molfractions, nrtl, Ptotal,dict_compounds))
    return Tbubble

def get_residual_dewpoint(T, dict_molfractionsV, dict_molfractionsL, nrtl, Ptotal, dict_compounds):
    gammas = nrtl.get_gammas(T, dict_molfractionsL)
    summ = 0.0
    for name in dict_molfractionsL:
        s = Ptotal * dict_molfractionsV[name] / gammas[name] / dict_compounds[name].Pvap(T)
        summ += s
    return summ - 1

def get_dew_point(Ptotal, dict_molfractions, nrtl, dict_compounds):
    bool_go = True
    dict_molfractionsL = dict_molfractions.copy()
    dict_molfractionsV = dict_molfractions.copy()
    Told = 300.0
    while bool_go:
        T = scopt.newton(get_residual_dewpoint, Told, args=(dict_molfractionsV, dict_molfractionsL, nrtl, Ptotal, dict_compounds))
        gammas = nrtl.get_gammas(T, dict_molfractionsL)
        sumx = 0.0
        for name in dict_molfractionsV:
            x = Ptotal * dict_molfractionsV[name] / gammas[name] / dict_compounds[name].Pvap(T)
            dict_molfractionsL[name] = x
            sumx += x
        for name in dict_molfractionsL:
            dict_molfractionsL[name] /= sumx
        
        if abs(T - Told) < 0.1:
            bool_go = False
        else:
            Told = T + 0.0
        #print(T)
    return T



#eqm_vector = np.vectorize(eqm_curve)

class BinaryDistillation(AnimateConc, AnimateStage, CoolingTower):
    def __init__(self, nrtl, dict_compounds):
        self.nrtl = nrtl
        self.dict_compounds = dict_compounds
        self.F = 100   # kmol/hr
        self.xf = 0.5   # mol fraction of hexane in feed
        self.xd = 0.9  # mol fraction of hexane in distillate
        self.D = 0.5*self.F  # Flowrate of top stream
        self.B = self.F - self.D  # Flowrate of bottom stream
        self.xb = (self.F*self.xf - self.D*self.xd)/self.B
        self.R = 2  # Ratio between Actual reflux ratio and minimum reflux ratio 
        self.q = 1  # feed condition
        self.Pt = 101325 # Pa, Total pressure 
        self.x = np.linspace(0,1,50)
        self.res_time = 0.005 #hr, residence time in each stage
        self.lambda_HC = 400 #kJ/kg
        self.lambda_W = 2260 #kJ/kg
        self.CpW = 4.18  # kJ/kg/K
        
        #########################
        # Initializzation of variables for cooling tower
        self.Td_air_in = 31 #31 # degCelcius, Dry bulb Temperature of inlet Air
        self.Tw_air_in = 22 #22 # degCelcius, Wet bulb Temperature of Inlet Air
        self.T_water_out = 25 #30 #degCelcius, Temperature of water in Outlet
        self.T_water_in = 90 #45 #degCelcius, Temperature of water in Inlet
        self.Lwater = 6000 #6000 # kg/m^2hr, Mass flowrate of water
        self.Ratio = 1.4 #1.4   # Ratio between Air flowrate and Minimum needed ar flowrate
        self.cwl = 4.178 #4.178  # kJ/Kg.K , Specific heat of water
        self.kya = 6000 #6000    # kg/m^3.hr.delY', individual gas phase mass transfer coefficient
        #self.Pt = 760 # mmHg
        #self.Tg, self.Ts = self.Td_air_in, self.Tw_air_in
        
        
        
    def q_line(self, xq):
        if self.q == 1:
            yq = xq

        else : 
            yq = (xq*self.q - self.xf)/(self.q - 1)

        return yq    
    
    def eqm_curve(self, x):
        self.names_list = []
        for name in self.dict_compounds.keys():
            self.names_list.append(name)
        dict_molfractions = {str(self.names_list[0]):x, str(self.names_list[1]):1-x}
        Td =  get_dew_point(self.Pt, dict_molfractions, self.nrtl, self.dict_compounds)
        gammas = self.nrtl.get_gammas(Td, dict_molfractions)
        dict_molfractionsV = dict_molfractions.copy()
        for name in dict_molfractions:
            y = dict_molfractions[name]*gammas[name]*self.dict_compounds[name].Pvap(Td)/self.Pt
            dict_molfractionsV[name] = y
        return dict_molfractionsV[str(self.names_list[0])]
    
    def show_xy(self):        
        plt.figure(figsize = (8,8))
        self.eqm_vector = np.vectorize(self.eqm_curve)
        
        self.y = self.eqm_vector(self.x)
        plt.plot(self.x, self.y, label = 'eqm-curve')
        plt.plot([0,1], [0,1])
        plt.grid()
        self.names_list = []
        for name in self.dict_compounds.keys():
            self.names_list.append(name)
        plt.xlabel('Mol fraction of ' + str(self.names_list[0]) + ' in Liquid Phase', fontsize = 15)
        plt.ylabel('Mol fraction of ' + str(self.names_list[0]) + ' in Vapour Phase',  fontsize = 15)
        plt.title('x-y Diagram');
        plt.xlim([0,1]) ; plt.ylim([0,1])
        plt.legend();
        
    def show_txy(self):
        Tdew = np.zeros(len(self.x))
        Tbub = np.zeros(len(self.x))
        for i in range(len(self.x)):
            dict_molfractions = {str(self.names_list[0]):self.x[i], str(self.names_list[1]):1-self.x[i]}
            
            Tbub[i] =  get_bubble_point(self.Pt, dict_molfractions, self.nrtl, self.dict_compounds)-273.16
            Tdew[i] =  get_dew_point(self.Pt, dict_molfractions, self.nrtl, self.dict_compounds)-273.16
        
        plt.figure(figsize=(10,8)) 
        plt.plot(self.x, Tdew, 'r', label = 'y');
        plt.plot(self.x, Tbub, 'b' , label = 'x');
        plt.fill_between(self.x, Tdew, Tdew[0]+0.5, 
                        color = 'red', alpha = 0.5)
        plt.fill_between(self.x, Tbub, np.min(Tbub)-0.5, 
                        color = 'skyblue', alpha = 0.5)
        plt.xlabel('Mol fraction of ' + str(self.names_list[0]) + ' in Liquid Phase', fontsize = 15)
        plt.ylabel('Temperature, degC', fontsize = 15)
        plt.xlim([0,1]); plt.ylim([np.min(Tbub)-0.5, Tdew[0]+0.5]);
        plt.annotate('Vapour', xy = (0.2,Tdew[0] ), fontsize = 15) ; plt.annotate('Liquid', xy = (0.2,np.min(Tbub) ), fontsize = 15); 
        plt.grid();
        plt.legend();
        plt.title("T-x-y diagram at " + str(self.Pt)+ " Pa", fontsize = 15);
        

    
        
    def show_q(self):
        self.show_xy()
        if self.q == 1:
            yq = self.q_line(np.linspace(self.xf,self.eqm_vector(self.xf),2))
            plt.plot(np.ones(len(yq))*self.xf,yq , label = 'q-line')
        else:
            yq = self.q_line(np.linspace(self.xf,-(1-self.q)*500, 2 ))
            plt.plot(np.linspace(self.xf,-(1-self.q)*500, 2),yq , label = 'q-line')
        plt.legend(loc = "best");   
        
    def get_min_reflux(self):
        if self.q == 1:
                xp = self.xf
                yp = self.eqm_vector(self.xf)
        else :
                def get_res_q(x):
                    res = self.q_line(x) - self.eqm_vector(x)
                    return res
                xp = scipy.optimize.fsolve(get_res_q, [0.5])
                yp = self.eqm_vector(xp)
        Rmin = (self.xd - yp)/(yp - xp)
        return (Rmin,xp,float(yp))

    def rectifying_line(self,x):
        self.Rmin, self.xp, self.yp = self.get_min_reflux()
        self.Ract = self.Rmin*self.R        
        y = (self.Ract*x + self.xd)/(self.Ract+1)
        return y
    
    
    def intersect_point(self):
        if self.q == 1:
            xi = self.xf
            yi = self.rectifying_line(self.xf)
        else :
            def get_res_int(x):
                res = self.q_line(x) - self.rectifying_line(x)
                return res
            xi = scipy.optimize.fsolve(get_res_int, [0.5])
            yi = self.rectifying_line(xi)

        return (xi,yi)
    
    def vol_flow(self):
        self.Lr = self.Ract*self.D
        self.Ls = self.Lr + self.q*self.F
        self.Vr = self.Lr + self.D
        self.Vs = self.Vr + (self.q - 1)*self.F
        
        
    def stripping_line(self,x):
        self.xb = (self.F*self.xf - self.D*self.xd)/self.B
        #self.vol_flow()
        m = (self.yi - self.xb)/(self.xi - self.xb)
        c = self.xb
        y = m*(x- self.xb) + c
        return y
        
    
    def show_opline(self, arg = True):
        self.xb = (self.F*self.xf - self.D*self.xd)/self.B
        self.xi, self.yi = self.intersect_point()
        self.show_q()
        plt.plot(self.xp, self.yp, 'k^', markersize = 8)
        plt.plot(self.xi, self.yi, 'kx', markersize = 8)
        plt.plot(np.linspace(self.xi, self.xd, 20), self.rectifying_line(np.linspace(self.xi, self.xd, 20)), label = "Rectifying Curve")
        plt.plot(np.array([self.xb, self.xi]), self.stripping_line(np.array([self.xb, self.xi])), label = "Stripping Curve")
        if arg:
            plt.plot([0, self.xi], [self.xd/(self.Ract+1), self.yi], 'k--')
       
            textstr =("Parameters\n \n"+ 
                      "q value: %.1f\n" %(self.q)+ 
                      "Min Reflux Ratio: %.1f\n" %(self.Rmin)+
                      "Reflux Ratio: %.1f\n" %(self.Ract))
            props = dict(boxstyle='round', facecolor='gold', alpha=0.5)
            plt.text(0.4, 0.2, textstr, fontsize=10, verticalalignment='top', bbox=props);
        plt.legend(loc = "upper left")
        
   
        
    def num_stages(self, arg = True):
        self.xb = (self.F*self.xf - self.D*self.xd)/self.B
        if arg:
            self.show_opline(arg = False)
        i = 0 ; r = 0 ; s = 0
        xm, ym = self.xd, self.xd
        def get_eq_x(y):
            yt = y
            def get_res(xt):
                y_temp = self.eqm_vector(xt)
                error = (y_temp - yt)**2
                return error
            
            x = scipy.optimize.fsolve(get_res, [0.5])
            return x
                
        while xm >= self.xb:
            xtemp, ytemp = get_eq_x(ym), ym
            if arg:
                plt.plot([xtemp, xm], [ytemp, ym], '-k')
            if xtemp >= self.xi :
                xm, ym = xtemp, self.rectifying_line(xtemp)
                r += 1
            else:
                xm, ym = xtemp, self.stripping_line(xtemp)
                s += 1
            if arg:
                plt.plot([xtemp, xm], [ytemp, ym], '-k')
            i += 1
        self.rect_stages = r
        self.strip_stages = s
        self.total_stages = i
        textstr =("Parameters\n \n"+ 
                  "q value: %.1f\n" %(self.q)+ 
                  "Min Reflux Ratio: %.3f\n" %(self.Rmin)+
                  "Reflux Ratio: %.3f\n" %(self.Ract)+
                  "Number of Stages in Rectifying Section : %.1f\n"%(r)+
                  "Number of Stages in Stripping Section : %.1f\n"%(s))
        props = dict(boxstyle='round', facecolor='gold', alpha=0.5)
        plt.text(0.45, 0.2, textstr, fontsize=10, verticalalignment='top', bbox=props);
        plt.plot(self.xd, self.xd, '^r', markersize = 8)
        plt.plot(self.xb, self.xb, '^r', markersize = 8)
        plt.annotate('Top Composition' , xy = (self.xd , self.xd - 0.03))
        plt.annotate('Bottom Composition' , xy = (self.xb , self.xb - 0.03))
        plt.title('McCabe-Thiele Plot')
        
    def duty_req(self, TCW = [30, 50]):
        #dict_molfractions = {str(self.names_list[0]):self.xd, str(self.names_list[1]):1-self.xd}
        #Ttop = get_dew_point(self.Pt, dict_molfractions, self.nrtl, self.dict_compounds)
        #dict_molfractions = {str(self.names_list[0]):self.xb, str(self.names_list[1]):1-self.xb}
        #Tbot = get_dew_point(self.Pt, dict_molfractions, self.nrtl, self.dict_compounds)
        #Hvap1R = self.dict_compounds[self.names_list[0]].Hvap(Ttop)
        #Hvap2R = self.dict_compounds[self.names_list[1]].Hvap(Ttop)
        #Hvap1S = self.dict_compounds[self.names_list[0]].Hvap(Tbot)
        #Hvap2S = self.dict_compounds[self.names_list[1]].Hvap(Tbot)
        #lambdaR = Hvap1R*self.xd + Hvap2R*(1-self.xd)
        #lambdaS = Hvap1S*self.xb + Hvap2S*(1-self.xb)
        self.TCWin = TCW[0] ; self.TCWout = TCW[1]
        self.vol_flow()
        self.dict_compounds[self.names_list[0]].getCriticalConstants()
        MW1 = self.dict_compounds[self.names_list[0]].MW
        self.dict_compounds[self.names_list[1]].getCriticalConstants()
        MW2 = self.dict_compounds[self.names_list[1]].MW
        
        #water = purecomponentdata.Compound("Water") 
        #HvapW = water.Hvap(Tbot)
        
        QR = self.Vr*(self.xd*MW1 + (1-self.xd)*MW2)  # kg/hr
        self.coooling_water_flowrate = (QR*self.lambda_HC/(self.CpW*(TCW[1]-TCW[0])))/18 # kmol/hr
        
        QS = self.Vs*(self.xb*MW1 + (1-self.xb)*MW2)  # kg/hr
        self.steam_flowrate = (QS*self.lambda_HC/self.lambda_W)/18 #kmol/hr
        
        #return self.coooling_water_flowrate,self.steam_flowrate 
        
        
        
    def simulate_transient(self, time):
            self.vol_flow()
            Lr, Ls, Vr, Vs = self.Lr, self.Ls, self.Vr, self.Vs
            feed_point_ind = self.rect_stages-1
            self.res_time = 0.01
            def solve_ode(SV, t):
                x = SV
                y = self.eqm_vector(x)
                dxdt = np.zeros(np.size(x))
                Hr = self.res_time*Lr ; Hs = self.res_time*Ls
                x[0] = self.xd ; x[-1] = self.xb
                y[0] = x[0] ; y[-1] = x[-1] 
                
                #dxdt[1:feed_point_ind+1] = (Vr*y[2:feed_point_ind+2] + Lr*x[0:feed_point_ind] - Lr*x[1:feed_point_ind+1] - Vr*y[1:feed_point_ind+1])/Hr
                #dxdt[feed_point_ind+1:-2]= (Vs*y[feed_point_ind+2:-1] + Ls*x[feed_point_ind:-3] - Ls*x[feed_point_ind+1:-2] - Vs*y[feed_point_ind+1:-2])/Hs 
                for i in range(1,self.total_stages+1):
                    j = i+1 ; k = i-1
                   
                    if i <= feed_point_ind: 
                        dxdt[i] = (Vr*y[j] + Lr*x[k] - Lr*x[i] - Vr*y[i])/Hr                        
                    else:
                        dxdt[i] = (Vs*y[j] + Ls*x[k] - Ls*x[i] - Vs*y[i])/Hs 
                
                #dxdt[0] =  (Lr*y[0] + Vr*y[1] - Lr*x[0] - Vr*y[0] )/Hr
                #dxdt[-1]=  (Ls*x[-2]  - Vs*y[-1] - Ls*x[-1] + Vs*x[-1] )/Hs            
                return dxdt
          
                
            self.time = time
            x0 = np.ones(self.total_stages+2)*self.xf
            steps = 50
            self.t = np.linspace(0, time, steps)                                             
            self.sol = scint.odeint(solve_ode, x0, self.t)
            
    def show_distillation_chars(self):
        self.data_dist = {}
        self.data_dist["Parameter"] = ['Feed Flowrate (kmol/hr)','Feed Composition',
                                       'Top Product Flowrate (kmol/hr)','Top Product Composition', 
                                       'Bottom Product Flowrate (kmol/hr)','Bottom Product Composition',
                                       'Feed Quality (q)', 'Minimum Reflux Ratio', 'Actual Reflux Ratio',
                                       'Number of Stages needed', 'Cooling water Flowrate (kmol/hr)', 
                                       'Steam Flowrate (kmol/hr)']
        self.data_dist["Value"] = [self.F, self.xf, self.D, self.xd, self.B, self.xb, self.q, self.Rmin,self.Ract,
                                   self.total_stages, self.coooling_water_flowrate,self.steam_flowrate]
        
        self.table_dist= pd.DataFrame(data = self.data_dist)
        display(HTML(self.table_dist.to_html()))
            