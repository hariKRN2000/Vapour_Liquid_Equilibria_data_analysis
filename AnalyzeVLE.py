# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 16:54:00 2021

@author: hp1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

from scipy.optimize import least_squares
from NRTL import NRTL
from PureComponentData import purecomponentdata

style.use("seaborn-dark-palette")

class AnalyzeVLE:
    
    def __init__(self, comp_list, df_vle):
        self.comp_list = comp_list
        self.df_vle = df_vle
        self.alpha = 0.3
        
    def initialize_reading(self):
        self.mvc_name = str(self.comp_list[0])
        self.lvc_name = str(self.comp_list[1])
        
        self.mvc = purecomponentdata.Compound(self.mvc_name)
        self.lvc = purecomponentdata.Compound(self.lvc_name)
        if len(self.lvc_name.split('-')) >= 2 :
            self.lvc_name =  self.lvc_name.split('-')[-1]
        if len(self.mvc_name.split('-')) >= 2 :
            self.mvc_name =  self.mvc_name.split('-')[-1]
            
        self.dict_compounds = {self.mvc_name:self.mvc, self.lvc_name:self.lvc}
    
    def process_data(self, ref_no = 3):
        self.ref_no = ref_no
        self.df_vle = self.df_vle[self.df_vle.x1 > 0]
        self.df_vle = self.df_vle[self.df_vle.x1 < 1]
        self.ref_val ={}
        for i in range(1,ref_no+1):
            self.ref_val[i] = self.df_vle[self.df_vle.Ref == i]
    def visualize_data(self):
        plt.figure(figsize = (7,7))
        for i in range(1,self.ref_no+1):
            plt.plot(self.ref_val[i]["x1"], self.ref_val[i]["T"], 'x', label = 'Ref' + str(i), 
                     markersize = 7)
        plt.xlabel('Liquid phase mole fraction of '+self.mvc_name, fontsize = 15)
        plt.ylabel("Temperature in K", fontsize = 15)
        plt.legend(loc = 'best')
        plt.grid()
        
        plt.figure(figsize = (7,7))
        for i in range(1,self.ref_no+1):
            plt.plot(self.ref_val[i]["x1"], self.ref_val[i]["y1"], 'x', label = 'Ref' + str(i), 
                     markersize = 7)
        plt.plot([0,1], [0,1])
        plt.xlabel('Liquid phase mole fraction of '+self.mvc_name, fontsize = 15)
        plt.ylabel('Vapour phase mole fraction of '+self.mvc_name, fontsize = 15)
        plt.legend()
        plt.grid()
        
    def get_expt_gamma_1(self,row):
        x1 = row["x1"]
        y1 = row["y1"]
        T  = row["T"]
        Psat = self.mvc.Pvap(T)
        P = row["P"]*1000
        gamma = P*y1/(x1*Psat)
        return gamma

    def get_expt_gamma_2(self,row):
        x2 = 1 - row["x1"] 
        y2 = 1 - row["y1"]
        T = row["T"]
        Psat = self.lvc.Pvap(T)
        P = row["P"]*1000
        gamma = P*y2/(x2*Psat)
        return gamma
    
    def get_expt_gamma(self, plot = True):
        self.df_vle["gamma_1"] = self.df_vle.apply(self.get_expt_gamma_1, axis = 1)
        self.df_vle["gamma_2"] = self.df_vle.apply(self.get_expt_gamma_2, axis = 1)
        for i in range(1,self.ref_no+1):
            self.ref_val[i] = self.df_vle[self.df_vle.Ref == i]
        
        if plot:
            plt.figure(figsize = (7,7))
            for i in range(1,self.ref_no+1):
                plt.plot(self.ref_val[i]["x1"], self.ref_val[i]["gamma_1"], 'x',
                         label = 'Ref' + str(i)+r'-$\gamma_{1}$',  markersize = 7)
                plt.plot(self.ref_val[i]["x1"], self.ref_val[i]["gamma_2"], 'x',
                         label = 'Ref' + str(i)+r'-$\gamma_{2}$',  markersize = 7)
            plt.xlabel('Liquid phase mole fraction of '+self.mvc_name, fontsize = 15)
            plt.ylabel('Activity Coefficient of '+self.mvc_name, fontsize = 15)
            plt.title('Activity Coefficient vs Composition')
            plt.legend()
            plt.grid()
            
    def get_nrtl_gammas(self, row, nrtl):
            x1 = row["x1"]
            x2 = 1 - x1
            T = row["T"]
            dict_gammas = nrtl.get_gammas(T, {self.lvc_name:x2, self.mvc_name:x1})
            return pd.Series({"gamma_nrtl_1":dict_gammas[self.mvc_name], "gamma_nrtl_2": dict_gammas[self.lvc_name]})
        
    def fit_nrtl(self, ref_exclude = 3):
        self.nrtl = NRTL.NRTL([self.mvc_name, self.lvc_name])
        
        name_nrtl = str(self.mvc_name+"-"+self.lvc_name)
        def get_residuals_nrtl_vle(fitting_parameters, nrtl, df_vle):
            [
             a12, a21, 
            ] = fitting_parameters #Fitting Matrix A, and alpha
            
            #self.alpha = 0.3 #0.2 gives a lower cost function than 0.5
            nrtl.populate_matrix("A",{name_nrtl:[a12, a21]})
          
            nrtl.populate_matrix("alpha", {name_nrtl:[self.alpha, self.alpha]})
            
            df_nrtl_gammas = df_vle.apply(self.get_nrtl_gammas, axis=1, args=(nrtl,))
            
            residuals1 = (df_vle.gamma_1 - df_nrtl_gammas.gamma_nrtl_1)/df_vle.gamma_1
            residuals2 = (df_vle.gamma_2 - df_nrtl_gammas.gamma_nrtl_2)/df_vle.gamma_2
            residuals = pd.concat([residuals1, residuals2])
            return residuals
        fitting_parameters_guess = [ 
                             0.0,0.0   #a12, a21
                           ] 
        self.optimized_vle_nrtl_parameters = least_squares(get_residuals_nrtl_vle,
                                                             fitting_parameters_guess,
                                                             args = (self.nrtl, 
                                                                     self.df_vle[self.df_vle.Ref != ref_exclude]))
        return self.nrtl, self.optimized_vle_nrtl_parameters['x'], self.optimized_vle_nrtl_parameters['cost']
    
    def observe_fit(self, Gex = True):
        self.df_nrtl_gammas = self.df_vle.sort_values("x1").apply(self.get_nrtl_gammas, axis=1, args=(self.nrtl,))
        plt.figure(figsize = (7,7))
        for i in range(1,self.ref_no+1):
                plt.plot(self.ref_val[i]["x1"], self.ref_val[i]["gamma_1"], 'x',
                         label = 'Ref' + str(i)+r'-$\gamma_{1}$',  markersize = 7)
                plt.plot(self.ref_val[i]["x1"], self.ref_val[i]["gamma_2"], 'x',
                         label = 'Ref' + str(i)+r'-$\gamma_{2}$',  markersize = 7)
                
        plt.plot(self.df_vle.sort_values("x1")["x1"], self.df_nrtl_gammas["gamma_nrtl_1"],'k', label = 'Fitted Curve')
        plt.plot(self.df_vle.sort_values("x1")["x1"], self.df_nrtl_gammas["gamma_nrtl_2"],'k')
        plt.xlabel('Liquid phase mole fraction of '+self.mvc_name, fontsize = 15)
        plt.ylabel('Activity Coefficient of '+self.mvc_name, fontsize = 15)
        plt.title('Activity Coefficient vs Composition')
        plt.legend()
        plt.grid()    
       
        self.Ge = {}
        for i in range(1,self.ref_no+1):
            self.Ge[i] = 8.314462*self.ref_val[i]["T"]*(self.ref_val[i]['x1']*np.log(self.ref_val[i]['gamma_1']) + (1-self.ref_val[i]['x1'])*np.log(self.ref_val[i]['gamma_2']))
        self.Ge_nrtl_pred = 8.314463*self.df_vle.sort_values("x1")["T"]*(self.df_vle.sort_values("x1")["x1"]*np.log(self.df_nrtl_gammas["gamma_nrtl_1"]) + (1-self.df_vle.sort_values("x1")["x1"])*np.log(self.df_nrtl_gammas["gamma_nrtl_2"]))
        if Gex:
            plt.figure(figsize = (7,7))
            for i in range(1,self.ref_no+1):
                plt.plot(self.ref_val[i]["x1"], self.Ge[i], 'x',
                         label = 'Ref' + str(i)+r'-$G^{E}$',  markersize = 7)
            plt.plot(self.df_vle.sort_values("x1")["x1"], self.Ge_nrtl_pred, '^', 
                         label = r'-$G^{E}$'+' Predicted',  markersize = 7)
            plt.ylabel(r'$G^{E} \; \; \frac{J}{mol}$')
            plt.xlabel('Liquid phase mole fraction of '+self.mvc_name, fontsize = 15) 
            plt.title('Excess Gibbs Free Energy vs Composition')
            plt.grid()
            plt.legend()
            
        
        
        