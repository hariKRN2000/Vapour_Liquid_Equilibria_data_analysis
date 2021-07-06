# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 19:45:17 2021

@author: hp1
"""
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import style
style.use("seaborn-dark-palette")
import numpy as np
import os


class AnimateStage:
    def animate_stage(self, rename = True):
        sol = self.sol
        t = self.t
        
        
        fig = plt.figure(facecolor='white', figsize=(6, 6))
        plt.suptitle("Stage Composition vs Time ")
        
        instack = np.ones((1,self.total_stages)) * self.xf
        line = plt.pcolormesh(instack, cmap = 'jet')
        plt.xlabel('Stages', fontsize = 15)
        
        plt.colorbar()
        
        time, conc = [], []
        
        def animate_plot(i):
            plt.title(r"$t = %.1f \; hr$" % t[i])
            time.append(t[i])
            
            conc.append(sol[i,1:-1])
            
            line.set_array(sol[i,1:-1].ravel())
            
            return line
        
        animation = FuncAnimation(fig, animate_plot, frames = len(self.t), interval = 20, blit = False);
        
        
        if rename:
            name = 'distillation_stage'+str(self.names_list[0])+'_'+str(self.names_list[1])+'.gif' 
        else:
            name = str(input("Enter the name you want for the gif file(including the .gif extension:) \n :"))
            
        cwd = os.getcwd()
        myfile = cwd+"\\" +name
        if os.path.isfile(myfile):
            os.remove(myfile)
            
        animation.save(name, writer = 'pillow', fps = int(len(self.t)/3))
        
            
        #IPdisplay.Image(url=name)