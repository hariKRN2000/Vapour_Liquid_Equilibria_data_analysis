# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 19:40:46 2021

@author: hp1
"""
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import style
style.use("seaborn-dark-palette")
import os


class AnimateConc:
    def show_anim(self, rename = True):
        sol = self.sol
        t = self.t
        xB = self.xb ; xD = self.xd ; F = self.F
        ispot1, ispot2, ispot3 = int(self.total_stages/3), int(2*self.total_stages/3), int(3*self.total_stages/3)
       
        fig = plt.figure(facecolor='white', figsize=(6, 6))
        plt.suptitle("Concentration vs Time ")
        plt.grid(True)
        plt.xlabel('Time, hr') ; plt.ylabel('Mole Fraction')            
        plt.xlim([0,t[-1]]); plt.ylim([self.xb-0.1,self.xd+0.1])
        textstr =("Mole Fraction :\n "+ 
                 "At Top : %.2f\n" %(xD)+ 
                 "At Bottom : %.2f\n" %(xB)+
                 "Feed Rate (kmol/hr): %.1f" %(F))
        props = dict(boxstyle='round', facecolor='cyan', alpha=0.6)
        fig.text(0.5, 0.2, textstr, fontsize=10,
        verticalalignment='baseline', bbox=props,  horizontalalignment='left' );
        line1, = plt.plot(0,self.xf, 'r', label= 'Top Tray' , lw =2 )
        line2, = plt.plot(0,self.xf, 'g', lw =2 )
        line3, = plt.plot(0,self.xf, 'y', lw =2 )
        line4, = plt.plot(0,self.xf, 'orange', lw =2 )
        line5, = plt.plot(0,self.xf, 'b', label= 'Bottom Tray' , lw =2 )
        plt.legend(loc='best', fontsize=5 , prop={"size":15})
        time , conc1, conc2, conc3, conc4, conc5 = [],[],[],[],[],[]
        
        def animate_plot(k):
            plt.title(r"$t = %.1f \; hr$" % t[k])
            time.append(t[k])
            
            conc1.append(sol[k,1]); conc2.append(sol[k,ispot1]) ; conc3.append(sol[k,ispot2]); conc4.append(sol[k,ispot3])
            conc5.append(sol[k,-2])
            line1.set_data(time, conc1);line2.set_data(time, conc2);line3.set_data(time, conc3);line4.set_data(time, conc4)
            line5.set_data(time, conc5)
            return line1, line2, line3, line4, line5
        
        animation = FuncAnimation(fig, animate_plot, frames = len(self.t), interval = 20, blit = True);
        if rename:
            name = 'distillation_'+str(self.names_list[0])+'_'+str(self.names_list[1])+'.gif'
        else:
            name = str(input("Enter the name you want for the gif file(including the .gif extension:) \n :"))
            
        cwd = os.getcwd()
        myfile = cwd+"\\" +name
        if os.path.isfile(myfile):
            os.remove(myfile)    
            
        animation.save(name, writer = 'pillow', fps = int(len(self.t)/3))
        
        #Image.open(name)      
        #IPdisplay.Image(url=name)