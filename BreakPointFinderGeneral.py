# %matplotlib tk
from matplotlib import pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.gridspec import GridSpec

import numpy as np
from scipy.optimize import curve_fit
import gc
import time
from scipy.signal import savgol_filter
import pandas as pd
from numba import jit,njit,prange,objmode 
import os
from pathlib import Path
from glob import glob
import pickle
import copy

from datetime import datetime

# SPDF API
# from cdasws import CdasWs
# cdas = CdasWs()

au_to_km = 1.496e8  # Conversion factor
rsun     = 696340   # Sun radius in units of  [km]

from scipy import constants
psp_ref_point   =  0.06176464216946593 # in [au]
mu0             =  constants.mu_0  # Vacuum magnetic permeability [N A^-2]
mu_0             =  constants.mu_0  # Vacuum magnetic permeability [N A^-2]
m_p             =  constants.m_p   # Proton mass [kg]
k          = constants.k                  # Boltzman's constant     [j/K]
au_to_km   = 1.496e8
T_to_Gauss = 1e4

from TSUtilities import curve_fit_log_wrap

class BreakPointFinder:
    def __init__(
        self, x, y, diagnostics=None,
        label = None, secondary = None, third=None,
        app = '4pt_slope',
        view_fit = True, no_avg =False, import_4pt_fit = None,
        import_4pt_intersect = None, import_2pt_avg = None,
        import_coi = None, suptitle = None
        ):
        """
        Initialize the class with fig, ax, and plot data
        data: {'xdata':, 'ydata':}
        Input: smoothed line with smoothed freqs(x) and smoothed PSD (y),
        app: 4pt_slope, 2pt_avg
        """

        # find breakpoint

        self.x = x
        self.y = y
        self.secondary = secondary
        self.third = third
        self.label = label
        self.app = app
        self.view_fit = view_fit
        self.app_list = ['4pt_slope','2pt_avg']
        self.import_4pt_fit = import_4pt_fit
        self.import_4pt_intersect = import_4pt_intersect
        self.import_2pt_avg = import_2pt_avg
        self.import_coi = import_coi
        self.suptitle = suptitle
        
        self.no_avg = no_avg
        # self.struc_funcs = struc_funcs

        self.diagnostics_template = {
            "QualityFlag": 0,
            '4pt_slope':{
                'fit1': None,
                'fit2': None,
                'Intersect': np.nan,
                'Intersect_r': np.nan,
                'xpts': [np.nan, np.nan, np.nan, np.nan]
            },
            '2pt_avg':{
                'fit': None,
                'avg_y': np.nan,
                'avg_x': np.nan,
                'std_y': np.nan,
                'x1': np.nan,
                'x2': np.nan
                }
        }

        if diagnostics is None:
            print("diagnostics is None")
            self.diagnostics = {}
        else:
            print("diagnostics is not None")
            self.diagnostics = diagnostics

        for k in self.diagnostics_template.keys():
            if k not in self.diagnostics.keys():
                self.diagnostics[k] = copy.deepcopy(self.diagnostics_template)[k]

        self.AxesInit()
        self.FigureInit()
        self.connect()

        # import intersects
        if self.import_4pt_intersect is not None:
            if ~(np.isnan(self.import_4pt_intersect)):
                self.intersect_r = self.import_4pt_intersect
                self.ResetFigure()

        # import 4pts
        if (self.import_4pt_fit is not None):
            if (len(self.import_4pt_fit)==4) & ~(np.isnan(self.import_4pt_fit[0])):
                print("Importing xpts...")
                self.calculate_4pt_slope()
                self.ResetFigure()

        # import 2pts
        if (self.import_2pt_avg is not None):
            if (len(self.import_2pt_avg)==2) & ~(np.isnan(self.import_2pt_avg[0])):
                print("Importing 2pt for avg...")
                self.calculate_2pt_avg()
                self.ResetFigure()


        
    def FigureInit(self):

        self.xs = []
        self.ys = []

        # init Cross Hair
        self.CrossHairInit()

        # plot PSD
        self.PlotPSD()

        # plot fit
        if self.view_fit:
            print("app=%s" %(self.app))
            if self.app == '4pt_slope':
                try:    
                    self.DrawFitLine()
                except:
                    pass
            elif self.app == '2pt_avg':
                try:
                    self.DrawAvgLine()
                except:
                    pass
            else:
                pass
        
        # Redraw the canvas
        self.DrawArts()

    def AxesInit(self):

        fig, ax = plt.subplots(1, figsize = [8,8])
        self.fig = fig
        self.arts = {
            "PSD": {'ax': ax}
        }



    def PlotPSD(self):
        view_fit = self.view_fit
        ax = self.arts['PSD']['ax']

        if self.secondary is not None:
            l2 = ax.plot(
                self.secondary['x'], 
                self.secondary['y'], 
                label = self.secondary['label'],
                color = 'C0'
            )

        if self.third is not None:
            l3 = ax.plot(
                self.third['x'], 
                self.third['y'], 
                label = self.third['label'],
                color = 'C2',
                ls = '--',
                lw = 1.5
            )

        l1 = ax.plot(self.x, self.y, label = self.label, color = 'C1')

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(loc = 3, fontsize = 'x-large')

        self.arts['PSD']['line_PSD'] = l1

        ax.set_title(
            "QF: %d" %(self.diagnostics['QualityFlag']),
            fontsize = 'xx-large'
        )
        
        if self.suptitle is not None:
            plt.gcf().suptitle(self.suptitle, fontsize = 'x-large')

        # draw cone of influence for wavelets
        if self.import_coi is not None:
            try:
                f0 = self.import_coi['range'][0]
                f1 = self.import_coi['range'][1]
                label = self.import_coi['label']
                self.arts['PSD']['coi_box'] = ax.axvspan(f0, f1, color = 'r', alpha = 0.05, label = label)
            except:
                raise ValueError(
                    "import_coi should be dictionary {'range': [f0,f1], 'label':label}"
                )

        self.DrawArts()



    # -----------------  Applications ----------------- #


    def calculate_4pt_slope(self):
        """ Calculate Slope and intersect"""
        ax = self.arts['PSD']['ax']
        art = self.arts['PSD']

        # import_4pt_xpts
        if self.import_4pt_fit is not None:
            self.xs = self.import_4pt_fit
            self.import_4pt_fit = None

        if len(self.xs) == 4:
            
            xs1 = np.min(self.xs[0:2])
            xe1 = np.max(self.xs[0:2])
            
            xs2 = np.min(self.xs[2:4])
            xe2 = np.max(self.xs[2:4])
            
            xdata, ydata = self.x, self.y
            ind = (np.isnan(xdata)) & (np.isnan(ydata))
            self.indtest = ind
            
            fit1,_,_,flag1 = curve_fit_log_wrap(xdata[~ind],ydata[~ind],xs1,xe1)
            fit2,_,_,flag2 = curve_fit_log_wrap(xdata[~ind],ydata[~ind],xs2,xe2)
            
            if flag1 | flag2:
                print("No enough Data!")
                print("Cleaning points...")
                self.xs = []
                self.ys = []
                art['RedCrossesLine'].set_data(self.xs, self.ys)
                art['RedCrossesLine'].figure.canvas.draw()
            else:
                
                intersect = 10**((fit1[0][0]-fit2[0][0])/(fit2[0][1]-fit1[0][1])) 
                
                self.diagnostics['4pt_slope']['Intersect'] = intersect
                self.diagnostics['4pt_slope']['fit1'] = fit1
                self.diagnostics['4pt_slope']['fit2'] = fit2
                self.diagnostics['4pt_slope']['xpts'] = [xs1, xe1, xs2, xe2]
                
                self.DrawFitLine()
                # self.ResetFigure()
        else:
            pass

    def DrawFitLine(self):
        """ Draw the fit line and intersect """
        ax = self.arts['PSD']['ax']
        fig = self.fig
        
        # create art
        self.arts['FitLine'] = {}
        self.arts['FitLine']['ax'] = ax

        # Clean legend
        self.arts['FitLine']['legend'] = ax.legend([])
        
        xdata, ydata = self.x, self.y
        
        try:
            intersect = self.diagnostics['4pt_slope']['Intersect']
            self.intersect = intersect
            fit1 = self.diagnostics['4pt_slope']['fit1']
            fit2 = self.diagnostics['4pt_slope']['fit2']
            
            f1 = lambda x: (10**fit1[0][0])*x**(fit1[0][1])
            f2 = lambda x: (10**fit2[0][0])*x**(fit2[0][1])

            self.arts['FitLine']['fitline1'], = ax.loglog(xdata, f1(xdata), color='gray', lw='2', alpha = 0.8, 
                                       label = r'$\alpha_B$'+' = %.4f' %(fit1[0][1]))
            self.arts['FitLine']['fitline2'], = ax.loglog(xdata, f2(xdata), color='gray', lw='2', alpha = 0.8, 
                                       label = r'$\alpha_B$'+' = %.4f' %(fit2[0][1]))
            self.arts['FitLine']['intersect'] = ax.axvline(intersect, color='r', lw='2', 
                                        label = "%.4f=%.2f min" %(np.log10(intersect), 1/intersect/60))
        except:
            pass
            # raise ValueError("Draw intersect failed!")
        

        try:
            if np.isnan(self.intersect_r):
                pass
            else:
                self.diagnostics['4pt_slope']['Intersect_r'] = self.intersect_r
                self.diagnostics['4pt_slope']['Intersect_flag'] = 1
                self.arts['FitLine']['intersect_right'] = ax.axvline(self.intersect_r, color='lightblue', lw='2',label = "%.4f" %(np.log10(self.intersect_r)))
        except:
            pass
            # raise ValueError("Draw intersect R failed!")

        # draw the plot location
        try:
            x1s = self.diagnostics['4pt_slope']['xpts'][0:2]
            x2s = self.diagnostics['4pt_slope']['xpts'][2:]
            plt.scatter(x1s, f1(np.array(x1s)), s = 100, marker = 'x', color = 'r', linewidths=1.5, zorder=5, alpha = 0.8)
            plt.scatter(x2s, f2(np.array(x2s)), s = 100, marker = 'x', color = 'r', linewidths=1.5, zorder=5, alpha = 0.8)
        except:
            # raise ValueError("Draw 4pts failed!")
            pass
            
        self.arts['FitLine']['legend'] = ax.legend(loc = 3, fontsize = 'x-large', frameon=False) 


    def calculate_2pt_avg(self):
        """ Calculate Slope """
        ax = self.arts['PSD']['ax']
        art = self.arts['PSD']

        # import_2pt_xpts
        if self.import_2pt_avg is not None:
            self.xs = self.import_2pt_avg
            self.import_2pt_avg = None

        if len(self.xs) == 2:
            xs1 = np.min(self.xs)
            xe1 = np.max(self.xs)

            xdata, ydata = self.x, self.y

            ind = (np.isnan(xdata)) & (np.isnan(ydata))
            fit1,_,_,flag1 = curve_fit_log_wrap(xdata[~ind],ydata[~ind],xs1,xe1)
            ind2 = (xdata >= xs1) & (xdata <= xe1)

            if flag1:
                print("No enough Data!")
                print("Cleaning points...")
                self.xs = []
                self.ys = []
                art['RedCrossesLine'].set_data(self.xs, self.ys)
                art['RedCrossesLine'].figure.canvas.draw()       
            else:
                
                self.diagnostics['2pt_avg']={
                    'fit': fit1,
                    'avg_y': np.nanmean(ydata[ind2]),
                    'avg_x': np.nanmean(xdata[ind2]),
                    'std_y': np.nanstd(ydata[ind2]),
                    'x1': xs1,
                    'x2': xe1
                }

                
                self.DrawAvgLine(no_avg = self.no_avg)

    def DrawAvgLine(self, no_avg = False):
        """ Draw the line created by 2pt avg """
        ax = self.arts['PSD']['ax']
        fig = self.fig

        # create art
        self.arts['AvgLine'] = {}
        self.arts['AvgLine']['ax'] = ax

        # clean legend
        self.arts['AvgLine']['legend'] = ax.legend([])

        xdata, ydata = self.x, self.y

        try:
            fit = self.diagnostics['2pt_avg']['fit']
            f = lambda x: (10**fit[0][0])*x**(fit[0][1])

            if no_avg:
                pass
            else:
                self.arts['AvgLine']['avgline'] = ax.axhline(
                    y = self.diagnostics['2pt_avg']['avg_y'],
                    color = 'm', lw = 1, ls = '--', alpha = 0.8,
                    label = r"avg = %.3f, std = %.3f" %(
                        self.diagnostics['2pt_avg']['avg_y'],
                        self.diagnostics['2pt_avg']['std_y']
                    )
                )

            self.arts['AvgLine']['avgfit'] = ax.loglog(
                xdata, f(xdata), color='g', lw=1, ls='--', alpha = 0.8, 
                label = r'$\alpha_B$'+' = %.4f' %(fit[0][1])
            )

        except:
            raise ValueError("Find avg line failed!")

        try:
            x1s = [self.diagnostics['2pt_avg']['x1'], self.diagnostics['2pt_avg']['x2']]
            plt.scatter(x1s, f(np.array(x1s)), s = 100, marker = 'x', color = 'r', linewidths=1.5, zorder=5, alpha = 0.8)
        except:
            # raise ValueError("Draw 4pts failed!")
            pass

        self.arts['AvgLine']['legend'] = ax.legend(loc = 3, fontsize = 'x-large', frameon=False) 


    # -----------------  Visual Part ----------------- #
    def set_cross_hair_visible(self, visible):
        need_redraw = self.horizontal_line.get_visible() != visible
        self.horizontal_line.set_visible(visible)
        self.vertical_line.set_visible(visible)
        self.text.set_visible(visible)
        return need_redraw


    def on_mouse_move(self, event):
        ax = self.arts['PSD']['ax']
        if not event.inaxes:
            need_redraw = self.set_cross_hair_visible(False)
            if need_redraw:
                ax.figure.canvas.draw()
        else:
            self.set_cross_hair_visible(True)
            x, y = event.xdata, event.ydata
            # update the line positions
            self.horizontal_line.set_ydata(y)
            self.vertical_line.set_xdata(x)
            self.text.set_text(
                'x=%1.4f, y=%1.4f' % (np.log10(x), np.log10(y))
            )
            self.text.set_fontsize(15)
            self.text.set_x(0.58)
            self.text.set_y(0.9)
            ax.figure.canvas.draw()


    def on_click_event(self, event):
        ax = self.arts['PSD']['ax']
        art = self.arts['PSD']
        if event.button is MouseButton.LEFT:
            """ Select Points """
            print('click', event) 

            self.xs.append(event.xdata)
            self.ys.append(event.ydata)
            if len(self.xs)==1:
                line, = ax.plot(self.xs, self.ys, 'rx',ms=12, lw=2)  # empty line
                art['RedCrossesLine'] = line
            else:
                art['RedCrossesLine'].set_data(self.xs, self.ys)
            ax.figure.canvas.draw()
                    
            self.DrawArts()

            # if 4pt, calculate slope
            if self.app == '4pt_slope':
                self.calculate_4pt_slope()    
            elif self.app == '2pt_avg':
                self.calculate_2pt_avg()
            else:
                raise ValueError("app = %s not supported" %(self.app))
                
            
        elif event.button is MouseButton.RIGHT:
            """ select the breakpoint by hand"""
            self.intersect_r = event.xdata
            self.DrawFitLine()

    def on_key_event(self, event):
        """ On Key Event """
        
        if event.key == 'q':
            """ Exit """
            # self.SaveProgress()
            self.disconnect()
            self.fig.clf()
            plt.close('all')
            return 
            
        elif event.key == 'c':
            """ Clear All existing Information """
            
            # Clean the current row
            self.arts['PSD']['ax'].text(0.5, 0.5, 'Cleaning Info...', transform=self.arts['PSD']['ax'].transAxes, fontsize=30, color = 'r')
            self.arts['PSD']['ax'].figure.canvas.draw()
            for k, v in copy.deepcopy(self.diagnostics_template).items():
                self.diagnostics[k] = v

            # clean the caches
            self.xs = []
            self.ys = []
            self.intersect = np.nan
            self.intersect_r = np.nan

            # time.sleep(1)
            self.ResetFigure()
            
        elif event.key == 'r':
            """ Simply reset the figure """
            self.ResetFigure()
            
        elif event.key == 'v':
            """ view current fit """
            if self.view_fit == True:
                self.view_fit = False
            else:
                self.view_fit = True
            self.ResetFigure()
            
        # Setting Quality Flags
        elif event.key == '1':
            self.diagnostics['QualityFlag'] = 1
            self.ResetFigure()
        elif event.key == '2':
            self.diagnostics['QualityFlag'] = 2
            self.ResetFigure()
        elif event.key == '3':
            self.diagnostics['QualityFlag'] = 3
            self.ResetFigure()
        elif event.key == '4':
            self.diagnostics['QualityFlag'] = 4
            self.ResetFigure()
        elif event.key == '5':
            self.diagnostics['QualityFlag'] = 5
            self.ResetFigure()
        elif event.key == '6':
            self.diagnostics['QualityFlag'] = 6
            self.ResetFigure()
        elif event.key == '7':
            self.diagnostics['QualityFlag'] = 7
            self.ResetFigure()
        elif event.key == '8':
            self.diagnostics['QualityFlag'] = 8
            self.ResetFigure()
        elif event.key == '9':
            self.diagnostics['QualityFlag'] = 9
            self.ResetFigure()
        elif event.key == '0':
            self.diagnostics['QualityFlag'] = 0
            self.ResetFigure()
            
    

    def disconnect(self):
        """ Disconnect all callbacks """
        self.fig.canvas.mpl_disconnect(self.cid0)
        self.fig.canvas.mpl_disconnect(self.cid1)
        self.fig.canvas.mpl_disconnect(self.cid2)
        

    def connect(self):
        """ Connect to matplotlib """
        self.cid0 = self.fig.canvas.mpl_connect('button_press_event', self.on_click_event)
        self.cid1 = self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.cid2 = self.fig.canvas.mpl_connect('key_press_event', self.on_key_event)


    def DrawArts(self):
        for art in self.arts:
            try:
                art['ax'].figure.canvas.draw()
            except:
                pass


    def ResetFigure(self):
        """ reset the figure """
        self.disconnect()
        plt.close(self.fig)
        self.AxesInit()
        self.FigureInit()
        self.connect()

    
    def CrossHairInit(self):

        ax = self.arts['PSD']['ax']

        # Show cross hair
        self.horizontal_line = ax.axhline(color='k', lw=0.8, ls='--')
        self.vertical_line = ax.axvline(color='k', lw=0.8, ls='--')
        
        # text location in axes coordinates
        self.text = ax.text(0.72, 0.9, '', transform=ax.transAxes)


