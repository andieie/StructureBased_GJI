
#Plotting parameters 

#Figure cosmetics for publications. """
from matplotlib import rcParams
import matplotlib.pyplot as plt

def set_legend_white(ax):
    legend = ax.legend()
    for text in legend.get_texts():
        text.set_color("white")

def set_style(fs=8):

    rcParams['pdf.fonttype'] = 42

    plt.rcParams['font.family'] = "Arial"
    plt.rcParams['xtick.minor.pad'] = 1
    plt.rcParams['ytick.minor.pad'] = 1

    plt.rcParams['font.weight'] = 'regular'
    plt.rcParams['font.size'] = fs
    plt.rcParams['axes.labelsize'] = fs
    plt.rcParams['axes.titlepad'] = 4
    plt.rcParams['axes.labelweight'] = 'regular'
    plt.rcParams['xtick.labelsize'] = fs
    plt.rcParams['ytick.labelsize'] = fs
    plt.rcParams['legend.fontsize'] = fs
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["savefig.pad_inches"] = 0.1
    plt.rcParams["savefig.transparent"] = False



scatter_kwargs = {
    'shots': {
        's': 50, 
        'c': 'limegreen', 
        'edgecolor': 'black',
        'marker': 'D',
        'label': 'Shots'
    },
    'receivers': {
        's': 50, 
        'edgecolor': 'black',
        'c': 'orange',
        'marker': 'D',
        'label': 'Receivers'
    },
    'fixed_points': {
        's': 80, 
        'c': 'red', 
        'edgecolor' : 'black',
        'label': 'Fixed points'
    },
    'moving_points': {
        's': 80, 
        'c': 'green', 
        'edgecolor' : 'black',
        'label': 'Moving points'
    },
    'ert_sensors': {
        's': 30, 
        'c': 'black', 
        'edgecolor' : 'black',
        'label': 'ERT sensors'
    },
    'true_positions': {
        's': 80, 
        'c': 'white', 
        'edgecolor' : 'black',
        'label': 'True model points',
        'alpha': 0.8
    }
}

pg_show_kwargs = { 
    'tt' : {
    'cMap': 'viridis', 
    'cMin': 2000,
    'cMax': 3000,
    'label': 'Velocity (m/s)',
    'pad': 0.6
},
    'ert' : {
    'cMap': 'Spectral_r',
    'cMin': 100,
    'cMax': 1500,
    'label': 'Resistivity ($\Omega$m)',
    'pad' : 0.6
}
}