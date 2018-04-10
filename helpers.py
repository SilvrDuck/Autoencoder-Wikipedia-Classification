import pickle
import time
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
import pylab


DATA_PATH = Path('data')
   
def write_pickle(data, name, data_path=DATA_PATH, end_with_date=True):
    '''Pickles the data.'''
    with open(file_name(name, data_path, end_with_date), 'wb') as p:
        pickle.dump(data, p, protocol=pickle.HIGHEST_PROTOCOL)
        
def file_name(name, data_path, end_with_date):
    '''Creates file name with possibly the current time at the end.'''
    return f'{data_path / name}{now_str() if end_with_date else ""}.pickle'
        
def read_pickle(name):
    '''Read a pickle file from its name.'''
    with open(name, 'rb') as p:
        return pickle.load(p)
    
def most_recent_dataset(data_path=DATA_PATH):
    '''Get the most recent dataset in data_path (based on the end of its name).'''
    files = [f for f in data_path.iterdir() if f.is_file()]
    dates = [str(file)[-22:-7] for file in files]
    file_to_load = files[dates.index(max(dates))]
    return read_pickle(file_to_load)
    
def now_str():
    '''Return string of the current time.'''
    return time.strftime('_%Y-%m-%d-%H%M')
    
## Plots

def define_colors(colors=[[0x65, 0xDE, 0xF1], [0xA8, 0XDC, 0XD1], [0xDC, 0XE2, 0XC8], [0x99, 0X89, 0X20], 
                       [0xF1, 0X7F, 0X29], [0x6A, 0X8E, 0X7F], [0xED, 0XC7, 0XCF], [0x6A, 0x6A, 0x6A]]):
    '''Converts array of colors of the form [[r,g,b],[r,g,b]] to the same shape, but with r,g,b between 0 and 1 instead of 0 and 255.'''
    return [[r/255, g/255, b/255] for r, g, b in [col for col in colors]]

def plot_2D(X, y, le, in_range=None):  
    '''Plots a scatter plot of the data.'''
    pca = PCA(n_components=2)
    pca.fit(X)
    X_viz = pca.transform(X)

    cols = define_colors()
    colors = [cols[int(i % len(cols))] for i in y]
           
    pylab.figure(figsize=(10,10))
    
    legends = []
    for i, c in enumerate(cols):
        legends.append(mpatches.Patch(color=c, label=le.inverse_transform(i)))
    pylab.legend(handles=legends)    
    
    if in_range is not None:
        pylab.xlim(in_range[0])
        pylab.ylim(in_range[1])

    pylab.scatter(X_viz[:,0], X_viz[:,1], c=colors)#, cmap=pylab.cm.cool)    
    pylab.show()
    
    
# The following functions were taken from the EPFL CS-456 course

def prepare_standardplot(title, xlabel):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(title)
    ax1.set_ylabel('loss')
    ax1.set_xlabel(xlabel)
    ax1.set_yscale('log')
    ax2.set_ylabel('accuracy [% correct]')
    ax2.set_xlabel(xlabel)
    return fig, ax1, ax2

def finalize_standardplot(fig, ax1, ax2):
    ax1handles, ax1labels = ax1.get_legend_handles_labels()
    if len(ax1labels) > 0:
        ax1.legend(ax1handles, ax1labels)
    ax2handles, ax2labels = ax2.get_legend_handles_labels()
    if len(ax2labels) > 0:
        ax2.legend(ax2handles, ax2labels)
    fig.tight_layout()
    plt.subplots_adjust(top=0.9)

def plot_history(history, title):
    fig, ax1, ax2 = prepare_standardplot(title, 'epoch')
    ax1.plot(history.history['loss'], label = "training")
    ax1.plot(history.history['val_loss'], label = "validation")
    ax2.plot(history.history['acc'], label = "training")
    ax2.plot(history.history['val_acc'], label = "validation")
    finalize_standardplot(fig, ax1, ax2)
    return fig

