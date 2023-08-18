# remove duplicate labels in legend, taken from
# https://stackoverflow.com/questions/19385639/duplicate-items-in-legend-in-matplotlib
# maybe understood?
def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), fontsize="xx-small", ncol=1)