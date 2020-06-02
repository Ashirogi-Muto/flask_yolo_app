import pandas as pd
import matplotlib
data_frame = pd.read_csv('plot_data.csv')
plot = data_frame.plot(x="pods")
figure = plot.get_figure()
figure.savefig("output_plot.png")
