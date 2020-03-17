import numpy as np
import plotly
import plotly.graph_objs as go


def create_plot(plot_data, n_generations):
    trace = go.Scatter(
        x=np.linspace(0, 1, n_generations),
        y=plot_data,
        mode='lines+markers',
        fill='tozeroy'
    )
    data = [trace]
    plotly.offline.plot({"data": data, "layout": go.Layout(title="LunarLander")}, filename="LunarLander_plot.html")
