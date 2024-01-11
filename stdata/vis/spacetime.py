import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import geopandas
import shapely


def plot_polygon_collection(
    ax,
    geoms,
    norm,
    values=None,
    colormap="Set1",
    facecolor=None,
    edgecolor=None,
    alpha=1.0,
    linewidth=1.0,
    **kwargs,
):
    """Plot a collection of Polygon geometries"""
    patches = []

    for poly in geoms:
        a = np.asarray(poly.exterior.xy).T
        if poly.has_z:
            poly = shapely.geometry.Polygon(poly.exterior.xy, z=poly.z)

        patches.append(Polygon(a))

    patches = PatchCollection(
        patches,
        facecolor=facecolor,
        linewidth=linewidth,
        edgecolor=edgecolor,
        alpha=alpha,
        norm=norm,
        **kwargs,
    )

    if values is not None:
        patches.set_array(values)
        patches.set_cmap(colormap)

    ax.add_collection(patches, autolim=True)
    ax.autoscale_view()
    return patches


class ST_GridPlot(object):
    def __init__(
        self,
        columns,
        col,
        fig,
        ax,
        train_df,
        test_df,
        cax_on_right=True,
        norm_on_training=True,
        label="",
        geopandas_flag=False,
    ):
        self.columns = columns
        self.col = col
        self.geopandas_flag = geopandas_flag
        self.fig = fig
        self.ax = ax
        self.train_df = train_df
        self.test_df = test_df
        self.norm_on_training = norm_on_training
        self.right_flag = cax_on_right
        self.label = label
        self.cmap = None
        self.grid_plot = None

    def get_spatial_slice(self, epoch):
        s = self.test_df[self.test_df[self.columns["epoch"]] == epoch]
        if len(s) == 0:
            return None, None, None
        return (
            s[self.columns["x"]].astype(np.float32),
            s[self.columns["y"]].astype(np.float32),
            s[self.columns[self.col]].astype(np.float32),
        )

    def get_data(self, epoch):
        x_train, y_train, z_train = self.get_spatial_slice(epoch)
        if x_train is None:
            return None, None

        print(z_train.shape)
        z_train = np.array(z_train)

        s = np.c_[x_train, y_train]
        n = int(np.sqrt(z_train.shape[0]))
        grid_index = np.lexsort((s[:, 0], s[:, 1]))
        s = s[grid_index, :]
        z_train = z_train[grid_index]
        z_train = (z_train).reshape(n, n)
        return s, z_train

    def setup(self):
        df = self.test_df
        if self.norm_on_training:
            df = self.train_df

        self.norm = matplotlib.colors.Normalize(
            vmin=np.min(df[self.columns[self.col]]),
            vmax=np.max(df[self.columns[self.col]]),
        )

        # setup color bar
        self.divider = make_axes_locatable(self.ax)
        dir_str = "left"
        if self.right_flag:
            dir_str = "right"
        self.color_bar_ax = self.divider.append_axes(dir_str, size="5%", pad=0.05)

    def update(self, epoch):
        if self.geopandas_flag:
            # If grid_plot is init with zero patches then we need to create them
            if self.grid_plot is None:
                self.plot(epoch)
                return
            df = self.test_df[self.test_df[self.columns["epoch"]] == epoch]
            df = df.sort_values(self.columns["id"])
            self.grid_plot.set_array(df[self.columns[self.col]])
        else:
            s, z_train = self.get_data(epoch)
            if z_train is None:
                if hasattr(self, "grid_plot"):
                    self.grid_plot.set_data([[]])
            else:
                if hasattr(self, "grid_plot"):
                    self.grid_plot.set_data(z_train)
                else:
                    self.plot(epoch)
        self.fig.canvas.draw()

    def plot(self, epoch):
        if self.geopandas_flag:
            df = self.test_df[self.test_df[self.columns["epoch"]] == epoch]
            # If grid_plot is init with zero patches we cannot plot later
            if df.shape[0] == 0:
                self.grid_plot = None
                return
            df = df.sort_values(self.columns["id"])
            geo_series = geopandas.GeoSeries(df["geom"])
            self.grid_plot = plot_polygon_collection(self.ax, geo_series, self.norm)
            self.grid_plot.set_array(df[self.columns[self.col]])
        else:
            s, z_train = self.get_data(epoch)
            if z_train is None:
                return
            # get extents
            min_x = s[0, 0]
            min_y = s[0, 1]
            max_x = s[s.shape[0] - 1, 0]
            max_y = s[s.shape[0] - 1, 1]
            self.grid_plot = self.ax.imshow(
                z_train,
                origin="lower",
                cmap=self.cmap,
                norm=self.norm,
                aspect="auto",
                extent=[min_x, max_x, min_y, max_y],
            )
            self.fig.colorbar(
                self.grid_plot, cax=self.color_bar_ax, orientation="vertical"
            )
        self.ax.set_title(f"Epoch {epoch} {self.label}")
        return self.grid_plot


class ST_SliderPlot(object):
    def __init__(self, fig, ax, unique_vals, callback):
        self.fig = fig
        self.ax = ax
        self.unique_vals = unique_vals
        self.callback = callback

    def set_text_format(self):
        datetime.fromtimestamp(1472860800).strftime("%Y-%m-%d %H")
        self.slider.valtext.set_text(
            datetime.fromtimestamp(self.slider.val).strftime("%Y-%m-%d %H")
        )

    def setup(self, start_val):
        self.slider = Slider(
            self.ax,
            "Date",
            np.min(self.unique_vals),
            np.max(self.unique_vals),
            valinit=start_val,
        )
        self.set_text_format()
        self.slider.on_changed(self.update)

    def update(self, i):
        cur_epoch_i = np.abs(self.unique_vals - i).argmin()
        cur_epoch = self.unique_vals[cur_epoch_i]
        self.set_text_format()
        self.callback(cur_epoch)


class ST_TimeSeriesPlot(object):
    def __init__(self, columns, fig, ax, train_df, test_df, test_start, grid_plot_flag):
        self.columns = columns
        self.fig = fig
        self.ax = ax
        self.train_df = train_df
        self.test_df = test_df
        self.min_test_epoch = np.min(self.train_df[columns["epoch"]])
        self.max_test_epoch = np.max(self.train_df[columns["epoch"]])
        self.test_start_epoch = test_start or self.min_test_epoch

    def setup(self):
        pass

    def get_time_series(self, _id, data):
        d = self.train_df[self.train_df[self.columns["id"]] == _id]

        print(f"Plotting timeseries: {_id}")

        d = d.sort_values(by=self.columns["epoch"])

        epochs = d[self.columns["epoch"]].astype(np.float32)
        var = d[self.columns["var"]].astype(np.float32)
        pred = d[self.columns["pred"]].astype(np.float32)
        observed = d[self.columns["observed"]].astype(np.float32)
        return epochs, var, pred, observed

    def plot(self, _id):
        epochs, var, pred, observed = self.get_time_series(_id, self.train_df)

        self.var_plot = self.ax.fill_between(
            epochs, pred - 1.96 * np.sqrt(var), pred + 1.96 * np.sqrt(var)
        )
        self.observed_scatter = self.ax.scatter(epochs, observed)
        self.pred_plot = self.ax.plot(epochs, pred)
        self.ax.set_xlim([self.min_test_epoch, self.max_test_epoch])
        self.min_line = self.ax.axvline(self.min_test_epoch)
        self.max_line = self.ax.axvline(self.max_test_epoch)
        self.test_start_line = self.ax.axvline(self.test_start_epoch)

    def plot_cur_epoch(self, epoch):
        self.cur_epoch_line = self.ax.axvline(epoch, ymin=0.25, ymax=1.0)

    def update_cur_epoch(self, epoch):
        self.cur_epoch_line.remove()
        self.plot_cur_epoch(epoch)

    def update(self, _id):
        try:
            self.var_plot.remove()
            self.observed_scatter.remove()
            self.pred_plot[0].remove()
            self.min_line.remove()
            self.max_line.remove()
        except ValueError as e:
            # already been removed so need to remove again
            print(e)
            pass

        self.plot(_id)


class ST_ScatterPlot(object):
    def __init__(self, columns, fig, ax, grid_plot, grid_plot_flag, callback, train_df):
        self.columns = columns
        self.fig = fig
        self.ax = ax
        self.train_df = train_df
        self.cmap = None
        if grid_plot_flag:
            self.norm = grid_plot.norm
        else:
            print("min: ", np.min(self.train_df[self.columns["pred"]]))
            print("max: ", np.max(self.train_df[self.columns["pred"]]))
            self.norm = matplotlib.colors.Normalize(
                vmin=np.min(self.train_df[self.columns["pred"]]), vmax=1000
            )

        self.callback = callback
        self.cur_epoch = None

    def setup(self):
        self.fig.canvas.mpl_connect("button_release_event", self.on_plot_hover)

    def get_closest_observed(self, p):
        d = np.array(self.train_df[[self.columns["x"], self.columns["y"]]]).astype(
            np.float32
        )
        dists = np.sum((d - p) ** 2, axis=1)
        i = np.argmin(dists)
        # if dists[i] <= 1e-4:
        if dists[i] <= 0.02:
            return self.train_df.iloc[i][self.columns["id"]]
        else:
            return None

    def on_plot_hover(self, event):
        if True or event.inaxes is self.ax:
            p = event.xdata, event.ydata
            _id = self.get_closest_observed(p)
            if _id is not None:
                self.callback(_id)

    def get_spatial_slice(self, epoch, data, _id=None):
        s = data[data[self.columns["epoch"]] == epoch]
        if _id:
            s = s[s[self.columns["id"]] == _id]
        return (
            s[self.columns["x"]].astype(np.float32),
            s[self.columns["y"]].astype(np.float32),
            s[self.columns["pred"]].astype(np.float32),
        )

    def plot(self, epoch):
        self.cur_epoch = epoch
        x, y, z = self.get_spatial_slice(epoch, self.train_df)
        self.scatter = self.ax.scatter(
            x, y, c=z, norm=self.norm, cmap=self.cmap, edgecolors="w"
        )

    def plot_active(self, _id):
        self.cur_id = _id
        x, y, z = self.get_spatial_slice(self.cur_epoch, self.train_df, _id)
        self.active_scatter = self.ax.scatter(
            x, y, c=z, norm=self.norm, cmap=self.cmap, edgecolors="y"
        )

    def update(self, epoch):
        self.scatter.remove()
        self.plot(epoch)
        self.update_active(self.cur_id)

    def update_active(self, _id):
        self.cur_id = _id
        self.active_scatter.remove()
        self.plot_active(_id)


class SpaceTimeVisualise(object):
    def __init__(
        self, train_df, test_df, sat_df=None, geopandas_flag=True, test_start=None
    ):
        columns = {
            "id": "id",
            "epoch": "epoch",
            "x": "lon",
            "y": "lat",
            "datetime": "datetime",
            "observed": "observed",
            "pred": "pred",
            "var": "var",
        }
        self.columns = columns
        self.geopandas_flag = geopandas_flag
        self.train_df = train_df
        self.test_df = test_df
        self.sat_df = sat_df

        self.grid_plot_flag = not (self.test_df is None)

        self.min_time = np.min(self.train_df[columns["epoch"]])
        self.max_time = np.max(self.train_df[columns["epoch"]])
        if test_start:
            self.test_start = test_start
        else:
            self.test_start = self.min_time
        self.unique_epochs = np.unique(self.train_df[columns["epoch"]])
        self.unique_ids = np.unique(self.train_df[columns["id"]])
        self.start_epoch = self.unique_epochs[-1]
        self.start_id = self.unique_ids[0]

    def update_timeseries(self, _id):
        self.time_series_plot.update(_id)
        self.val_scatter_plot.update_active(_id)
        self.fig.canvas.draw_idle()

    def update_epoch(self, epoch):
        if self.grid_plot_flag:
            self.val_grid_plot.update(epoch)
            self.var_grid_plot.update(epoch)
        self.time_series_plot.update_cur_epoch(epoch)
        self.val_scatter_plot.update(epoch)

    def show(self):
        self.fig = plt.figure(figsize=(12, 6))

        self.gs = matplotlib.gridspec.GridSpec(12, 4, wspace=0.25, hspace=0.25)
        self.grid_plot_1_ax = self.fig.add_subplot(
            self.gs[0:7, 0:2]
        )  # first row, first col
        self.grid_plot_2_ax = self.fig.add_subplot(
            self.gs[0:7, 2:4]
        )  # first row, second col
        self.epoch_slider_ax = self.fig.add_subplot(
            self.gs[7, 1:3]
        )  # first row, second col
        self.time_series_ax = self.fig.add_subplot(self.gs[8:11, :])  # full second row
        self.scale_slider_ax = self.fig.add_subplot(
            self.gs[11, 1:3]
        )  # first row, second col
        if self.grid_plot_flag:
            self.val_grid_plot = ST_GridPlot(
                self.columns,
                "pred",
                self.fig,
                self.grid_plot_1_ax,
                self.train_df,
                self.test_df,
                cax_on_right=False,
                norm_on_training=True,
                label="NO2",
                geopandas_flag=self.geopandas_flag,
            )
            self.val_grid_plot.setup()
            self.var_grid_plot = ST_GridPlot(
                self.columns,
                "var",
                self.fig,
                self.grid_plot_2_ax,
                self.train_df,
                self.test_df,
                cax_on_right=False,
                norm_on_training=True,
                label="NO2",
                geopandas_flag=self.geopandas_flag,
            )
            self.var_grid_plot.setup()
        else:
            self.val_grid_plot = None
            self.var_grid_plot = None
        self.val_scatter_plot = ST_ScatterPlot(
            self.columns,
            self.fig,
            self.grid_plot_1_ax,
            self.val_grid_plot,
            self.grid_plot_flag,
            self.update_timeseries,
            self.train_df,
        )
        self.val_scatter_plot.setup()
        self.slider_plot = ST_SliderPlot(
            self.fig, self.epoch_slider_ax, self.unique_epochs, self.update_epoch
        )
        self.slider_plot.setup(self.start_epoch)
        self.time_series_plot = ST_TimeSeriesPlot(
            self.columns,
            self.fig,
            self.time_series_ax,
            self.train_df,
            self.test_df,
            self.test_start,
            self.grid_plot_flag,
        )
        self.time_series_plot.setup()

        if self.grid_plot_flag:
            self.val_grid_plot.plot(self.start_epoch)
            self.var_grid_plot.plot(self.start_epoch)

        self.val_scatter_plot.plot(self.start_epoch)

        self.time_series_plot.plot_cur_epoch(self.start_epoch)
        self.time_series_plot.plot(self.start_id)

        self.val_scatter_plot.plot_active(self.start_id)

        if self.sat_df is not None:
            self.time_series_plot.ax.scatter(
                self.sat_df["epoch"], self.sat_df[self.columns["observed"]], alpha=0.4
            )

        # Add a vertical line at the end of training time
        self.time_series_ax.axvline(
            self.start_epoch, color="red", linestyle="--", label="End of Training"
        )
        self.time_series_ax.legend()

        plt.show()
