import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


def rgb2hex(r, g, b):
    return f"#{r:02x}{g:02x}{b:02x}"


def pannel_plot(ds, lat_region, lon_region):
    f, ax = plt.subplots(2, 2, figsize=(24, 12))
    ds.lst_unc_sys.isel(lat=lat_region, lon=lon_region).plot(ax=ax[0, 0], robust=True)
    ds.lst_unc_ran.isel(lat=lat_region, lon=lon_region).plot(ax=ax[0, 1], robust=True)
    ds.lst_unc_loc_atm.isel(lat=lat_region, lon=lon_region).plot(
        ax=ax[1, 0], robust=True
    )
    ds.lst_unc_loc_sfc.isel(lat=lat_region, lon=lon_region).plot(
        ax=ax[1, 1], robust=True
    )
    f.tight_layout
    for i in range(2):
        for j in range(2):
            ax[i, j].set_title("")


def pannel_hist(ds, lat_region, lon_region):
    f, ax = plt.subplots(2, 2, figsize=(24, 12))
    ds.lst_unc_sys.isel(lat=lat_region, lon=lon_region).plot.hist(
        ax=ax[0, 0], bins=50, alpha=0.5, color="steelblue", edgecolor="grey"
    )
    ds.lst_unc_ran.isel(lat=lat_region, lon=lon_region).plot.hist(
        ax=ax[0, 1], bins=50, alpha=0.5, color="steelblue", edgecolor="grey"
    )
    ds.lst_unc_loc_atm.isel(lat=lat_region, lon=lon_region).plot.hist(
        ax=ax[1, 0], bins=50, alpha=0.5, color="steelblue", edgecolor="grey"
    )
    ds.lst_unc_loc_sfc.isel(lat=lat_region, lon=lon_region).plot.hist(
        ax=ax[1, 1], bins=50, alpha=0.5, color="steelblue", edgecolor="grey"
    )
    f.tight_layout
    for i in range(2):
        for j in range(2):
            ax[i, j].set_title("")


class lccs_manger:
    def __init__(self, ds) -> None:
        self.ds = ds
        self.legend = pd.read_csv("Data/ESACCI_LC_Legend_LST_UPDATE.csv")

        # levels = self.ds.lcc.attrs["flag_values"]

    def analyse_region(self):
        ticks = list(np.unique(self.ds.lcc.values))
        ticks.append(220)
        colours_present = []
        labels_present = []
        for index, value in enumerate(self.legend.NB_LAB):
            if value in ticks:
                colours_present.append(self.legend.HEX_COL[index])
                labels_present.append(self.legend.LCCOwnLabel[index])
        labels_present[-1] = ""
        self.plotting_info = {
            "ticks": ticks,
            "colours": colours_present,
            "labels": labels_present,
        }

    def plot_map(self):
        fig, ax = plt.subplots(figsize=(16, 10))
        plot = self.ds.lcc.plot(
            ax=ax,
            levels=self.plotting_info["ticks"],
            colors=self.plotting_info["colours"],
        )
        ax.set_title(f"time = {np.datetime_as_string(self.ds.time.values[0],unit='M')}")
        cb = fig.axes[-1]
        cb.set_yticks(self.plotting_info["ticks"], self.plotting_info["labels"])
        dx = 0 / 72.0
        dy = 14 / 72.0
        offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
        for label in cb.yaxis.get_majorticklabels():
            label.set_transform(label.get_transform() + offset)

    def plot_distribution(self, percentage=False):
        h, e = np.histogram(self.ds.lcc.values, bins=self.ds.lcc.attrs["flag_values"])
        if percentage:
            values = (h / np.sum(h)) * 100.00
            label = f"Percentage Cover (%)"
        else:
            values = h
            label = f"Pixel Count"
        fig, ax = plt.subplots(figsize=(6, 12))
        ax.barh(
            range(len(self.legend.LCCOwnLabel)),
            values,
            edgecolor="k",
            tick_label=self.legend.LCCOwnLabel,
            color=self.legend.HEX_COL.values,
        )
        ax.set_xlabel(label)

    def bin_lst_by_lcc(self):
        indexes = np.digitize(self.ds.lcc.values, bins=self.ds.lcc.attrs["flag_values"])
        data = []
        for idx in np.unique(indexes):
            idx_data = np.where(indexes == idx, self.ds.lst.squeeze().values, np.nan)
            data.append(idx_data[~np.isnan(idx_data)])
        self.lcc_binned_lst = data

    def box_plot(self):
        fig, ax = plt.subplots(figsize=(6, 10))
        bplot = ax.boxplot(self.lcc_binned_lst, vert=False, patch_artist=True)
        ax.set_yticks(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        )
        ax.set_yticklabels(self.plotting_info["labels"][:-1])
        ax.set_xlabel("Land Surface Temperature (K)")
        for patch, color in zip(bplot["boxes"], self.plotting_info["colours"]):
            patch.set_facecolor(color)

    def volin_plot(self):
        fig, ax = plt.subplots(figsize=(6, 10))
        vplot = ax.violinplot(
            self.lcc_binned_lst, vert=False, showextrema=False, showmedians=True
        )
        ax.set_yticks(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        )
        ax.set_yticklabels(self.plotting_info["labels"][:-1])
        ax.set_xlabel("Land Surface Temperature (K)")
        for patch, color in zip(vplot["bodies"], self.plotting_info["colours"]):
            patch.set_facecolor(color)
            patch.set_edgecolor("grey")
            patch.set_alpha(1)
