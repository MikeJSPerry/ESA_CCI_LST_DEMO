import numpy as np
import matplotlib.pyplot as plt


def set_up(ds):
    classes = ds.lcc.attrs["flag_meanings"].split(" ")
    bins = ds.lcc.attrs["flag_values"]
    fig = plt.figure(figsize=(12, 4))
    h, e = np.histogram(ds.lcc.values, bins=bins)
    fig = plt.figure(figsize=(12, 4))
    plt.bar(range(len(bins) - 1), h, width=1, edgecolor="k", tick_label=classes[:-1])
    plt.xticks(rotation=90)


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
