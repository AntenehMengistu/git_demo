import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Input
# ============================================================
file = "flux_CH4_AGW.nc"
ds = xr.open_dataset(file)

x = ds["x"].values
xb = ds["xb"].values

# ============================================================
# Grid dimensions
# ============================================================
lat_global_len, lon_global_len = 90, 180
lat_nest_len, lon_nest_len = 76, 90

num_nest_points = lat_nest_len * lon_nest_len

# ============================================================
# Select physical time indices
# ============================================================
index = np.arange(ds.sizes["time_phys"])

# ============================================================
# Extract nested prior and posterior
# ============================================================
nested_prior = (
    xb[:, :num_nest_points]
    .reshape(-1, lat_nest_len, lon_nest_len)[index]
)

nested_post = (
    x[:, :num_nest_points]
    .reshape(-1, lat_nest_len, lon_nest_len)[index]
)

# ============================================================
# Mean prior/posterior over selected MC members
# ============================================================
nested_prior_mean = nested_prior.mean(axis=0)
nested_post_mean = nested_post.mean(axis=0)

# ============================================================
# If you want uncertainty reduction from b_std / pa_std
# ============================================================
b = ds["b_std_phys"].values
pa = ds["pa_std_phys"].values

nested_b = (
    b[:, :num_nest_points]
    .reshape(-1, lat_nest_len, lon_nest_len)[index]
)

nested_pa = (
    pa[:, :num_nest_points]
    .reshape(-1, lat_nest_len, lon_nest_len)[index]
)

ur = 100 * (1 - nested_pa / nested_b)

ur = np.where(
    np.isfinite(ur) & (nested_b > 0),
    ur,
    np.nan
)

# ============================================================
# Statistics
# ============================================================
mean_ur = np.nanmean(ur, axis=(1, 2))
min_ur = np.nanmin(ur, axis=(1, 2))
max_ur = np.nanmax(ur, axis=(1, 2))

negative_fraction = np.nanmean(ur < 0, axis=(1, 2)) * 100

for i, t in enumerate(ds["time_phys"].values):
    print(
        f"{str(t)[:10]}  "
        f"mean={mean_ur[i]:6.2f}%  "
        f"min={min_ur[i]:7.2f}%  "
        f"max={max_ur[i]:7.2f}%  "
        f"negative={negative_fraction[i]:6.2f}%"
    )

# ============================================================
# Nested coordinates
# ============================================================
lat = ds["lat"].values[:num_nest_points].reshape(
    lat_nest_len, lon_nest_len
)

lon = ds["lon"].values[:num_nest_points].reshape(
    lat_nest_len, lon_nest_len
)

# ============================================================
# Plot mean uncertainty reduction over all physical months
# ============================================================
ur_mean = np.nanmean(ur, axis=0)

plt.figure(figsize=(10, 7))

pcm = plt.pcolormesh(
    lon,
    lat,
    ur_mean,
    cmap="RdBu_r",
    vmin=-50,
    vmax=100,
    shading="auto"
)

plt.colorbar(pcm, label="Uncertainty reduction (%)")

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("CH₄ AGW Mean Uncertainty Reduction")

plt.tight_layout()
plt.show()
