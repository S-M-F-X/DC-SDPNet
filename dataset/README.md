# Dataset

The preprocessed dataset used by DC-SDPNet is distributed as a GitHub Release asset.

Download it from:

[`DC-SDPNet v1.0.0`](https://github.com/S-M-F-X/DC-SDPNet/releases/tag/v1.0.0)

Required file:

```text
pv_power_1,7_3d.npy
```

After downloading, place the file in this directory so that the project structure becomes:

```text
dataset/
├─ README.md
└─ pv_power_1,7_3d.npy
```

The expected dataset shape is:

```text
[time, feature, station]
```

with 227 stations and 8 features.

The default loader expects the dataset at:

```text
dataset/pv_power_1,7_3d.npy
```

Project paths are resolved relative to `main.py`, so no additional path modification is required when the file is placed at the location above.
