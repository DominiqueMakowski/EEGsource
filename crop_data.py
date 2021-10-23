# https://mne.tools/dev/auto_tutorials/forward/35_eeg_no_mri.html

import os

import mne
import numpy as np
import TruScanEEGpy

# Preprocess raw EEG =========================================================
raw = mne.io.read_raw_edf("participant.edf", preload=True)
eog = nk.mne_channel_extract(raw, ["124", "125"])
eog = eog["124"] - eog["125"]
raw = nk.mne_channel_add(raw, eog, channel_type="eog", channel_name="EOG")
raw.drop_channels(["124", "125"])

# Montage
mne.rename_channels(
    raw.info,
    dict(zip(raw.info["ch_names"], TruScanEEGpy.convert_to_tenfive(raw.info["ch_names"]))),
)
montage = TruScanEEGpy.montage_mne_128(TruScanEEGpy.layout_128(names="10-5"))
extra_channels = np.array(raw.info["ch_names"])[
    np.array([i not in montage.ch_names for i in raw.info["ch_names"]])
]
raw = raw.drop_channels(extra_channels[np.array([i not in ["EOG", "ECG"] for i in extra_channels])])
raw = raw.set_montage(montage)

# Reference
raw = raw.resample(sfreq=500)
raw.set_eeg_reference(ref_channels="average", projection=True)  # needed for inverse modeling

raw.crop(60, 120).save("data_short.fif")
