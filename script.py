# https://mne.tools/dev/auto_tutorials/forward/35_eeg_no_mri.html

import os

import mne
import numpy as np
import TruScanEEGpy


def eeg_templateMRI():
    """
    Get template MRI.
    See https://mne.tools/stable/auto_tutorials/forward/35_eeg_no_mri.html
    """

    # Try loading pooch (needed by mne)
    try:
        import pooch
    except ImportError as e:
        raise ImportError(
            "The 'pooch' module is required for this function to run. ",
            "Please install it first (`pip install pooch`).",
        ) from e

    # Try loading mne
    try:
        import mne
    except ImportError as e:
        raise ImportError(
            "NeuroKit error: eeg_channel_add(): the 'mne' module is required for this function to run. ",
            "Please install it first (`pip install mne`).",
        ) from e

    # Download fsaverage files
    fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
    subjects_dir = os.path.dirname(fs_dir)

    # The files live in:
    subject = "fsaverage"
    trans = "fsaverage"  # MNE has a built-in fsaverage transformation
    src = os.path.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")
    bem = os.path.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")
    return src, bem


# Preprocess raw EEG =========================================================
raw = mne.io.read_raw_fif("data_short.fif", preload=True)


# =============================================================================
src, bem = eeg_templateMRI()

# requires PySide2, ipyvtklink
# Check that the locations of EEG electrodes is correct with respect to MRI
mne.viz.plot_alignment(
    raw.info,
    src=src,
    eeg=["original", "projected"],
    trans="fsaverage",  # MNE has a built-in fsaverage transformation
    # show_axes=True,
    mri_fiducials=True,
    dig="fiducials",
)

# Setup source space and compute forward
fwd = mne.make_forward_solution(
    raw.info,
    trans="fsaverage",
    src=src,
    bem=bem,
    eeg=True,
    mindist=5.0,
    n_jobs=1,
)
# print(fwd)

noise_cov = mne.compute_raw_covariance(raw, tmin=0, tmax=None)

inverse_operator = mne.minimum_norm.make_inverse_operator(
    raw.info, fwd, noise_cov, loose=0.2, depth=0.8
)

snr = 1.0  # use smaller SNR for raw data
lambda2 = 1.0 / snr ** 2
start, stop = raw.time_as_index([0, 15])  # read the first 15s of data
# Compute inverse solution
stc = mne.minimum_norm.apply_inverse_raw(
    raw,
    inverse_operator,
    lambda2,
    method="sLORETA",  # sLORETA method (could also be MNE or dSPM)
    start=start,
    stop=stop,
    pick_ori=None,
)

# Inflated
stc.plot()
# Flat
brain = stc.plot(surface="flat", hemi="both")
brain.add_annotation("HCPMMP1_combined", borders=2)
# Normal
brain = stc.plot(surface="white")
brain.add_annotation("PALS_B12_Lobes", borders=2)
brain.add_annotation("HCPMMP1_combined", borders=2)


# # Specific region
# src = inverse_operator["src"]
# fname_aseg = os.path.join(
#     mne.datasets.sample.data_path(), "subjects/sample/mri/aparc.a2009s+aseg.mgz"
# )
# label_names = mne.get_volume_labels_from_aseg(fname_aseg)
# label_tc = stc.extract_label_time_course(fname_aseg, src=src)

# volume_label = 'Left-Amygdala'
# sphere = (0, 0, 0, 0.12)
# lh_cereb = setup_volume_source_space(
#     subject, mri=aseg_fname, sphere=sphere, volume_label=volume_label,
#     subjects_dir=subjects_dir, sphere_units='m')


# label_names = mne.get_volume_labels_from_aseg(fname_aseg)
# label_tc = stc.extract_label_time_course(fname_aseg, src=src)
# stc.extract_label_time_course('Left-Amygdala', src=src)

# stc_back = mne.labels_to_stc(fname_aseg, label_tc, src=src)
# stc_back.plot(src, subjects_dir=subjects_dir, mode="glass_brain")


# mne.extract_label_time_course(stc, "Left-Hippocampus", src)
