import numpy as np
import os
import mne
from utilities import *


def de_feat_cal(eeg_data, args):
    if os.path.exists(os.path.join('../data/de_feat/', f"{args.subject}_{args.granularity}_de.npy")):
        return np.load(os.path.join('../data/de_feat/', f"{args.subject}_{args.granularity}_de.npy"))
    else:
        channel_names = [f'EEG{i}' for i in range(1, 63)]
        info = mne.create_info(ch_names=channel_names, sfreq=1000, ch_types='eeg')
        _epochs = mne.EpochsArray(data=eeg_data, info=info)

        de_feat_list = []
        for f_min, f_max in FREQ_BANDS.values():
            # Compute power spectral density (PSD) via averaged fast fourier transforms (output: (4000,62,T))
            spectrum = _epochs.compute_psd(fmin=f_min, fmax=f_max)
            # Add small delta to ensure there is no log(0)
            psd = spectrum.get_data() + 1e-10
            # Compute the differential entropy for each frequency band, summed over the timesteps (output: (4000,62))
            diff_entropy = np.sum(np.log(psd), axis=-1)
            de_feat_list.append(diff_entropy)
        # Concatenate the differential entropy features for all 5frequency bands (output: (4000,310))
        _de_feat = np.concatenate(de_feat_list, axis=1)
        # print(_de_feat.shape)  # de_feat.shape = (4000, 310), normally
        np.save(os.path.join('../data/de_feat/', f"{args.subject}_{args.granularity}_de.npy"), _de_feat)
        return _de_feat

def de_feat_temp(eeg_data, args):
    save_path = os.path.join('../data/de_feat_temp/', f"{args.subject}_{args.granularity}_de.npy")
    
    if os.path.exists(save_path):
        return np.load(save_path)
    
    # Define EEG channel names
    channel_names = [f'EEG{i}' for i in range(1, 63)]
    info = mne.create_info(ch_names=channel_names, sfreq=1000, ch_types='eeg')
    eeg_data = eeg_data[:, :, 40:440]
    _epochs = mne.EpochsArray(data=eeg_data, info=info)

    de_feat_list = []
    
    for f_min, f_max in FREQ_BANDS.values():
        # Compute power spectral density (PSD) for the given frequency band
        spectrum = _epochs.compute_psd(fmin=f_min, fmax=f_max)
        psd = spectrum.get_data() + 1e-10  # Avoid log(0)

        # Compute differential entropy (log of PSD without summing over time)
        diff_entropy = np.log(psd)  # Shape: (trials, channels, time_steps)

        de_feat_list.append(diff_entropy)
    
    # Stack features across frequency bands, maintaining the time dimension
    _de_feat = np.concatenate(de_feat_list, axis=1)  # Shape: (trials, channels * bands, time_steps)

    # Save and return the processed features
    np.save(save_path, _de_feat)
    return _de_feat
