import numpy as np
import os
import mne
import pywt
from scipy.signal import resample
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

def temp_feat(eeg_data, args):
    save_path = os.path.join('../data/temp_feat/', f"{args.subject}_{args.granularity}_wavelet.npy")
    if os.path.exists(save_path):
        return np.load(save_path)

    n_trials, n_channels, n_times = eeg_data.shape
    sampling_rate = 1000  # Hz

    # Frequencies of interest (e.g., EEG bands: delta to gamma)
    freq_bands = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 50)
    }

    # Prepare output array: [trials, channels * bands, time]
    n_bands = len(freq_bands)
    wavelet_features = []

    for trial in range(n_trials):
        trial_feat = []
        for ch in range(n_channels):
            signal = eeg_data[trial, ch, :]
            ch_feat = []

            for band, (fmin, fmax) in freq_bands.items():
                center_freq = (fmin + fmax) / 2
                scale = pywt.central_frequency('cmor1.5-1.0') * sampling_rate / center_freq
                coef, _ = pywt.cwt(signal, [scale], 'cmor1.5-1.0', sampling_period=1.0 / sampling_rate)
                power = np.abs(coef[0])  # Power from complex coefficients
                ch_feat.append(power)
            
            trial_feat.append(np.stack(ch_feat, axis=0))  # [bands, time]
        
        trial_feat = np.stack(trial_feat, axis=0)  # [channels, bands, time]
        trial_feat = trial_feat.reshape(n_channels * n_bands, n_times)  # Flatten channels*bands
        wavelet_features.append(trial_feat)
    
    wavelet_features = np.stack(wavelet_features, axis=0)  # [trials, chans*bands, time]
    np.save(save_path, wavelet_features)
    return wavelet_features
