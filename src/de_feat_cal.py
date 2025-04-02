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
    
    # Convert EEG data into MNE epochs
    _epochs = mne.EpochsArray(data=eeg_data, info=info)

    de_feat_list = []
    
    # PSD Parameters
    n_fft = 256  # Window size in samples
    n_per_seg = n_fft  # Use same size for segment
    n_overlap = n_per_seg // 2  # 50% overlap
    n_freqs = 20  # Target number of frequency points
    
    for f_min, f_max in FREQ_BANDS.values():
        # Compute PSD over time windows
        spectrum = _epochs.compute_psd(
            fmin=f_min,
            fmax=f_max,
            method='welch',
            n_fft=n_fft,
            n_per_seg=n_per_seg,
            n_overlap=n_overlap,
            average='mean',
            window='hann'
        )
        psd = spectrum.get_data() + 1e-10  # Avoid log(0)

        # Get frequency points for this band
        freqs = spectrum.freqs
        
        # Interpolate to fixed number of frequency points
        from scipy.interpolate import interp1d
        x_old = np.linspace(0, 1, len(freqs))
        x_new = np.linspace(0, 1, n_freqs)
        
        # Initialize array for interpolated data
        interpolated = np.zeros((psd.shape[0], psd.shape[1], n_freqs))
        
        # Interpolate each trial and channel
        for i in range(psd.shape[0]):
            for j in range(psd.shape[1]):
                f = interp1d(x_old, psd[i, j], kind='linear')
                interpolated[i, j] = f(x_new)
        
        # Compute differential entropy
        diff_entropy = np.log(interpolated)
        de_feat_list.append(diff_entropy)
    
    # Stack features across frequency bands
    _de_feat = np.concatenate(de_feat_list, axis=1)  # Shape: (trials, channels * bands, n_freqs)

    # Save and return the processed features
    np.save(save_path, _de_feat)
    return _de_feat
