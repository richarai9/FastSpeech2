import argparse
import os

import librosa
import numpy as np
# from sklearn.metrics import mean_absolute_error
from scipy.interpolate import interp1d
from scipy.signal import welch
import tgt
import torch
from tqdm import tqdm

def get_alignment(tier, sampling_rate, hop_length):
    sil_phones = ["sil", "sp", "spn"]

    phones = []
    durations = []
    start_time = 0
    end_time = 0
    end_idx = 0
    for t in tier._objects:
        s, e, p = t.start_time, t.end_time, t.text

        # Trim leading silences
        if phones == []:
            if p in sil_phones:
                continue
            else:
                start_time = s

        if p not in sil_phones:
            # For ordinary phones
            phones.append(p)
            end_time = e
            end_idx = len(phones)
        else:
            # For silent phones
            phones.append(p)

        durations.append(
            int(
                np.round(e * sampling_rate / hop_length)
                - np.round(s * sampling_rate / hop_length)
            )
        )

    # Trim tailing silences
    phones = phones[:end_idx]
    durations = durations[:end_idx]

    return phones, durations, start_time, end_time


def get_spectral_tilt_for(window) -> float:
    frequencies, powers = welch(window, fs=22050)
    log_powers = np.log10(powers)
    return np.polyfit(frequencies, log_powers, 1)[0]

def get_spectral_tilt(wav):
    window_length = 1024

    padded_wav = librosa.util.pad_center(wav, 1024+len(wav))
    windowed_data = np.lib.stride_tricks.sliding_window_view(padded_wav, window_length)[::256]
    spectral_tilt = [get_spectral_tilt_for(window) for window in windowed_data]
    spectral_tilt = np.array(spectral_tilt)

    return spectral_tilt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generates MAE loss figures for spectral tilt of synthesized WAVs vs. ground truth WAVs. You must have already generated the synthesized waveforms you would like to find the MAE loss for. These must have the same name format as the original dataset.")
    parser.add_argument('-e', '--eval_path', type=str, default='./output/result/LJSpeech/test')
    parser.add_argument('-t', '--tilt_path', type=str, default='./preprocessed_data/LJSpeech/tilt/')
    parser.add_argument('-g', '--text_grid_path', type=str, default='./preprocessed_data/LJSpeech/TextGrid/LJSpeech/')
    parser.add_argument('--sampling_rate', type=int, default=22050)
    parser.add_argument('--hop_length', type=int, default=256)
    args = parser.parse_args()

    # Load ground truth spectral tilts
    grounds_truths = {}
    synthesized = {}
    for filename in tqdm(os.listdir(args.eval_path)):
        ljspeech_name = filename.split('.')[0]
        grounds_truths[ljspeech_name] = np.load(os.path.join(args.tilt_path, 'LJSpeech-spec-{}.npy'.format(ljspeech_name)))

        # Calculate spectral tilt for synthesized WAV files
        wav, _ = librosa.load(os.path.join(args.eval_path, filename))
        textgrid = tgt.io.read_textgrid(os.path.join(args.text_grid_path, '{}.TextGrid'.format(ljspeech_name)))
        phone, duration, start, end = get_alignment(
            textgrid.get_tier_by_name("phones"),
            args.sampling_rate, args.hop_length
        )

        # Tilts
        tilt = get_spectral_tilt(wav)
        pos = 0
        for i, d in enumerate(duration):
            if d > 0:
                tilt[i] = np.mean(tilt[pos : pos + d])
            else:
                tilt[i] = 0
            pos += d
        tilt = tilt[: len(duration)]

        synthesized[ljspeech_name] = tilt.T

    # Find MAE between ground truth and synthesized
    mae_losses = []
    loss = torch.nn.L1Loss()
    for name, ground_truth in grounds_truths.items():
        mae = loss(torch.from_numpy(ground_truth), torch.from_numpy(synthesized[name]))
        if not torch.isnan(mae):
            mae_losses.append(mae.item())

    print('\nFINAL AVERAGE MAE: {}'.format(sum(mae_losses) / len(mae_losses)))