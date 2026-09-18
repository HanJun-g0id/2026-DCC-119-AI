import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio

TARGET_SPEAKER_CALLER = 1
TARGET_SPEAKER_DISPATCHER = 0


def load_label_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_speaker_segments(label_data, target_speaker=TARGET_SPEAKER_CALLER):
    utterances = label_data.get("utterances", [])
    segments = []
    for utt in utterances:
        speaker = utt.get("speaker")
        start_at = utt.get("startAt")
        end_at = utt.get("endAt")
        if speaker != target_speaker:
            continue
        if start_at is None or end_at is None:
            continue
        if end_at <= start_at:
            continue
        segments.append((float(start_at), float(end_at)))
    segments.sort(key=lambda x: x[0])
    return segments


def load_audio(audio_path, target_sample_rate=16000):
    waveform, sr = torchaudio.load(str(audio_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)
        waveform = resampler(waveform)
    return waveform.squeeze(0)


def crop_speaker_audio(waveform, segments, sample_rate=16000, time_unit="ms"):
    total_len = waveform.shape[0]
    chunks = []
    for start, end in segments:
        if time_unit == "ms":
            start_sample = int(round(start / 1000.0 * sample_rate))
            end_sample = int(round(end / 1000.0 * sample_rate))
        else:
            start_sample = int(round(start * sample_rate))
            end_sample = int(round(end * sample_rate))
        start_sample = max(0, min(start_sample, total_len))
        end_sample = max(start_sample, min(end_sample, total_len))
        if end_sample > start_sample:
            chunks.append(waveform[start_sample:end_sample])
    if not chunks:
        return torch.zeros(1, dtype=waveform.dtype)
    return torch.cat(chunks, dim=0)


def fix_length(waveform, target_len, random_crop=False):
    length = waveform.shape[0]
    if length == target_len:
        attn = torch.ones(target_len, dtype=torch.long)
        return waveform, attn
    if length > target_len:
        if random_crop:
            start = random.randint(0, length - target_len)
        else:
            start = 0
        cropped = waveform[start:start + target_len]
        attn = torch.ones(target_len, dtype=torch.long)
        return cropped, attn
    pad_len = target_len - length
    padded = F.pad(waveform, (0, pad_len))
    attn = torch.cat([torch.ones(length, dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)])
    return padded, attn


def sliding_windows(waveform, target_len, stride):
    length = waveform.shape[0]
    if length <= target_len:
        win, attn = fix_length(waveform, target_len, random_crop=False)
        return [(win, attn)]

    windows = []
    start = 0
    while start < length:
        end = start + target_len
        chunk = waveform[start:end]
        if chunk.shape[0] < target_len:
            chunk, attn = fix_length(chunk, target_len, random_crop=False)
        else:
            attn = torch.ones(target_len, dtype=torch.long)
        windows.append((chunk, attn))
        if end >= length:
            break
        start += stride
    return windows
