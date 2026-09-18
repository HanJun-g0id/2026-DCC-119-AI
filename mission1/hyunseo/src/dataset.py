from pathlib import Path

import torch
from torch.utils.data import Dataset

from audio_utils import (
    load_label_json,
    get_speaker_segments,
    load_audio,
    crop_speaker_audio,
    fix_length,
    TARGET_SPEAKER_CALLER,
)

GENDER_LABEL_MAP = {"M": 0, "F": 1}
GENDER_LABEL_INV = {0: "M", 1: "F"}


class SpeakerGenderDataset(Dataset):
    def __init__(
        self,
        audio_dir,
        label_dir,
        sample_rate=16000,
        max_duration_sec=6.0,
        region_filter="서울",
        time_unit="ms",
        mode="train",
    ):
        self.audio_dir = Path(audio_dir)
        self.label_dir = Path(label_dir)
        self.sample_rate = sample_rate
        self.max_duration_sec = max_duration_sec
        self.target_len = int(max_duration_sec * sample_rate)
        self.time_unit = time_unit
        self.mode = mode
        self.samples = []
        self._build_index(region_filter)

    def _build_index(self, region_filter):
        json_paths = sorted(self.label_dir.glob("*.json"))
        skipped_region, skipped_no_audio, skipped_no_label, skipped_no_seg = 0, 0, 0, 0

        for jp in json_paths:
            try:
                data = load_label_json(jp)
            except Exception:
                continue

            address = data.get("address", "") or ""
            if region_filter and region_filter not in address:
                skipped_region += 1
                continue

            gender_raw = data.get("gender")
            if gender_raw not in GENDER_LABEL_MAP:
                skipped_no_label += 1
                continue

            stem = jp.stem
            audio_path = self.audio_dir / f"{stem}.wav"
            if not audio_path.exists():
                skipped_no_audio += 1
                continue

            segments = get_speaker_segments(data, target_speaker=TARGET_SPEAKER_CALLER)
            if not segments:
                skipped_no_seg += 1
                continue

            self.samples.append(
                {
                    "stem": stem,
                    "audio_path": audio_path,
                    "segments": segments,
                    "label": GENDER_LABEL_MAP[gender_raw],
                }
            )

        print(
            f"[SpeakerGenderDataset:{self.mode}] 총 json {len(json_paths)}개 중 "
            f"채택 {len(self.samples)}개 "
            f"(지역필터제외 {skipped_region}, gender라벨없음 {skipped_no_label}, "
            f"wav없음 {skipped_no_audio}, 화자구간없음 {skipped_no_seg})"
        )

        if len(self.samples) == 0:
            raise RuntimeError(
                "조건을 만족하는 학습 샘플이 없습니다. audio_dir / label_dir 경로와 "
                "region_filter 설정을 확인하세요."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        waveform = load_audio(item["audio_path"], target_sample_rate=self.sample_rate)
        speaker_wave = crop_speaker_audio(
            waveform, item["segments"], sample_rate=self.sample_rate, time_unit=self.time_unit
        )
        random_crop = self.mode == "train"
        fixed_wave, attn_mask = fix_length(speaker_wave, self.target_len, random_crop=random_crop)
        return {
            "input_values": fixed_wave,
            "attention_mask": attn_mask,
            "label": torch.tensor(item["label"], dtype=torch.long),
            "stem": item["stem"],
        }


def collate_fn(batch):
    input_values = torch.stack([b["input_values"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    stems = [b["stem"] for b in batch]
    return {
        "input_values": input_values,
        "attention_mask": attention_mask,
        "label": labels,
        "stem": stems,
    }
