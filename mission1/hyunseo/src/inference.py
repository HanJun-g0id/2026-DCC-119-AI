import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F

from audio_utils import (
    load_label_json,
    get_speaker_segments,
    load_audio,
    crop_speaker_audio,
    sliding_windows,
    TARGET_SPEAKER_CALLER,
)
from model import GenderClassifier

GENDER_LABEL_INV = {0: "M", 1: "F"}


def parse_args():
    p = argparse.ArgumentParser(description="Mission 1: 신고자 성별 분류 추론 스크립트")
    p.add_argument("-audio_dir", "--audio_dir", type=str, required=True)
    p.add_argument("-label_dir", "--label_dir", type=str, required=True)
    p.add_argument("-ckpt_path", "--ckpt_path", type=str, required=True)
    p.add_argument("-output", "--output", type=str, required=True)
    p.add_argument("--stride_sec", type=float, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] device = {device}")
    print(f"[INFO] 체크포인트 로드: {args.ckpt_path}")
    ckpt = torch.load(args.ckpt_path, map_location=device)
    config = ckpt["config"]

    sample_rate = config["sample_rate"]
    max_duration_sec = config["max_duration_sec"]
    time_unit = config.get("time_unit", "ms")
    target_len = int(max_duration_sec * sample_rate)

    stride_sec = args.stride_sec if args.stride_sec is not None else max_duration_sec / 2.0
    stride = max(1, int(stride_sec * sample_rate))

    model = GenderClassifier(
        pretrained_model_name=config["pretrained_model_name"],
        num_classes=config.get("num_classes", 2),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    audio_dir = Path(args.audio_dir)
    label_dir = Path(args.label_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    json_paths = sorted(label_dir.glob("*.json"))
    if not json_paths:
        raise RuntimeError(f"label_dir에 json 파일이 없습니다: {label_dir}")

    rows = []
    num_fallback = 0

    with torch.no_grad():
        for jp in json_paths:
            stem = jp.stem
            audio_file_name = f"{stem}.wav"
            audio_path = audio_dir / audio_file_name

            if not audio_path.exists():
                continue

            try:
                data = load_label_json(jp)
                segments = get_speaker_segments(data, target_speaker=TARGET_SPEAKER_CALLER)

                waveform = load_audio(audio_path, target_sample_rate=sample_rate)
                speaker_wave = crop_speaker_audio(
                    waveform, segments, sample_rate=sample_rate, time_unit=time_unit
                )
                windows = sliding_windows(speaker_wave, target_len=target_len, stride=stride)

                probs_sum = None
                for win_wave, win_mask in windows:
                    input_values = win_wave.unsqueeze(0).to(device)
                    attention_mask = win_mask.unsqueeze(0).to(device)
                    logits = model(input_values, attention_mask=attention_mask)
                    probs = F.softmax(logits, dim=-1)
                    probs_sum = probs if probs_sum is None else probs_sum + probs

                probs_avg = probs_sum / len(windows)
                pred_idx = int(probs_avg.argmax(dim=-1).item())
                pred_gender = GENDER_LABEL_INV[pred_idx]

            except Exception as e:
                print(f"[WARN] {stem} 처리 중 오류: {e}")
                pred_gender = "F"
                num_fallback += 1

            rows.append({"audio_file_name": audio_file_name, "gender": pred_gender})

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_file_name", "gender"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DONE] 총 {len(rows)}건 완료 -> {output_path}")


if __name__ == "__main__":
    main()
