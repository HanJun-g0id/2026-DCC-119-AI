from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


# ============================================================
# DCC 119 - Mission 1~3 dataset builder
#
# This script is designed for the ACTUAL extracted Windows
# Training data structure:
#
# Training/
#   1.원천데이터/
#       TS_서울_화재/
#       TS_서울_기타/
#       TS_서울_구조/
#       TS_서울_구급/
#
#   2.라벨링데이터/
#       TL_서울_화재/
#       TL_서울_기타/
#       TL_서울_구조/
#       TL_서울_구급/
#
# IMPORTANT:
# - Only Training data are processed here.
# - Official Validation folders are NOT used for training.
# - Split is done at CONVERSATION/JSON level, not utterance level.
# - Mission 2 does NOT export transcript text.
# - Mission 3 uses transcript text only as model input.
# - Audio is NOT physically cut into millions of WAV files.
# ============================================================


# ------------------------------------------------------------
# 1. Default paths
# ------------------------------------------------------------

DEFAULT_BASE_DIR = Path(
    r"C:\Users\hanju\Downloads\088.위급상황 음성-음향_고도화_119 지능형 신고접수 음성 인식 데이터"
) / "3.개방데이터" / "1.데이터"

TARGET_SYMPTOMS = [
    "고열",
    "구토",
    "두통",
    "복통",
    "어지러움",
    "열상",
    "오심",
    "전신쇠약",
    "호흡곤란",
]

# Conversation-level split ratio.
DEFAULT_VALID_RATIO = 0.10

# Deterministic salt. Keep unchanged once experiments start.
SPLIT_SALT = "dcc119-v1"


# ------------------------------------------------------------
# 2. Generic helpers
# ------------------------------------------------------------

def load_json(path: Path) -> Any:
    """Read UTF-8 JSON, including BOM-containing files."""
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def expand_records(data: Any) -> list[dict[str, Any]]:
    """
    Usually one JSON = one conversation.
    Also handles a top-level list defensively.
    """
    if isinstance(data, dict):
        return [data]

    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]

    return []


def normalize_gender(value: Any) -> str | None:
    if value is None:
        return None

    v = str(value).strip().upper()

    mapping = {
        "M": "M",
        "MALE": "M",
        "남": "M",
        "남성": "M",
        "F": "F",
        "FEMALE": "F",
        "여": "F",
        "여성": "F",
    }

    return mapping.get(v)


def normalize_speaker(value: Any) -> int | None:
    try:
        v = int(value)
    except (TypeError, ValueError):
        return None

    return v if v in (0, 1) else None


def normalize_symptoms(value: Any) -> list[str]:
    """
    Keep only the 9 competition target symptoms.
    Preserve the competition-defined order.
    """
    if value is None:
        return []

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []

        try:
            parsed = json.loads(value)
            value = parsed if isinstance(parsed, list) else [value]
        except json.JSONDecodeError:
            value = [value]

    if not isinstance(value, list):
        return []

    result = []
    seen = set()

    for item in value:
        s = str(item).strip()
        if s in TARGET_SYMPTOMS and s not in seen:
            seen.add(s)
            result.append(s)

    return sorted(result, key=TARGET_SYMPTOMS.index)


def safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def get_conversation_id(data: dict[str, Any], json_path: Path) -> str:
    """
    Prefer dataset _id.
    Fall back to recordId, then file stem.
    """
    for key in ("_id", "recordId"):
        value = data.get(key)
        if value is not None and str(value).strip():
            return str(value)

    return json_path.stem


# ------------------------------------------------------------
# 3. Deterministic conversation-level split
# ------------------------------------------------------------

def assign_split(conversation_id: str, valid_ratio: float) -> str:
    """
    Deterministically assign a conversation to train/valid.

    The same conversation will ALWAYS go to the same split,
    regardless of file traversal order.
    """
    raw = f"{SPLIT_SALT}:{conversation_id}".encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()

    # Convert first 8 hex chars to [0, 1)
    value = int(digest[:8], 16) / 0xFFFFFFFF

    return "valid" if value < valid_ratio else "train"


# ------------------------------------------------------------
# 4. WAV matching
# ------------------------------------------------------------

def build_audio_index(audio_root: Path):
    """
    The user's actual data was verified to use matching names:
      651e..._20220101.wav
      651e..._20220101.json

    So exact filename/stem matching is primary.
    """
    by_name: dict[str, Path] = {}
    by_stem: dict[str, Path] = {}

    wav_files = sorted(audio_root.rglob("*.wav"))

    for wav in wav_files:
        by_name[wav.name.lower()] = wav
        by_stem[wav.stem.lower()] = wav

    return by_name, by_stem, wav_files


def resolve_audio(
    data: dict[str, Any],
    json_path: Path,
    by_name: dict[str, Path],
    by_stem: dict[str, Path],
) -> Path | None:
    """
    Match JSON to WAV.

    Primary path:
        JSON filename stem == WAV filename stem

    Fallback:
        audioPath basename
        _id
        recordId
    """

    # 1) Most reliable for the verified dataset:
    stem = json_path.stem.lower()

    if stem in by_stem:
        return by_stem[stem]

    # 2) audioPath basename
    audio_path = data.get("audioPath")
    if audio_path:
        name = Path(str(audio_path).replace("\\", "/")).name.lower()

        if name in by_name:
            return by_name[name]

        stem2 = Path(name).stem.lower()
        if stem2 in by_stem:
            return by_stem[stem2]

    # 3) _id
    json_id = data.get("_id")
    if json_id:
        target = str(json_id).lower()
        candidates = [
            p for s, p in by_stem.items()
            if s.startswith(target + "_")
        ]
        if len(candidates) == 1:
            return candidates[0]

    # 4) recordId
    record_id = data.get("recordId")
    if record_id:
        target = str(record_id).lower()
        candidates = [
            p for s, p in by_stem.items()
            if s.startswith(target)
        ]
        if len(candidates) == 1:
            return candidates[0]

    return None


# ------------------------------------------------------------
# 5. Transcript construction
# ------------------------------------------------------------

def build_transcript(utterances: list[Any]) -> str:
    """
    Concatenate utterance text in chronological order.

    No speaker/gender/disaster metadata are injected.
    """
    valid = []

    for index, utt in enumerate(utterances):
        if not isinstance(utt, dict):
            continue

        text = utt.get("text")
        if text is None:
            continue

        text = str(text).strip()
        if not text:
            continue

        start = safe_int(utt.get("startAt"))
        # If startAt is absent, retain original order via index.
        sort_key = (start if start is not None else 10**18, index)

        valid.append((sort_key, text))

    valid.sort(key=lambda x: x[0])

    return " ".join(text for _, text in valid)


# ------------------------------------------------------------
# 6. Main builder
# ------------------------------------------------------------

def build_dataset(
    base_dir: Path,
    output_dir: Path,
    valid_ratio: float = DEFAULT_VALID_RATIO,
):
    training_root = base_dir / "Training"
    audio_root = training_root / "1.원천데이터"
    label_root = training_root / "2.라벨링데이터"

    if not training_root.exists():
        raise FileNotFoundError(f"Training 폴더가 없습니다: {training_root}")

    if not audio_root.exists():
        raise FileNotFoundError(f"원천데이터 폴더가 없습니다: {audio_root}")

    if not label_root.exists():
        raise FileNotFoundError(f"라벨링데이터 폴더가 없습니다: {label_root}")

    # Expected subfolders. Missing ones are skipped, not silently replaced.
    categories = ["화재", "기타", "구조", "구급"]

    for category in categories:
        a = audio_root / f"TS_서울_{category}"
        l = label_root / f"TL_서울_{category}"

        if not a.exists():
            raise FileNotFoundError(f"WAV 폴더가 없습니다: {a}")

        if not l.exists():
            raise FileNotFoundError(f"JSON 폴더가 없습니다: {l}")

    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Output files
    m1_train = output_dir / "mission1_train.csv"
    m1_valid = output_dir / "mission1_valid.csv"

    m2_train = output_dir / "mission2_train.csv"
    m2_valid = output_dir / "mission2_valid.csv"

    m3_train = output_dir / "mission3_train.csv"
    m3_valid = output_dir / "mission3_valid.csv"

    unmatched_path = reports_dir / "unmatched_audio.csv"
    invalid_path = reports_dir / "invalid_json.csv"

    # Counters
    stats = Counter()
    gender_counts = Counter()
    speaker_counts = Counter()
    symptom_counts_train = Counter()
    symptom_counts_valid = Counter()

    # Conversation IDs written to each split, useful for leakage checks.
    split_ids = {
        "train": set(),
        "valid": set(),
    }

    # Prepare audio indexes category by category.
    audio_indexes: dict[str, tuple[dict[str, Path], dict[str, Path], list[Path]]] = {}

    print("=" * 72)
    print("DCC 119 Dataset Builder")
    print("=" * 72)
    print(f"Base directory : {base_dir}")
    print(f"Output         : {output_dir}")
    print(f"Validation ratio (internal split): {valid_ratio:.2%}")
    print()

    for category in categories:
        category_audio_root = audio_root / f"TS_서울_{category}"
        print(f"[INDEX] WAV: 서울_{category}")

        by_name, by_stem, wav_files = build_audio_index(category_audio_root)
        audio_indexes[category] = (by_name, by_stem, wav_files)

        print(f"        WAV count: {len(wav_files):,}")

    # CSV headers
    m1_header = [
        "conversation_id",
        "category",
        "audio_path",
        "audio_file_name",
        "startAt",
        "endAt",
        "gender",
        "json_file",
        "utterance_index",
    ]

    m2_header = [
        "conversation_id",
        "category",
        "audio_path",
        "audio_file_name",
        "startAt",
        "endAt",
        "speaker",
        "json_file",
        "utterance_index",
    ]

    m3_header = [
        "conversation_id",
        "category",
        "json_file",
        "transcript",
        "symptom",
        *TARGET_SYMPTOMS,
    ]

    unmatched_fields = [
        "json_file",
        "conversation_id",
        "category",
        "_id",
        "record_id",
        "audioPath",
    ]

    invalid_fields = [
        "json_file",
        "error",
    ]

    # Open all writers once. This avoids keeping millions of rows in RAM.
    files_and_writers = {}

    try:
        for path, header in [
            (m1_train, m1_header),
            (m1_valid, m1_header),
            (m2_train, m2_header),
            (m2_valid, m2_header),
            (m3_train, m3_header),
            (m3_valid, m3_header),
            (unmatched_path, unmatched_fields),
            (invalid_path, invalid_fields),
        ]:
            f = path.open("w", encoding="utf-8-sig", newline="")
            writer = csv.DictWriter(
                f,
                fieldnames=header,
                extrasaction="ignore",
            )
            writer.writeheader()
            files_and_writers[path] = (f, writer)

        m1w_train = files_and_writers[m1_train][1]
        m1w_valid = files_and_writers[m1_valid][1]

        m2w_train = files_and_writers[m2_train][1]
        m2w_valid = files_and_writers[m2_valid][1]

        m3w_train = files_and_writers[m3_train][1]
        m3w_valid = files_and_writers[m3_valid][1]

        unmatched_writer = files_and_writers[unmatched_path][1]
        invalid_writer = files_and_writers[invalid_path][1]

        # Process each disaster category.
        for category in categories:
            label_category_root = label_root / f"TL_서울_{category}"

            json_files = sorted(label_category_root.rglob("*.json"))

            print()
            print(f"[PROCESS] 서울_{category}")
            print(f"          JSON count: {len(json_files):,}")

            for idx, json_path in enumerate(json_files, start=1):

                if idx == 1 or idx % 1000 == 0 or idx == len(json_files):
                    print(
                        f"\r          {idx:,}/{len(json_files):,}",
                        end="",
                        flush=True,
                    )

                stats["json_files_seen"] += 1

                try:
                    raw = load_json(json_path)
                    records = expand_records(raw)

                    if not records:
                        raise ValueError("JSON record가 dict/list 형태가 아닙니다.")

                except Exception as exc:
                    invalid_writer.writerow({
                        "json_file": str(json_path),
                        "error": repr(exc),
                    })
                    stats["invalid_json"] += 1
                    continue

                for data in records:
                    conversation_id = get_conversation_id(data, json_path)

                    split = assign_split(
                        conversation_id,
                        valid_ratio,
                    )

                    split_ids[split].add(conversation_id)

                    # ----------------------------------------
                    # WAV match
                    # ----------------------------------------

                    by_name, by_stem, _ = audio_indexes[category]

                    audio_path = resolve_audio(
                        data,
                        json_path,
                        by_name,
                        by_stem,
                    )

                    if audio_path is None:
                        unmatched_writer.writerow({
                            "json_file": str(json_path),
                            "conversation_id": conversation_id,
                            "category": category,
                            "_id": data.get("_id"),
                            "record_id": data.get("recordId"),
                            "audioPath": data.get("audioPath"),
                        })

                        stats["unmatched_audio"] += 1
                        continue

                    stats["conversations_processed"] += 1

                    # Save relative path from dataset root, not absolute Windows path.
                    try:
                        relative_audio = audio_path.relative_to(base_dir)
                    except ValueError:
                        relative_audio = audio_path

                    audio_rel = str(relative_audio)
                    audio_name = audio_path.name

                    utterances = data.get("utterances", [])
                    if not isinstance(utterances, list):
                        utterances = []

                    gender = normalize_gender(data.get("gender"))

                    # ----------------------------------------
                    # Mission 1
                    # ----------------------------------------

                    if gender in ("M", "F"):
                        for utterance_index, utt in enumerate(utterances):
                            if not isinstance(utt, dict):
                                continue

                            speaker = normalize_speaker(
                                utt.get("speaker")
                            )

                            # 신고자 only
                            if speaker != 1:
                                continue

                            start_at = safe_int(utt.get("startAt"))
                            end_at = safe_int(utt.get("endAt"))

                            if (
                                start_at is None
                                or end_at is None
                                or end_at <= start_at
                            ):
                                continue

                            row = {
                                "conversation_id": conversation_id,
                                "category": category,
                                "audio_path": audio_rel,
                                "audio_file_name": audio_name,
                                "startAt": start_at,
                                "endAt": end_at,
                                "gender": gender,
                                "json_file": str(
                                    json_path.relative_to(base_dir)
                                ),
                                "utterance_index": utterance_index,
                            }

                            if split == "train":
                                m1w_train.writerow(row)
                            else:
                                m1w_valid.writerow(row)

                            gender_counts[(split, gender)] += 1
                            stats[f"mission1_{split}"] += 1

                    # ----------------------------------------
                    # Mission 2
                    # ----------------------------------------

                    for utterance_index, utt in enumerate(utterances):
                        if not isinstance(utt, dict):
                            continue

                        speaker = normalize_speaker(
                            utt.get("speaker")
                        )

                        if speaker is None:
                            continue

                        start_at = safe_int(utt.get("startAt"))
                        end_at = safe_int(utt.get("endAt"))

                        if (
                            start_at is None
                            or end_at is None
                            or end_at <= start_at
                        ):
                            continue

                        # NOTE: No text column here.
                        row = {
                            "conversation_id": conversation_id,
                            "category": category,
                            "audio_path": audio_rel,
                            "audio_file_name": audio_name,
                            "startAt": start_at,
                            "endAt": end_at,
                            "speaker": speaker,
                            "json_file": str(
                                json_path.relative_to(base_dir)
                            ),
                            "utterance_index": utterance_index,
                        }

                        if split == "train":
                            m2w_train.writerow(row)
                        else:
                            m2w_valid.writerow(row)

                        speaker_counts[(split, speaker)] += 1
                        stats[f"mission2_{split}"] += 1

                    # ----------------------------------------
                    # Mission 3
                    # ----------------------------------------

                    transcript = build_transcript(utterances)

                    if transcript:
                        symptoms = normalize_symptoms(
                            data.get("symptom")
                        )

                        binary = {
                            symptom: int(symptom in symptoms)
                            for symptom in TARGET_SYMPTOMS
                        }

                        row = {
                            "conversation_id": conversation_id,
                            "category": category,
                            "json_file": str(
                                json_path.relative_to(base_dir)
                            ),
                            "transcript": transcript,
                            "symptom": json.dumps(
                                symptoms,
                                ensure_ascii=False,
                            ),
                            **binary,
                        }

                        if split == "train":
                            m3w_train.writerow(row)
                        else:
                            m3w_valid.writerow(row)

                        stats[f"mission3_{split}"] += 1

                        counter = (
                            symptom_counts_train
                            if split == "train"
                            else symptom_counts_valid
                        )

                        for symptom in symptoms:
                            counter[symptom] += 1

    finally:
        for f, _ in files_and_writers.values():
            f.close()

    # --------------------------------------------------------
    # Reports
    # --------------------------------------------------------

    split_id_overlap = split_ids["train"] & split_ids["valid"]

    summary_path = reports_dir / "summary.txt"

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("DCC 119 Dataset Builder Summary\n")
        f.write("=" * 72 + "\n")
        f.write(f"Base directory: {base_dir}\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(f"Internal validation ratio: {valid_ratio:.2%}\n")
        f.write("\n")

        f.write("[Conversation split]\n")
        f.write(f"train conversations: {len(split_ids['train']):,}\n")
        f.write(f"valid conversations: {len(split_ids['valid']):,}\n")
        f.write(
            f"train/valid overlap: {len(split_id_overlap):,}\n"
        )
        f.write("\n")

        f.write("[JSON / WAV]\n")
        f.write(
            f"JSON files seen: {stats['json_files_seen']:,}\n"
        )
        f.write(
            f"Conversations processed: "
            f"{stats['conversations_processed']:,}\n"
        )
        f.write(
            f"Invalid JSON: {stats['invalid_json']:,}\n"
        )
        f.write(
            f"Unmatched audio: {stats['unmatched_audio']:,}\n"
        )
        f.write("\n")

        f.write("[Mission 1]\n")
        f.write(
            f"Train samples: {stats['mission1_train']:,}\n"
        )
        f.write(
            f"Valid samples: {stats['mission1_valid']:,}\n"
        )

        for split in ("train", "valid"):
            f.write(f"  {split} gender:\n")
            for gender in ("M", "F"):
                f.write(
                    f"    {gender}: "
                    f"{gender_counts[(split, gender)]:,}\n"
                )

        f.write("\n")

        f.write("[Mission 2]\n")
        f.write(
            f"Train samples: {stats['mission2_train']:,}\n"
        )
        f.write(
            f"Valid samples: {stats['mission2_valid']:,}\n"
        )

        for split in ("train", "valid"):
            f.write(f"  {split} speaker:\n")
            for speaker in (0, 1):
                f.write(
                    f"    {speaker}: "
                    f"{speaker_counts[(split, speaker)]:,}\n"
                )

        f.write("\n")

        f.write("[Mission 3]\n")
        f.write(
            f"Train conversations: {stats['mission3_train']:,}\n"
        )
        f.write(
            f"Valid conversations: {stats['mission3_valid']:,}\n"
        )

        f.write("\n  Train symptom counts:\n")
        for symptom in TARGET_SYMPTOMS:
            f.write(
                f"    {symptom}: "
                f"{symptom_counts_train[symptom]:,}\n"
            )

        f.write("\n  Valid symptom counts:\n")
        for symptom in TARGET_SYMPTOMS:
            f.write(
                f"    {symptom}: "
                f"{symptom_counts_valid[symptom]:,}\n"
            )

        f.write("\n")

        f.write("[Leakage check]\n")
        f.write(
            f"Conversation IDs present in both splits: "
            f"{len(split_id_overlap):,}\n"
        )

        if split_id_overlap:
            f.write(
                "WARNING: overlap detected. "
                "Do not start model training until this is resolved.\n"
            )
        else:
            f.write("OK: no conversation overlap.\n")

    # Console summary
    print()
    print()
    print("=" * 72)
    print("BUILD COMPLETE")
    print("=" * 72)

    print(
        f"Conversations processed : "
        f"{stats['conversations_processed']:,}"
    )
    print(
        f"Unmatched audio         : "
        f"{stats['unmatched_audio']:,}"
    )
    print(
        f"Invalid JSON             : "
        f"{stats['invalid_json']:,}"
    )

    print()
    print(
        f"Mission 1 train/valid   : "
        f"{stats['mission1_train']:,} / "
        f"{stats['mission1_valid']:,}"
    )

    print(
        f"Mission 2 train/valid   : "
        f"{stats['mission2_train']:,} / "
        f"{stats['mission2_valid']:,}"
    )

    print(
        f"Mission 3 train/valid   : "
        f"{stats['mission3_train']:,} / "
        f"{stats['mission3_valid']:,}"
    )

    print()
    print(
        f"Conversation overlap    : "
        f"{len(split_id_overlap):,}"
    )

    print()
    print(f"Output: {output_dir}")
    print(f"Summary: {summary_path}")

    if stats["unmatched_audio"] != 0:
        print(
            "\nWARNING: unmatched audio exists. "
            "Inspect reports/unmatched_audio.csv."
        )

    if stats["invalid_json"] != 0:
        print(
            "\nWARNING: invalid JSON exists. "
            "Inspect reports/invalid_json.csv."
        )

    if split_id_overlap:
        raise RuntimeError(
            "Conversation leakage detected: train/valid overlap != 0."
        )


# ------------------------------------------------------------
# 7. CLI
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build DCC 119 Mission 1~3 metadata from extracted Training data."
    )

    parser.add_argument(
        "--base-dir",
        type=str,
        default=str(DEFAULT_BASE_DIR),
        help="...\\3.개방데이터\\1.데이터",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="dcc119_generated",
        help="Output directory for CSV files.",
    )

    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=DEFAULT_VALID_RATIO,
        help="Internal train/valid ratio at conversation level. Default=0.10",
    )

    args = parser.parse_args()

    if not 0 < args.valid_ratio < 1:
        raise ValueError("--valid-ratio must be between 0 and 1.")

    build_dataset(
        base_dir=Path(args.base_dir),
        output_dir=Path(args.output_dir),
        valid_ratio=args.valid_ratio,
    )


if __name__ == "__main__":
    main()
