from pathlib import Path
import pandas as pd
import numpy as np
import soundfile as sf
import random
from tqdm import tqdm

# =========================================================
# 1. 경로
# =========================================================

BASE_DIR = Path(
    r"C:\Users\hanju\Downloads\088.위급상황 음성-음향_고도화_119 지능형 신고접수 음성 인식 데이터"
)

DATA_ROOT = BASE_DIR / "3.개방데이터" / "1.데이터"

TS_ROOT = (
    DATA_ROOT
    / "Training"
    / "1.원천데이터"
    / "TS_서울_구급"
)

CSV_ROOT = DATA_ROOT / "dcc119_gubup_only"

TRAIN_CSV = CSV_ROOT / "mission1_train.csv"
VALID_CSV = CSV_ROOT / "mission1_valid.csv"

OUT_ROOT = Path(
    r"C:\Users\hanju\Downloads\dcc119_m1_large_subset"
)

# =========================================================
# 2. 설정
# =========================================================

TRAIN_TARGET = 10_000
VALID_TARGET = 3_000

SEED = 42

TARGET_SR = 16_000
TARGET_SECONDS = 4
TARGET_SAMPLES = TARGET_SR * TARGET_SECONDS

random.seed(SEED)
np.random.seed(SEED)

# =========================================================
# 3. 파일 존재 확인
# =========================================================

print("TS_ROOT :", TS_ROOT)
print("TRAIN_CSV:", TRAIN_CSV)
print("VALID_CSV:", VALID_CSV)

if not TS_ROOT.exists():
    raise FileNotFoundError(
        f"TS_서울_구급 폴더가 없습니다:\n{TS_ROOT}"
    )

if not TRAIN_CSV.exists():
    raise FileNotFoundError(
        f"mission1_train.csv가 없습니다:\n{TRAIN_CSV}"
    )

if not VALID_CSV.exists():
    raise FileNotFoundError(
        f"mission1_valid.csv가 없습니다:\n{VALID_CSV}"
    )

# =========================================================
# 4. CSV 읽기
# =========================================================

train_df = pd.read_csv(TRAIN_CSV)
valid_df = pd.read_csv(VALID_CSV)

print("\n===== CSV 확인 =====")

print("Train shape:", train_df.shape)
print("Valid shape:", valid_df.shape)

print("\nTrain gender:")
print(train_df["gender"].value_counts())

print("\nValid gender:")
print(valid_df["gender"].value_counts())

print("\nTrain conversations:",
      train_df["conversation_id"].nunique())

print("Valid conversations:",
      valid_df["conversation_id"].nunique())

# conversation overlap
train_conv = set(
    train_df["conversation_id"].astype(str)
)

valid_conv = set(
    valid_df["conversation_id"].astype(str)
)

overlap = train_conv & valid_conv

print(
    "Train/Valid conversation overlap:",
    len(overlap)
)

if overlap:
    raise RuntimeError(
        "Train/Valid conversation overlap detected!"
    )

# =========================================================
# 5. 성별 균형 샘플링
# =========================================================

def balanced_sample(df, target):
    half = target // 2

    f = df[df["gender"] == "F"].sample(
        n=half,
        random_state=SEED
    )

    m = df[df["gender"] == "M"].sample(
        n=target - half,
        random_state=SEED
    )

    result = pd.concat(
        [f, m],
        ignore_index=True
    )

    result = result.sample(
        frac=1,
        random_state=SEED
    ).reset_index(drop=True)

    return result


selected_train = balanced_sample(
    train_df,
    TRAIN_TARGET
)

selected_valid = balanced_sample(
    valid_df,
    VALID_TARGET
)

print("\n===== 선택 결과 =====")

print(
    "Train:",
    len(selected_train),
    selected_train["gender"].value_counts().to_dict()
)

print(
    "Valid:",
    len(selected_valid),
    selected_valid["gender"].value_counts().to_dict()
)

# =========================================================
# 6. WAV 경로 해결
# =========================================================

# audio_path가 데이터 루트 기준 상대경로라면
# 파일이 실제로 어디에 있는지 확인

def resolve_audio(row):

    audio_name = str(row["audio_file_name"])

    # 1. TS_ROOT 아래에서 파일명 검색
    matches = list(
        TS_ROOT.rglob(audio_name)
    )

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise RuntimeError(
            f"동일한 WAV가 여러 개 발견됨: {audio_name}"
        )

    raise FileNotFoundError(
        f"WAV를 찾을 수 없습니다: {audio_name}"
    )

# =========================================================
# 7. 4초 audio extraction
# =========================================================

def extract_4sec(
    wav_path,
    start_ms,
    end_ms
):

    with sf.SoundFile(
        str(wav_path),
        mode="r"
    ) as f:

        src_sr = f.samplerate

        # 발화 중앙
        center_ms = (
            float(start_ms)
            + float(end_ms)
        ) / 2

        # 중앙 기준 앞뒤 2초
        start_ms_4s = center_ms - 2000

        start_frame = int(
            start_ms_4s
            * src_sr
            / 1000
        )

        start_frame = max(
            0,
            start_frame
        )

        f.seek(start_frame)

        frames = int(
            TARGET_SECONDS
            * src_sr
        )

        audio = f.read(
            frames,
            dtype="float32",
            always_2d=True
        )

    # stereo → mono
    audio = audio.mean(axis=1)

    # sample rate 변환
    if src_sr != TARGET_SR:

        import librosa

        audio = librosa.resample(
            audio,
            orig_sr=src_sr,
            target_sr=TARGET_SR
        )

    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    # 정확히 4초
    if len(audio) >= TARGET_SAMPLES:

        extra = (
            len(audio)
            - TARGET_SAMPLES
        )

        left = extra // 2

        audio = audio[
            left:left + TARGET_SAMPLES
        ]

    else:

        padded = np.zeros(
            TARGET_SAMPLES,
            dtype=np.float32
        )

        left = (
            TARGET_SAMPLES
            - len(audio)
        ) // 2

        padded[
            left:left + len(audio)
        ] = audio

        audio = padded

    return audio

# =========================================================
# 8. 저장 함수
# =========================================================

def process_split(
    df,
    split_name
):

    out_root = (
        OUT_ROOT
        / split_name
    )

    (out_root / "F").mkdir(
        parents=True,
        exist_ok=True
    )

    (out_root / "M").mkdir(
        parents=True,
        exist_ok=True
    )

    metadata = []
    failures = []

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=f"{split_name}"
    ):

        gender = str(
            row["gender"]
        )

        conversation_id = str(
            row["conversation_id"]
        )

        utterance_index = int(
            row["utterance_index"]
        )

        sample_id = (
            f"{conversation_id}"
            f"_utt{utterance_index:04d}"
            f"_{gender}"
        )

        out_path = (
            out_root
            / gender
            / f"{sample_id}.wav"
        )

        try:

            wav_path = resolve_audio(row)

            audio = extract_4sec(
                wav_path,
                row["startAt"],
                row["endAt"]
            )

            sf.write(
                str(out_path),
                audio,
                TARGET_SR,
                subtype="PCM_16"
            )

            metadata.append({
                "sample_id": sample_id,
                "conversation_id": conversation_id,
                "original_audio":
                    row["audio_file_name"],
                "startAt":
                    row["startAt"],
                "endAt":
                    row["endAt"],
                "gender":
                    gender,
                "output_path":
                    str(out_path),
                "utterance_index":
                    utterance_index,
            })

        except Exception as e:

            failures.append({
                "sample_id":
                    sample_id,
                "audio":
                    row["audio_file_name"],
                "error":
                    repr(e)
            })

    return metadata, failures

# =========================================================
# 9. 실행
# =========================================================

OUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

train_meta, train_failures = process_split(
    selected_train,
    "train"
)

valid_meta, valid_failures = process_split(
    selected_valid,
    "valid"
)

# =========================================================
# 10. metadata 저장
# =========================================================

META_ROOT = (
    OUT_ROOT / "metadata"
)

META_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

train_meta_df = pd.DataFrame(
    train_meta
)

valid_meta_df = pd.DataFrame(
    valid_meta
)

train_meta_df.to_csv(
    META_ROOT / "train.csv",
    index=False,
    encoding="utf-8-sig"
)

valid_meta_df.to_csv(
    META_ROOT / "valid.csv",
    index=False,
    encoding="utf-8-sig"
)

# =========================================================
# 11. 최종 결과
# =========================================================

print("\n")
print("=" * 60)
print("Mission 1 Large Subset 완료")
print("=" * 60)

print(
    "Train saved:",
    len(train_meta_df)
)

print(
    "Valid saved:",
    len(valid_meta_df)
)

print(
    "Train failures:",
    len(train_failures)
)

print(
    "Valid failures:",
    len(valid_failures)
)

print(
    "\nTrain gender:",
    train_meta_df["gender"]
    .value_counts()
    .to_dict()
)

print(
    "Valid gender:",
    valid_meta_df["gender"]
    .value_counts()
    .to_dict()
)

print(
    "\nTrain conversations:",
    train_meta_df[
        "conversation_id"
    ].nunique()
)

print(
    "Valid conversations:",
    valid_meta_df[
        "conversation_id"
    ].nunique()
)

print(
    "Conversation overlap:",
    len(
        set(
            train_meta_df[
                "conversation_id"
            ].astype(str)
        )
        &
        set(
            valid_meta_df[
                "conversation_id"
            ].astype(str)
        )
    )
)

print(
    "\nOutput:",
    OUT_ROOT
)