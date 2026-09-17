# DCC119 Mission 1 — 신고자 성별 분류

2026 DCC 대학부 예선의 **Mission 1**을 진행하며 만든 데이터 전처리 및 음성 분류 baseline 기록이다.

## 문제 정의

Mission 1의 목표는 **음성으로부터 신고자의 성별을 분류**하는 것이다.

- Input: 원천 음성
- 출력: `M` / `F`
- 학습 시 사용한 annotation: `startAt`, `endAt`, `Speaker`
- 평가 지표: Accuracy

현재 baseline은 신고자의 발화(`Speaker=1`)만 추출한 뒤 음성을 16kHz mono 4초로 통일하고, Log-Mel Spectrogram을 만든 다음 Small CNN으로 M/F를 분류한다.

## 현재 진행 상황

현재 실험은 서울 `구급` Training 데이터의 소규모 subset으로 진행했다.

- Train: 2,000 samples
- Validation: 1,000 utterances
- Validation conversations: 920
- Input audio: 16kHz / mono / 4 sec
- Feature: Log-Mel Spectrogram
- Model: Small CNN
- Best utterance-level Accuracy: **74.00%**
- Conversation-level Majority Vote Accuracy: **74.57%**
- Conversation-level Mean Probability Accuracy: **75.00%**
- Train/Validation conversation overlap: **0**

현재 conversation-level 100%는 재현되지 않았으며, 이후 실험에서는 데이터 규모와 모델 구조를 확대해 성능을 비교할 예정이다.

ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
**[09/17]**
현재 진행 상황

1. 서울 구급 Training 데이터의 WAV/JSON 매칭 및 신고자 발화 추출
2. 16kHz mono 4초 음성 데이터셋 구성
3. Log-Mel Spectrogram 기반 Small CNN baseline 구축
4. 2,000/1,000 subset에서 약 74% Accuracy 확인
5. 10,000/3,000 대규모 subset으로 확장
6. Best Validation Accuracy 75.60% 확인
7. Conversation-level 평가 및 Train/Valid conversation overlap 검증

## 저장소 구성

```text
DCC119-Mission1/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── DCC119_Mission1_Baseline_and_Validation.ipynb
├── scripts/
│   └── build_dcc_dataset.py
└── docs/
    └── MISSION1_PROGRESS.md
```

## 실행 순서

### 1. Google Colab

Colab Runtime을 GPU로 설정한다.

### 2. 데이터

경쟁 데이터는 저장소에 업로드하지 않는다.

현재 개발 데이터는 로컬/Google Drive에서 별도로 관리한다.

예시:

```text
Google Drive
└── DCC119/
    ├── Training/
    └── mission1/
```

### 3. Notebook

`notebooks/DCC119_Mission1_Baseline_and_Validation.ipynb`를 Colab에서 실행한다.

현재 notebook은 3,000개 subset 기반 baseline 재현과 평가 검증을 위한 용도다.

## 다음 실험

현재 과제

- 모델의 M/F별 성능 편차 개선
- 더 큰/다양한 서울 Training 데이터 적용
- ResNet18 등 모델과 비교
- 최종 inference.py 구축

## 주의

경쟁 원본 데이터, 대규모 WAV, JSON 라벨, 캐시, 체크포인트는 GitHub 저장소에 올리지 않는다.

이 저장소는 **코드 / 실험 노트북 / 실행 방법 / 결과 기록**을 관리하는 용도다.
