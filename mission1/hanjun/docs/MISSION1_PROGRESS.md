# Mission 1 Progress

## 1. 데이터 확인

서울 Training `구급` 데이터에서 WAV와 JSON을 매칭했다.

- WAV: 62,362
- JSON: 62,362
- WAV ↔ JSON match: 62,362
- invalid JSON: 0

## 2. 신고자 발화 추출

Mission 1의 목표가 신고자 성별 분류이므로 `Speaker=1`인 발화를 사용했다.

`startAt ~ endAt` 구간을 잘라 신고자 발화 샘플로 사용했다.

## 3. 소규모 baseline dataset

- Train: 2,000
- Valid: 1,000
- Train gender: F 1,010 / M 990
- Valid gender: F 499 / M 501
- Audio: 16kHz / mono / 4 seconds

## 4. Baseline model

```text
Audio
→ Log-Mel Spectrogram
→ Small CNN
→ M / F
```

### Best experiment

- Best epoch: 4
- Train Accuracy: 76.50%
- Validation Accuracy: 75.20% (학습 중 표시된 기준)
- 재검증 결과의 utterance Accuracy: 74.00%

> 재현 실행에서는 seed / cache / 학습 상태에 따라 결과가 달라질 수 있으므로, 이후 비교 실험에서는 같은 설정을 유지한다.

## 5. Conversation-level validation

Validation 1,000 utterances가 920개 conversation으로 구성되어 있었다.

- Train/Valid conversation overlap: 0
- Majority Vote Accuracy: 74.57%
- Mean Probability Accuracy: 75.00%
- Majority ties: 22

Conversation-level 100%는 새 평가 파이프라인에서 재현되지 않았다.

## 6. 현재 판단

현재 결과는 baseline 확보 단계로 본다.

다음 우선순위는:

1. 더 많은 학습 데이터 확보
2. 입력 길이 비교
3. ResNet18 등 모델 비교
4. 최종 inference pipeline 구축
