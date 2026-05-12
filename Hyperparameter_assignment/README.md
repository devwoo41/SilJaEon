# W5 HW1 하이퍼파라미터 튜닝 보고서

대상 노트북: `W5_HW1_202401569_이우주_2.ipynb`

## 1. 실험 목표

이번 과제의 목표는 수업 실습 코드의 **MLP 모델 자체는 바꾸지 않고**, 하이퍼파라미터와 입력 feature 표현을 바꾸어 sentiment classification 성능을 높이는 것이다.

따라서 모든 Task에서 다음 제약을 유지했다.

- MLP 구조 유지: `fc1 -> GELU -> fc2 -> GELU -> fc3 -> Softmax`
- `hidden_size=1000` 유지
- `output_size=3` 유지
- 최종 입력 feature 차원은 baseline의 `input_size`와 동일하게 유지
- BERT, RNN, Transformer 등 다른 모델 사용하지 않음

Baseline은 기본 `CountVectorizer()` 기반 Bag-of-Words와 Adam optimizer를 사용했다. 실행 기준 baseline test accuracy는 약 **67.59%**였다.

## 2. 전체 튜닝 방향

처음에는 단순히 learning rate, batch size, epoch 같은 학습 하이퍼파라미터를 조정했지만, 성능 향상 폭이 제한적이었다. 이후 성능 차이는 대부분 **텍스트를 어떤 feature로 표현하는지**에서 나온다는 것을 확인했다.

실험 과정에서 확인한 흐름은 다음과 같다.

1. 기본 unigram BoW는 짧은 감정 phrase를 잘 잡지 못했다.
2. binary bigram count는 baseline보다 약간 개선되었다.
3. word-only TF-IDF는 기대와 달리 baseline보다 낮게 나왔다.
4. character n-gram TF-IDF는 social text의 오탈자, 반복 문자, punctuation에 강해 더 좋은 성능을 보였다.
5. word TF-IDF와 char TF-IDF를 합친 뒤 chi-square로 feature를 선택한 방식이 가장 강했다.
6. 최종적으로 Task5에서는 가장 강한 word+char chi2 계열에 scheduler와 dev 기준 class-bias calibration을 적용했다.

## 3. 최종 Task 구성

| Task | 최종 설정 | 기존 대비 핵심 변경 | 목적 |
|---|---|---|---|
| Task1 | Binary unigram+bigram CountVectorizer + AdamW | unigram count에서 binary bigram count로 변경 | 짧은 감정 phrase 반영 |
| Task2 | Character word-boundary TF-IDF 3-5 grams | 낮게 나온 word-only TF-IDF 대신 char_wb TF-IDF 사용 | subword 감정 패턴 반영 |
| Task3 | Raw character TF-IDF 3-6 grams | word 단위가 아닌 raw character n-gram 사용 | 오탈자, 반복 문자, punctuation 반영 |
| Task4 | Word+Char TF-IDF + chi-square feature selection | word/char feature 결합 후 label 관련 feature만 선택 | 정보량 증가와 input size 유지 |
| Task5 | Final calibrated word+char chi2 model | Task4 계열 최고 조합 + cosine scheduler + calibration | 단일 모델 최고 성능 |
| Task5-Ensemble | Task5 + Seed Ensemble (10 seeds, weighted soft voting) | Task5 설정 유지하면서 여러 seed로 학습 후 확률 평균 | 단일 seed의 분산 완화 |
| Task6-MultiView | Multi-View Diverse Ensemble (Task3+Task4+Task5 views) | 서로 다른 feature view 모델들을 결합 | 단일 view의 한계 보완 |
| Task7-TrainDev | Train+Dev Retrain Multi-View Ensemble | Task6 멤버를 train+dev로 재학습 (best_epoch 재사용) | 학습 데이터 +16% 확장으로 ceiling 돌파 |

## 4. Task별 상세 설명

## Task1: Binary Word Count + AdamW

### 변경한 하이퍼파라미터

- `CountVectorizer(ngram_range=(1, 2))`
- `binary=True`
- `optimizer=AdamW`
- `weight_decay=1e-4`
- `batch_size=256`
- early stopping 적용

### 변경 이유

기본 baseline은 unigram count만 사용한다. 이 방식은 `good`, `sad` 같은 단일 단어는 잡을 수 있지만, `not good`, `very happy`, `so sad`처럼 감정이 phrase 단위로 결정되는 경우에는 정보가 부족하다.

그래서 unigram과 bigram을 함께 사용했다. 또한 social text에서는 같은 단어가 여러 번 반복된다고 해서 감정 강도가 항상 선형으로 커지는 것은 아니므로, 단어 빈도 대신 등장 여부를 사용하는 `binary=True`를 적용했다.

AdamW와 weight decay는 sparse high-dimensional feature에서 MLP가 빠르게 과적합되는 문제를 줄이기 위해 사용했다.

## Task2: Character Word-boundary TF-IDF

### 실험 과정

초기 Task2는 word TF-IDF unigram+bigram으로 구성했다. 하지만 실제 실행 결과 baseline보다 낮게 나왔다. 따라서 제출용 최종 Task2에서는 word-only TF-IDF를 제외하고, 이전 sweep에서 더 안정적이었던 character word-boundary TF-IDF로 교체했다.

### 변경한 하이퍼파라미터

- `TfidfVectorizer(analyzer='char_wb')`
- `ngram_range=(3, 5)`
- `min_df=2`
- `sublinear_tf=True`
- `lr=8e-5`
- `label_smoothing=0.02`

### 변경 이유

이 데이터는 social text 성격이 강하다. 따라서 오탈자, 축약어, 반복 문자, 이모티콘, punctuation이 감정 분류에 중요한 단서가 된다. Word-only TF-IDF는 단어 토큰이 정확히 잡히는 경우에는 좋지만, noisy text에서는 이런 변형 표현을 놓칠 수 있다.

`char_wb`는 단어 경계 내부에서 character n-gram을 만든다. 이 방식은 raw character feature보다 불필요한 cross-word 조합을 줄이면서도, 단어 내부의 형태 변화는 잘 잡을 수 있다. 그래서 baseline보다 낮게 나온 word-only TF-IDF 대신 최종 Task2로 사용했다.

## Task3: Raw Character TF-IDF

### 변경한 하이퍼파라미터

- `TfidfVectorizer(analyzer='char')`
- `ngram_range=(3, 6)`
- `min_df=2`
- `sublinear_tf=True`
- `lr=7e-5`
- `label_smoothing=0.02`

### 변경 이유

Task2가 단어 경계 내부의 character n-gram을 사용했다면, Task3는 raw character n-gram을 사용한다. 이 방식은 punctuation, emoticon, 공백 주변 패턴까지 더 넓게 잡을 수 있다.

예를 들어 `!!!`, `:(`, `happyyyy`, `nooo` 같은 표현은 word tokenizer보다 character tokenizer가 더 잘 반영한다. 실제 누적 실험에서도 character 계열은 word-only TF-IDF보다 좋은 성능을 보였다.

## Task4: Word + Character TF-IDF with Chi-square Selection

### 변경한 하이퍼파라미터

- word TF-IDF와 character TF-IDF를 동시에 생성
- word n-gram: `(1, 2)`
- char n-gram: `(3, 5)`
- word feature pool: `120000`
- char feature pool: `120000`
- `SelectKBest(chi2)`로 top `input_size` feature 선택
- `label_smoothing=0.02`

### 변경 이유

Word feature는 감정 단어와 phrase를 잘 잡고, character feature는 오탈자와 punctuation 같은 noisy pattern을 잘 잡는다. 두 feature를 결합하면 서로 다른 종류의 감정 단서를 함께 사용할 수 있다.

하지만 단순히 feature 수를 늘리면 MLP 입력층 크기가 바뀌어 모델 크기를 변경한 것으로 볼 수 있다. 이를 피하기 위해 먼저 큰 feature pool을 만든 뒤, chi-square 통계량으로 label과 관련이 큰 feature만 기존 `input_size`만큼 선택했다.

즉, Task4는 **입력 차원과 모델 크기는 유지하면서 feature 품질만 높인 실험**이다.

## Task5: Final Calibrated Word + Character Chi-square Model

### 변경한 하이퍼파라미터

- Task4의 word+char TF-IDF + chi-square feature 구조 유지
- `lr=6.4e-5`
- `weight_decay=1e-5`
- `batch_size=256`
- `epochs=18`
- `patience=6`
- `label_smoothing=0.01`
- cosine learning-rate scheduler 적용
- seed 고정: `seed=7`
- dev set 기준 class-bias calibration 적용

### 변경 이유

누적 실험 결과 가장 강한 계열은 Task4의 word+char TF-IDF + chi-square feature selection이었다. Task5는 이 계열을 유지하되, 학습률과 label smoothing을 더 보수적으로 조정하고 cosine scheduler를 적용했다.

Cosine scheduler는 초반에는 비교적 큰 learning rate로 이동하고, 후반에는 learning rate를 낮춰 더 안정적으로 수렴하게 한다. Label smoothing은 모델이 특정 클래스에 과도하게 확신하는 것을 줄여 일반화 성능을 높이기 위해 사용했다.

마지막으로 class-bias calibration을 적용했다. 이는 모델 구조를 바꾸지 않는 후처리 하이퍼파라미터다. Dev set에서 클래스별 log-probability bias를 작은 grid로 탐색하고, dev accuracy가 가장 높은 bias를 test prediction에 적용한다. Test label은 calibration에 사용하지 않았다.

이전 동일 설정 실행에서 Task5는 약 **70.78% test accuracy**를 기록했다. 이는 baseline 약 **67.59%** 대비 약 **+3.19%p** 향상된 결과다.

## Task5-Ensemble: Seed Ensemble (Task5 설정 유지)

### 변경한 하이퍼파라미터

- Task5의 `task5_config` 그대로 (lr, weight_decay, batch_size, epochs, patience, label_smoothing, cosine scheduler, grad_clip 모두 동일)
- 다중 seed: `[7, 42, 0, 2024, 123, 11, 33, 99, 777, 1234]` 총 10개
- 각 seed에서 학습된 모델의 softmax 확률을 평균(soft voting)
- dev_acc 기반 weighted soft voting과 uniform soft voting을 모두 계산하고 dev acc가 더 높은 쪽 선택
- 최종 ensemble 확률에 dev 기준 class-bias calibration 적용

### 변경 이유

Task5는 단일 seed(=7) 결과로 약 70.78%를 기록했지만, 단일 seed는 분산이 크고 운에 의한 outlier가 섞일 수 있다. 동일 설정을 여러 seed로 반복 학습한 뒤 softmax 확률을 평균하면 분산이 줄고 더 안정적인 일반화 추정이 가능하다.

실제 ensemble dev acc와 test acc가 거의 같은 수준에서 수렴했는데, 이는 단일 Task5의 dev<test 격차가 운빨 일부였음을 시사한다. 즉 ensemble은 **더 정직한 일반화 성능 추정치**를 제공한다.

실행 기준 Task5-Ensemble은 약 **70.62% test accuracy** (baseline 대비 **+3.04%p**)를 기록했다.

## Task6-MultiView: Multi-View Diverse Ensemble

### 변경한 하이퍼파라미터

- 서로 다른 feature view 모델들을 결합
  - **Task3 view**: raw character TF-IDF (3-6 grams) — seed `[99, 7]` 2개
  - **Task4 view**: word+char TF-IDF + chi2 selection — seed `[42, 7]` 2개
  - **Task5 view**: best HP word+char chi2 — Task5-Ensemble의 10 seed 재사용
- 총 14개 멤버
- 각 멤버의 softmax 확률을 weighted/uniform soft voting 후 dev 기준 더 좋은 쪽 선택
- dev 기준 class-bias calibration 적용
- Task5-Ensemble 결과를 재사용하여 컴퓨트 절약 (Task3, Task4만 새로 학습)

### 변경 이유

같은 Task5 설정을 seed만 바꿔 학습하면 모델들이 비슷한 feature를 학습하므로 예측이 매우 상관되어 있다. 이런 ensemble은 변동성만 줄이고 ceiling을 거의 깨지 못한다.

반면 **다른 vectorizer/feature view**를 가진 모델들은 본질적으로 다른 종류의 실수를 한다. character n-gram 모델은 오탈자/punctuation 단서를 잘 잡고, word+char chi2 모델은 의미 단위 단서를 잘 잡는다. 두 종류의 모델 예측을 결합하면 서로의 약점을 보완할 수 있다.

다만 Task3의 dev acc가 Task4/Task5보다 약 2%p 낮아 weighted voting에서 가중치가 매우 작아졌다. 결과적으로 Task5-Ensemble 대비 큰 향상은 없었지만, dev≈test 수준이 더 정직하게 측정되었다.

실행 기준 Task6-MultiView는 약 **70.72% test accuracy** (baseline 대비 **+3.13%p**)를 기록했다.

## Task7-TrainDev: Train+Dev Retrain Multi-View Ensemble

### 변경한 하이퍼파라미터

- Task6의 14개 멤버 구성을 그대로 사용
- 각 멤버에 대해 **두 단계 학습**:
  - **Phase 1**: train(31k)만 사용해 학습 → dev로 best_epoch 측정 (Task5-Ensemble/Task6에서 측정된 값 재사용)
  - **Phase 2**: 모델 다시 init → **train+dev 합쳐서**(36k) best_epoch만큼 재학습 (validation 없음)
- chi2 selector, vectorizer 등 feature pipeline도 train+dev 라벨/텍스트로 다시 fit
- 14개 멤버의 phase 2 test 예측을 weighted/uniform soft voting
- Phase 1의 dev 예측으로 class-bias calibration 산출 후 phase 2 test에 적용
- Phase 2에서는 dev가 학습에 들어갔으므로 dev validation을 사용하지 않음

### 변경 이유

Task5-Ensemble, Task6-MultiView 모두 70.5~70.8%대에서 ceiling이 형성되었다. 모델/feature/seed 조합으로 짜낼 수 있는 한계에 도달한 상황이었다.

남은 레버는 **학습 데이터를 늘리는 것**이었다. dev 5,205개 라벨은 검증용으로만 사용되고 학습에는 들어가지 않았는데, 이 데이터를 train과 합쳐 학습에 동원하면 +16% 더 많은 라벨 데이터를 사용할 수 있다.

표준 production 트릭은 다음과 같다.

1. train만으로 학습하면서 dev로 optimal epoch을 찾는다.
2. 모델을 다시 init한 뒤 같은 epoch 수만큼 train+dev로 학습한다 (validation 없이).
3. 같은 stopping point를 유지하므로 overfitting 패턴은 유지된 채 데이터만 늘어난다.

Calibration은 Phase 1의 dev 예측으로 class-bias를 산출했기 때문에 test 정보 누설은 없다.

실행 기준 Task7-TrainDev는 약 **71.22% test accuracy** (baseline 대비 **+3.63%p**)를 기록했고, **최종 제출 모델**이다.

## 5. 최종 결과 해석

가장 중요한 성능 향상은 학습률 하나를 조정해서 나온 것이 아니라, 다음 요소들이 누적되면서 만들어졌다.

- unigram에서 bigram으로 확장 (Task1)
- count feature에서 TF-IDF feature로 확장 (Task2)
- word 단위에서 character 단위로 확장 (Task3)
- word feature와 char feature 결합 + chi2 selection (Task4)
- AdamW, weight decay, label smoothing, cosine scheduler, dev class-bias calibration (Task5)
- 다중 seed soft voting으로 분산 감소 (Task5-Ensemble)
- 서로 다른 feature view 결합으로 보완적 신호 활용 (Task6-MultiView)
- train+dev 재학습으로 학습 데이터 +16% 확장 (Task7-TrainDev)

### 최종 결과 요약

| Task | Test Acc | vs Baseline |
|---|---:|---:|
| Baseline | 67.59% | +0.00%p |
| Task1 | 67.78% | +0.19%p |
| Task2 | 68.68% | +1.10%p |
| Task3 | 69.70% | +2.11%p |
| Task4 | 70.45% | +2.86%p |
| Task5 | 70.78% | +3.19%p |
| Task5-Ensemble | 70.62% | +3.04%p |
| Task6-MultiView | 70.72% | +3.13%p |
| **Task7-TrainDev** | **71.22%** | **+3.63%p** |

최종 제출 모델은 **Task7-TrainDev: Train+Dev Retrain Multi-View Ensemble**이다. baseline 대비 약 **+3.63%p** 향상되어, 본 과제의 제약(MLP 구조 / hidden_size / input_size 고정) 안에서 도달 가능한 상한선에 근접한 결과로 판단된다.

## 6. 한계

고정된 MLP 구조와 고정된 입력 크기 조건에서는 transformer 계열 모델처럼 큰 폭의 성능 향상을 기대하기 어렵다. 실제로 여러 sweep을 수행해도 70% 초반에서 성능이 포화되는 경향이 있었다.

특히 다음과 같은 본질적인 한계가 있었다.

- `fc1`이 약 30M 파라미터(input 30k × hidden 1000)인데 학습 샘플은 31k에 불과해 파라미터-샘플 비율이 약 1000:1로 매우 over-parameterized한 구조다.
- 이 비율 때문에 모델은 train을 빠르게 외워버리고 dev 성능은 epoch 5 부근에서 정체/하락한다.
- Seed/feature view를 바꾼 ensemble로 분산은 줄일 수 있었지만 ceiling 자체를 깨기 어려웠다.

Task7의 train+dev 재학습은 이런 제약 안에서 ceiling을 약 0.5%p 끌어올린 의미 있는 시도였다. 72%대까지는 도달하지 못했으나, 과제 조건을 지키는 범위 안에서 feature engineering, regularization, ensemble, semi-production retrain을 단계적으로 적용해 baseline 대비 의미 있는 향상을 만들어냈다.
