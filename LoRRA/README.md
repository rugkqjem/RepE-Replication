# LoRRA (Low-Rank Representation Adaptation) Implementation
이 디렉토리는 Llama-2 모델의 제어 능력을 향상 시키기 위한 **LoRRA(Low-Rank Representation Adaptation)** 기법의 재구현 및 실험 코드를 포함하고 있습니다. 


## 📌개요 
LoRRA는 모델의 내부 표현(Representation)을 직접 제어하여 특정 속성 (Honesty, Truthfulness)을 조절하는 기법입니다. 본 구현체는 **RepE(Representation Engineering)** 프레임워크를 기반으로 합니다.
## 🙏 Acknowledgements
본 구현체는 [andyzoujm/representation-engineering](https://github.com/andyzoujm/representation-engineering)의 공식 레포지토리 코드를 기반으로 하며, 최신 라이브러리 환경(`transformers 4.47.1`, `peft 0.11.1`)에서 동작할 수 있도록 환경 설정 및 일부 로직을 수정하여 재구현하였습니다.


## 🛠 환경 설정
아래의 설정을 권장합니다. 

### 1. 권장 사양
* **Python**: 3.10 이상
* **PyTorch**:2.4.0 (CUDA 12.1 빌드 권장)
* **Transformers**: 4.47.1

### 2. 설치 방법
```bash
#PyTorch 설치(CUDA 12.1 기준)
pip install torch==2.4.0 --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

#라이브러리 설치
pip install transformers==4.47.1 peft==0.11.1 datasets==4.5.0 bitsandbytes==0.49.1 accelerate==1.12.0
```

**`> [!CAUTION]`** `Transformers` 버전이 5.x 이상으로 올라갈 경우, 기존 `TrainingArguments` 인자들과 호환되지 않아 에러가 발생할 수 있습니다. 반드시 4.47.1 버전을 유지해 주세요.

## 📂 파일 구조
* `train_lorra.py` : LoRRA 학습 메인 스크립트
* `data_preprocessing.py` : TruthfulQA, ARC 등 데이터 셋 로딩 및 전처리 
* `args.py` : 학습 하이퍼파라미터 및 설정 인자 정의

### 실행 방법 
터미널에서 제공된 쉘 스크립트를 통해 학습을 시작할 수 있습니다. 
```bash
bash run_train.sh
```