# 코드 running을 위한 정보

## 상위 디렉토리 설명

일단 연구 repository: isedgx.kaist.ac.kr/home/dmlab/sundong/revisit/ 

코드 디렉토리 
* __survival-revisit-code/tensorflow2_0/__: 이 디렉토리 안에 있는 게 현재 버전 코드 
    * __Tensorflow2.0 alpha에서 작동__: https://pypi.org/project/tf-nightly-gpu-2.0-preview/
    * __CUDA10, CUDNN 7.4.1__ 이상인가 필요
    * __requirement.txt__: 필요한 library들 - pip install -r requirement.txt 커맨드로 한번에 설치 가능
    * survival-revisit-code/: 과거 버전 코드가 있을 수 있음
    * survival-revisit-code/keras: 무시해도 됨 코드 디렉토리

코드 디테일
* main.py: 메인인데 현재 안씀 -  multiprocessing, gpu 배치 등을 위해 만들었었음
* __survrevtensorflow2.py__: 우리 SurvRev 모델 메인 - 현재는 이거 돌리면 eval.poisson_process_baseline() 형식으로 베이스라인들도 같이 돌아감  (line 200)
    * __loss.py__: SurvRev 모델을 위한 loss 정의
    * __evaluation.py__: SurvRev 모델을 평가를 위한 procedure 정의
    * __params.py__: 각종 파라미터 - 다른 코드에도 통용됨
    * __utils.py__: 각종 code snippet
    * survrev.py, survrevclas.py, survrevregr.py, survrevt.py: Deprecated  (survrev.py는 Keras 버전 - TF 1에서 작동)
* __aaai19tensorflow2.py__: DRSA[AAAI19] 논문 우리 코드에 돌아가게 구현한 것
* __wsdm17nsrtensorflow2.py__: Neural survival recommder[WSDM17] 논문 우리 코드에 돌아가게 구현한 것
* numtr.sh: 현재 돌리고 있는 반복실험 bash 코드
* MHP.py: 다른 베이스라인(Hawkes)에 필요한 소스코드

데이터 관련
* data_raw/indoor/: ZOYI 실내 움직임 원본 데이터 - store A, B, C, D, E 
* __data/indoor/__: 이번 survival 논문을 쓰면서 만든 벤치마크 데이터 (50,000명)
    * store A-E로 구성되어 있으며, train_240days/ 디렉토리는 전체 1년치 중 240일을 트레이닝 기간으로 만든 데이터임 
    * 트레이닝 기간이 길어질수록 트레이닝 셋 내의 재방문 확률이 높을 수 있음
* __data_sample_5000/__: 위 벤치마크 데이터에서 5000명을 뽑은 작은 데이터셋 - 일단 이걸로 돌려서 디펜스 준비하는 중
* __data_sample_1000/__: 위 벤치마크 데이터에서 1000명을 뽑은 토이 데이터셋 - 코드 mini-run에 용이
* notebook_prepare_open_dataset/: ZOYI 원본 데이터에서 벤치마크 데이터를 만드는 코드가 있음 (.py 파일과 .sh 파일)
* data_sample/: data_sample_1000과 같음, 무시해도 됨

실험 결과 로깅 
* __results/__: 여기에 일단 저장됨 
    * results/all/: 전체 run의 주요 결과가 aggregated되어 all_results.csv 파일로 저장
    * results/callback/: 각 run의 epoch별 결과가 {unique_exp_ID}.csv로 저장 <- all_results.csv과 exp_id를 key로 이어져 있음.
    * results/epochresults.csv: 무시해도 됨

* notebook/: 각종 exploratory data anaiysis

현재는 필요 없는 참고용 repo
* DeepHit/: DeepHit 저자들이 제공한 repo
* drsa/: DRSA 저자들이 제공한 repo + 환준 초기 버전 
* tutorials/: Unnecessary
* wsdm-code/: Deprecated
# 코드 running을 위한 정보

## 상위 디렉토리 설명

일단 연구 repository: isedgx.kaist.ac.kr/home/dmlab/sundong/revisit/ 

코드 디렉토리 
* __survival-revisit-code/tensorflow2_0/__: 이 디렉토리 안에 있는 게 현재 버전 코드 
    * __Tensorflow2.0 alpha에서 작동__: https://pypi.org/project/tf-nightly-gpu-2.0-preview/
    * __CUDA10, CUDNN 7.4.1__ 이상인가 필요
    * __requirement.txt__: 필요한 library들 - pip install -r requirement.txt 커맨드로 한번에 설치 가능
    * survival-revisit-code/: 과거 버전 코드가 있을 수 있음
    * survival-revisit-code/keras: 무시해도 됨 코드 디렉토리

코드 디테일
* main.py: 메인인데 현재 안씀 -  multiprocessing, gpu 배치 등을 위해 만들었었음
* __survrevtensorflow2.py__: 우리 SurvRev 모델 메인 - 현재는 이거 돌리면 eval.poisson_process_baseline() 형식으로 베이스라인들도 같이 돌아감  (line 200)
    * __loss.py__: SurvRev 모델을 위한 loss 정의
    * __evaluation.py__: SurvRev 모델을 평가를 위한 procedure 정의
    * __params.py__: 각종 파라미터 - 다른 코드에도 통용됨
    * __utils.py__: 각종 code snippet
    * survrev.py, survrevclas.py, survrevregr.py, survrevt.py: Deprecated  (survrev.py는 Keras 버전 - TF 1에서 작동)
* __aaai19tensorflow2.py__: DRSA[AAAI19] 논문 우리 코드에 돌아가게 구현한 것
* __wsdm17nsrtensorflow2.py__: Neural survival recommder[WSDM17] 논문 우리 코드에 돌아가게 구현한 것
* numtr.sh: 현재 돌리고 있는 반복실험 bash 코드
* MHP.py: 다른 베이스라인(Hawkes)에 필요한 소스코드

데이터 관련
* data_raw/indoor/: ZOYI 실내 움직임 원본 데이터 - store A, B, C, D, E 
* __data/indoor/__: 이번 survival 논문을 쓰면서 만든 벤치마크 데이터 (50,000명)
    * store A-E로 구성되어 있으며, train_240days/ 디렉토리는 전체 1년치 중 240일을 트레이닝 기간으로 만든 데이터임 
    * 트레이닝 기간이 길어질수록 트레이닝 셋 내의 재방문 확률이 높을 수 있음
* __data_sample_5000/__: 위 벤치마크 데이터에서 5000명을 뽑은 작은 데이터셋 - 일단 이걸로 돌려서 디펜스 준비하는 중
* __data_sample_1000/__: 위 벤치마크 데이터에서 1000명을 뽑은 토이 데이터셋 - 코드 mini-run에 용이
* notebook_prepare_open_dataset/: ZOYI 원본 데이터에서 벤치마크 데이터를 만드는 코드가 있음 (.py 파일과 .sh 파일)
* data_sample/: data_sample_1000과 같음, 무시해도 됨

실험 결과 로깅 
* __results/__: 여기에 일단 저장됨 
    * results/all/: 전체 run의 주요 결과가 aggregated되어 all_results.csv 파일로 저장
    * results/callback/: 각 run의 epoch별 결과가 {unique_exp_ID}.csv로 저장 <- all_results.csv과 exp_id를 key로 이어져 있음.
    * results/epochresults.csv: 무시해도 됨

* notebook/: 각종 exploratory data anaiysis

현재는 필요 없는 참고용 repo
* DeepHit/: DeepHit 저자들이 제공한 repo
* drsa/: DRSA 저자들이 제공한 repo + 환준 초기 버전 
* tutorials/: Unnecessary
* wsdm-code/: Deprecated
