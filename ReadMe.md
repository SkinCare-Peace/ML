
### 사진 크롭 커멘드 = 
```
python tool/img_crop.py
```


### 데이터셋 폴더 위치설정

  
{$ROOT}   
ㅣ-- dataset  
ㅣ      ㄴㅡ img -- 01,02,03   
ㅣ      ㄴㅡ label -- 01, 02, 03   
ㅣ      ㄴㅡ cropped_img -- 01, 02, 03  
ㅣ-- tool

  
### 요구사항
```
pip install requests einops tensorboardX pillow scipy scikit-learn
```
```
pip install errno json cv2 os tqdm
```

### 학습 커멘드
mode는 따로 입력하지 않으면 "육안평가"가 되고 mode를 regression을 입력하면 "정밀 기기측정값" 예측이 된다
커멘드 : python tool/main.py --name "체크포인트 이름" --mode "class" or "regression"
ex) python tool/main.py --name "test" --mode "class"  // test1.. n 으로

### 검증
```
python tool/test.py --name "
```
앞서 저장한 체크포인트 이름"
```
--mode "class" or "regression"
```


### 참고사항
모델 변경 전임


### 메모
CUDA 설치  
nvidia-smi

