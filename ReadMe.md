### 

사진 크롭 커멘드 = 
python tool/img_crop.py
```

### 데이터셋 폴더 위치설정
```
{$ROOT}
|-- dataset
|    ㄴㅡ img -- 01,02,03
|    ㄴㅡ label -- 01, 02, 03
|    ㄴㅡ cropped_img -- 01, 02, 03
|-- tool

```
### 학습 커멘드
mode는 따로 입력하지 않으면 "육안평가"가 되고 mode를 regression을 입력하면 "정밀 기기측정값" 예측이 된다
```
python tool/main.py --name "체크포인트 이름" --mode "class" or "regression"
예를들어서  python tool/main.py --name "test" --mode "class"
```

### 검증
```
python tool/test.py --name "앞서 저장한 체크포인트 이름" --mode "class" or "regression"
```

### 참고사항

모델 변경 전임
