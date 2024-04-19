# 프롬프트 엔지니어링을 활용한 법원 판결문 요약 연구
---
- 로스쿨에서 판결문 사례를 공부할 때 활용하는 케이스브리프(casebrief)를 키워드로 추출하여 판결문 요약 진행
- 향후 연구 내용 추후 업데이트 예정

### 법원 판결문 요약 사용 방법
```
python3 generation.py \
    --cot <cot> \ # [None, "cot", "law", "casebrief", "t5"]
    --start_id <start_id> \ # int
    --end_id <end_id> # int 
```

### 평가지표 계산 사용 방법
```
cd evaluation
python3 eva.py \
    --cot <cot> \ # [None, "cot", "law", "caebrief", "t5"]
    --start_id <start_id> \ # int
    --end_id <end_id> # int
```