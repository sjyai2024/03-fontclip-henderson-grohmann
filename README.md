# 03B2 FontCLIP Henderson–Grohmann v1.0

## 실행
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 첫 실행
1. `logo_dataset_68_fixed.zip` 업로드
2. `A only` 유지
3. Permutation `10000`
4. `03B2 분석 실행`
5. 결과 ZIP을 다운로드하여 ChatGPT에 업로드

## 분석의 핵심
FontCLIP에 Aaker traits를 직접 묻지 않습니다.
FontCLIP으로 서체 조형특성 5개를 측정하고,
Grohmann et al. (2013)의 published standardized coefficients로 Aaker 5D를 도출합니다.

Primary H1:
**same-brand Text–Typeface distance < mismatched-brand distance**
