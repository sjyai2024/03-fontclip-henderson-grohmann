# 03B2 FontCLIP → Henderson/Grohmann → Aaker 5D 방법 v1.0
**고정일:** 2026-09-27

## 1. 목적
FontCLIP에 Aaker의 브랜드 개성 형용사를 직접 넣어 브랜드 개성을 측정하지 않는다.
FontCLIP은 먼저 선행연구에서 정의된 **서체 조형특성**을 측정하고,
그 특성에서 브랜드 개성으로 이어지는 관계는 Grohmann, Giese & Parkman (2013)의
공개된 표준화 회귀계수를 외부 고정계수로 사용한다.

## 2. 이론적 경로
`wordmark image → FontCLIP → typeface design characteristics → published mapping → Aaker 5D`

### Typeface characteristics
Grohmann et al. (2013)은 Henderson et al. (2004)에 기반하여:
- Harmony: symmetry and balance
- Natural: representative and organic
- Elaborate: complexity and depth
- Weight: heavy and compressed
- Flourish: presence/absence of serifs
로 설명한다.

본 연구의 FontCLIP prompt proxy는 이 정의에서만 도출한다.

### Prompt ensemble
- Elaborate: `This is an elaborate font.` + `This is a complex font.`
- Harmony: `This is a harmonious font.` + `This is a symmetrical font.` + `This is a balanced font.`
- Natural: `This is a natural font.` + `This is an organic font.`
- Weight: `This is a heavy font.` + `This is a compressed font.`
- Flourish: `This is a serif font.`

FontCLIP 논문의 속성 평가 prompt 형식인 `This is a [A] font.`를 따른다.

## 3. 표준화
Grohmann의 회귀계수는 standardized beta이므로,
FontCLIP에서 얻은 다섯 feature score를 A-only reference pool에서 feature별 z-score로 변환한다.

`z_feature = (feature - reference mean) / reference SD`

B는 민감도 reference에만 추가한다.

## 4. Primary mapping: Grohmann Study 3
Study 3을 primary로 정한 이유:
- multiple brands / multiple fonts context
- 36 fonts
- 3,141 complete font ratings
- color를 포함했지만 black이 regression baseline
- 현재 연구의 multi-brand 비교상황과 가장 가까움

### Standardized coefficients
| Dimension | Elaborate | Harmony | Natural | Weight | Flourish |
|---|---:|---:|---:|---:|---:|
| Sincerity | -.02 | .14 | .16 | .02 | .06 |
| Excitement | .12 | .00 | .11 | .01 | .05 |
| Competence | -.05 | .15 | .05 | .05 | .07 |
| Sophistication | .00 | .12 | .16 | -.06 | .10 |
| Ruggedness | .00 | -.03 | -.01 | .17 | .04 |

계산:
`Predicted_Dimension = Σ(Standardized_Beta × z_TypefaceFeature)`

별도 회귀모형은 적합하지 않는다.

## 5. Sensitivity mapping: Study 1
Study 1의 standardized coefficients를 사전에 sensitivity로 고정한다.
이 결과가 더 좋아 보여도 primary로 교체하지 않는다.

## 6. H1 검정
H1:
**동일 브랜드의 텍스트 기반 브랜드 개성과 서체 기반 브랜드 개성 간 거리는
비일치 브랜드 조합의 거리보다 유의하게 작을 것이다.**

### 비교
1. Text 5D와 Typeface-derived 5D의 공통 A 브랜드만 사용
2. 각 모달리티 내부에서 dimension별 z-standardization
3. 5D Euclidean distance 계산
4. matched brand distance의 평균과 mismatched distance의 평균 비교
5. brand labels를 10,000회 permutation
6. one-sided `matched distance < permuted matched distance`

### Retrieval 보조지표
각 브랜드 텍스트가 모든 로고 중 자기 로고를 몇 등으로 검색하는지 계산:
- mean rank
- Top-1
- Top-3
- Top-5

## 7. Text models
결과 사후선택을 피하기 위해 기존 순서를 유지:
- Primary: MiniLM
- Robustness: mDeBERTa

## 8. 한계
FontCLIP score는 Henderson/Grohmann의 인간평정 factor score와 동일하지 않다.
본 단계는 **FontCLIP semantic score를 published typeface constructs의 computational proxy로 사용**한 것이다.
따라서 최종 외적 타당성은 인간평가에서 확인한다.

## References
- Henderson, P. W., Giese, J. L., & Cote, J. A. (2004). Impression Management Using Typeface Design. Journal of Marketing, 68(4), 60–72.
- Grohmann, B., Giese, J. L., & Parkman, I. D. (2013). Using type font characteristics to communicate brand personality of new brands. Journal of Brand Management, 20, 389–403. https://doi.org/10.1057/bm.2012.23
- Tatsukawa et al. (2024). FontCLIP: A Semantic Typography Visual-Language Model for Multilingual Font Applications. Computer Graphics Forum.
