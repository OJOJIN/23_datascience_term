# seoul-traffic-accidents-analysis

# Contributor
김세중 노현지 양다인 오진영

# Architecture
<img width="896" alt="a" src="https://github.com/OJOJIN/seoul-traffic-accidents-analysis/assets/99639919/0f9696ff-5fa1-4a5e-b4d6-5c27dbd449c9">


# End to End process

## 1. Business objective

<img width="163" alt="image" src="https://github.com/OJOJIN/seoul-traffic-accidents-analysis/assets/86943499/f51574ae-4167-460d-9e36-d48d0975e0b6">
<img width="305" alt="image" src="https://github.com/OJOJIN/seoul-traffic-accidents-analysis/assets/86943499/656c2527-3b09-4525-a484-110e1c4a8cd3">


The risk of traffic accident deaths in Korea is decreasing year by year.

However, Korea still has a high risk, ranking 27th in the number of traffic accident deaths compared to the population among 36 OECD countries.

Therefore, our team aimed to reduce the risk of traffic accidents by creating a model that predicts the risk of traffic accidents in certain areas of Seoul to help people go out at safer times and to strengthen crackdowns and guide them to be alert during high-risk times

## 2. Data Inspection
Accident.csv
<img width="896" alt="a" src="https://github.com/OJOJIN/seoul-traffic-accidents-analysis/assets/82256962/f7b8be1a-5ebe-4ad7-94de-05f2d3ca3d0e">
>114442 rows x 10 columns
Columns
- 발생일
- 발생시간
- 발생지_시도
- 발생지_시군구
- 법정동명
- 사고건수
- 사망자수
- 중상자수
- 경상자수
- 부상신고자수

Population.csv
<img width="896" alt="a" src="https://github.com/OJOJIN/seoul-traffic-accidents-analysis/assets/82256962/9bdd9c1d-d51e-422b-802c-afffbd058e2d">
>450 rows x 14 columns
Columns
- 동별(1)
- 동별(2)
- 동별(3)
- 2017 1/4
- 2017 2/4
- 2017 3/4
- 2017 4/4
- 2018 1/4
- 2018 2/4
- 2018 3/4
- 2018 4/4
- 2019 1/4
- 2019 2/4
- 2019 3/4
- 2019 4/4

Region_mapping.csv
<img width="896" alt="a" src="https://github.com/OJOJIN/seoul-traffic-accidents-analysis/assets/82256962/e036e315-45ba-431c-af1c-8c0b22f33d6c">
>765 rows x 7 columns
Columns
- 행정동코드
- 시도명
- 시군구명
- 읍면동명
- 법정동코드
- 동리명
- 생성일자
<img width="896" alt="a" src="https://github.com/OJOJIN/seoul-traffic-accidents-analysis/assets/82256962/69627daa-f5b9-4fc1-8132-80cfb4c0311b">
Population_region_name_change.csv

>180 rows x 2 columns
- population_regin
- mapping_region
