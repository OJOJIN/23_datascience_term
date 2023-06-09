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
### Accident.csv

<img width="896" alt="a" src="https://github.com/OJOJIN/seoul-traffic-accidents-analysis/assets/82256962/f7b8be1a-5ebe-4ad7-94de-05f2d3ca3d0e">

> 114442 rows x 10
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

### Population.csv

<img width="896" alt="a" src="https://github.com/OJOJIN/seoul-traffic-accidents-analysis/assets/82256962/9bdd9c1d-d51e-422b-802c-afffbd058e2d">

> 450 rows x 14
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

### Region_mapping.csv

<img width="a" alt="a" src="https://github.com/OJOJIN/seoul-traffic-accidents-analysis/assets/82256962/e036e315-45ba-431c-af1c-8c0b22f33d6c">

> 765 rows x 7
Columns
- 행정동코드
- 시도명
- 시군구명
- 읍면동명
- 법정동코드
- 동리명
- 생성일자

### Population_region_name_change.csv

<img width="a" alt="a" src="https://github.com/OJOJIN/seoul-traffic-accidents-analysis/assets/82256962/69627daa-f5b9-4fc1-8132-80cfb4c0311b">

> 180 rows x 2
- population_regin
- mapping_region



## 3. Data Preprocessing
- Cleaning dirty data
- Feature creation derived from '발생일'
- Derive New '인구' features by aggregatingg over two entities
- Featrue creation '위험도' columns in accident_df
- '법정동명', '발생_요일' with LabelEncoder



## 4. Clustering & evaluation
- Create a new feature related to multiple columns using k-means clustering.
- Perform outliner detection to turn the unbalanced distribution of categories A through D into a balanced one.

## 5. Random Forest modeling & evaluation
- Create a predictive model using RandomForest Algorithm.
- Use GridSearchCV to find the most appropriate hyperparameters.
- Using k-fold croos validation for testing. The average score was 57%.



## 6. Decision Tree classification & eveluation
- The model was trained based on the clustering of the target features.
- Create a predictive model using DesicionTree Classification.
- Using k-fold croos validation for testing. The average score was 65%.



# External factors impact
The Road Traffic Authority(도로교통공단) has reported that traffic fatality rates are three to four times higher on foggy days.
However, our team's data analysis was inaccurate because it only considered specific dates and locations. It didn't take into account the impact of weather conditions on traffic accidents.
It only relied on inputs such as time, day, and month to predict the risk level. As weather conditions play an important role in the incidence of traffic accidents, it is necessary to consider external factors as well.


# What you have learned doing the project
- During the data modeling and analysis process, the team members gained valuable insights and experiences. We recognized the importance of finding optimal parameters for accurate modeling and the challenges involved in achieving this.
- Preprocessing and analyzing diverse datasets were seen as novel and rewarding tasks, highlighting the significance of data exploration and objective review.
- We learned the difficulty of data preprocessing, especially in matching values between different files, which plays an important role in achieving better results in data modeling.
- The team's collective experience has taught them the importance of parameter optimization, thorough data exploration, and effective preprocessing techniques in data science.
