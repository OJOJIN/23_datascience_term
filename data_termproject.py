import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

region_population_mapping = dict()

# 사고 지역과 날짜 넣으면 해당 지역의 날짜의 인구를 반환해주는 함수
def get_region_population(accident_region_name, accident_year, accident_month):
    # print(accident_region_name, accident_year, accident_month)
    # 분기 구하기
    quarter = int((accident_month - 1) / 3 + 1)
    accident_quarter = str(accident_year)+ ' ' + str(quarter) + '/4'

    # 인구수 담을 변수
    popluation = 0
    population_regions = []

    accident_popul_key = accident_region_name + accident_quarter

    # 사고 지역 이름
    if accident_popul_key not in region_population_mapping:
        # 사고 동(법정명)에서 인구 동(행정동명) 뽑기
        # population_regions = region_mapping_df.loc[region_mapping_df['동리명'] == accident_popul_key]['읍면동명'].values
        population_regions = region_mapping_df.loc[region_mapping_df['동리명'] == accident_region_name]['읍면동명'].values
          
        for region in population_regions:
            # 지역에서 인구수 뽑기
            # if len(population_df.loc[population_df['행정동명'] == region][accident_quarter]) > 0:
            if len(population_df.loc[population_df['행정동명'] == region][accident_quarter]) > 0:
                popluation += int(population_df.loc[population_df['행정동명'] == region][accident_quarter].values[0])
    
        region_population_mapping[accident_popul_key] = popluation
    else:
        popluation = region_population_mapping[accident_popul_key]

    # 지역별 평균 인구도 구할거 여기서
    # region_mapping_df +=
        
    return popluation


# 인구 행정동명과 행정동명 <-> 법정동명 매칭 테이블상의 행정동명과 다른 경우를 맞춰주기 위한 preprocessing
def population_region_name_fit():
  for i in range(len(population_df)):
      # print(population_df.iloc[i]['행정동명'])
      change_population_region_name = population_region_name_change_df.loc[population_region_name_change_df['population_region'] == population_df.iloc[i]['행정동명']]['mapping_region'].values
      if len(change_population_region_name) > 0:
          population_df.iloc[i]['행정동명'] = change_population_region_name[0]


# def 
# Read csv files
# cat accident data
accident_df = pd.read_csv("accident.csv", sep=",", encoding="cp949")

# population accident data
population_df = pd.read_csv("population.csv", sep=",", encoding_errors="ignore")

# car accident data
population_region_name_change_df = pd.read_csv("population_region_name_change.csv", sep=",", encoding="cp949")

# Region mapping data
region_mapping_df = pd.read_csv("region_mapping.csv", sep=",", encoding="cp949")



# Data exploration (including dataset description)
# Data exploration (including dataset description)
# Data exploration (including dataset description)

# Print data
print("\n====================== accident data ======================\n")
print(accident_df.head())
print(accident_df.describe())

# Make ['발생일'] type to datetime
accident_df['발생일'] = pd.to_datetime(accident_df['발생일'])
accident_df['사고건수'] = pd.to_numeric(accident_df['사고건수'])
accident_df['사망자수'] = pd.to_numeric(accident_df['사망자수'])
accident_df['중상자수'] = pd.to_numeric(accident_df['중상자수'])
accident_df['경상자수'] = pd.to_numeric(accident_df['경상자수'])
accident_df['부상신고자수'] = pd.to_numeric(accident_df['부상신고자수'])

print("\n====================== population data ======================\n")
print(population_df.head())
print(population_df.describe())

print("\n====================== population_region_name_change data ======================\n")
print(population_region_name_change_df.head())
print(population_region_name_change_df.describe())

print("\n====================== population_region_name_change data ======================\n")
print(region_mapping_df.head())
print(region_mapping_df.describe())

#  Data preprocessing
#  Data preprocessing
#  Data preprocessing

# Feature engineering
# Feature engineering


# Cleaning dirty data(빈 값 4개있었음)
print(len(accident_df))
accident_df.dropna(axis=0, how='any', inplace=True)
print(len(accident_df))

# Change columns name
population_col_name = ['합계', '구', '행정동명','2017 1/4', '2017 2/4', '2017 3/4', '2017 4/4', '2018 1/4', '2018 2/4', '2018 3/4', '2018 4/4', '2019 1/4', '2019 2/4', '2019 3/4', '2019 4/4']
population_df.columns = population_col_name

# Drop useless axis
population_df.drop(labels="합계", axis=1, inplace=True)
population_df.drop(labels=0, axis=0, inplace=True)
population_df.drop(labels=1, axis=0, inplace=True)


# Feature creation (derive New Features from existing ones)
# Feature creation (derive New Features from existing ones)

accident_df['발생_연도'] = accident_df['발생일'].dt.year
accident_df['발생_월'] = accident_df['발생일'].dt.month
accident_df['발생_일'] = accident_df['발생일'].dt.day
accident_df['발생_요일'] = accident_df['발생일'].dt.day_name()


# 인구 행정동명 검색되게 맞춰주기
population_region_name_fit()

# Derive New Features by Aggregating Over Two Different Entities

accident_df['법정동명_인구'] = np.nan

print(accident_df.loc[(accident_df['발생일'] == '2019-12-31') & (accident_df['법정동명'] == '구로동')]['사고건수'])


regions = region_mapping_df.drop_duplicates('동리명')['동리명'].values

# 시간 0~23 List
hour_list = ['hour_' + str(i) for i in range(24)]
# 요일 List
day_list = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# 월 List
month_list = ['month_' + str(i) for i in range(1, 13) ]

# 시간 요일 월 리스트 합친 리스트 추가
total_list = hour_list + day_list + month_list

# 모델 돌릴 dataframe
model_df = pd.DataFrame(columns = total_list, index=regions)
model_df.fillna(0, inplace=True)
print(model_df)

print(accident_df.head(10))
# 사고별 해당 법정동명의 사람들 인구 수 col 추가
for i in range(len(accident_df)):
    # print(accident_df.loc[i])
    accident_df.iat[i, 14] = get_region_population(accident_df.iat[i, 4], accident_df.iat[i, 10], accident_df.iat[i, 11])

# 인구 0인 오류 동 찾음 (항동)
accident_df['법정동명_인구'].replace(0, np.NaN, inplace=True)

# Drop useless axis
accident_df.dropna(axis=0, how='any',inplace=True)


# 모델 돌릴 dataset 만듬
for i in range(len(accident_df)):
    # accident_score = accident_df.iat[i, 6] * 10 + accident_df.iat[i, 7] * 5 + accident_df.iat[i, 8] * 3 + accident_df.iat[i, 9] * 1
    accident_score = ((accident_df.iat[i, 6] * 10 + accident_df.iat[i, 7] * 5 + accident_df.iat[i, 8] * 3 + accident_df.iat[i, 9] * 1) * 1000) / accident_df.iat[i, 14]
    model_df.at[accident_df.iat[i, 4], accident_df.iat[i, 13]] += accident_score
    model_df.at[accident_df.iat[i, 4], 'hour_' + str(accident_df.iat[i, 1])] += accident_score
    model_df.at[accident_df.iat[i, 4], 'month_' + str(accident_df.iat[i, 11])] += accident_score

print(model_df.head(100))