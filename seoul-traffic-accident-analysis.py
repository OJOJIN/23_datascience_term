import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

region_population_mapping = dict()
region_do_population_mapping = dict()

# def
# Read csv files
# cat accident data
accident_df = pd.read_csv("accident.csv", sep=",", encoding="cp949")

# population accident data
population_df = pd.read_csv("population.csv", sep=",", encoding_errors="ignore")

# car accident data
population_region_name_change_df = pd.read_csv(
    "population_region_name_change.csv", sep=",", encoding="cp949"
)

# Region mapping data
region_mapping_df = pd.read_csv("region_mapping.csv", sep=",", encoding="cp949")

# Make ['발생일'] type to datetime
accident_df["발생일"] = pd.to_datetime(accident_df["발생일"])
accident_df["사고건수"] = pd.to_numeric(accident_df["사고건수"])
accident_df["사망자수"] = pd.to_numeric(accident_df["사망자수"])
accident_df["중상자수"] = pd.to_numeric(accident_df["중상자수"])
accident_df["경상자수"] = pd.to_numeric(accident_df["경상자수"])
accident_df["부상신고자수"] = pd.to_numeric(accident_df["부상신고자수"])

population_df.head(10)

# 더티데이터 체크하는 곳

# Check dirty data
print(accident_df.isnull().sum())

# Cleaning dirty data
accident_df.dropna(axis=0, how="any", inplace=True)
accident_df.isnull().sum()

# accident_df의 column별 타입을 지정해줌

# Feature creation (derive New Features from existing ones)
accident_df["발생_연도"] = accident_df["발생일"].dt.year
accident_df["발생_월"] = accident_df["발생일"].dt.month
accident_df["발생_일"] = accident_df["발생일"].dt.day
accident_df["발생_요일"] = accident_df["발생일"].dt.day_name()

accident_df

# 인구 dataframe의 col이름 맞추고 필요없는 col 하나 삭제

# Change columns name
population_col_name = [
    "합계",
    "구",
    "행정동명",
    "2017 1/4",
    "2017 2/4",
    "2017 3/4",
    "2017 4/4",
    "2018 1/4",
    "2018 2/4",
    "2018 3/4",
    "2018 4/4",
    "2019 1/4",
    "2019 2/4",
    "2019 3/4",
    "2019 4/4",
]
population_df.columns = population_col_name

# Drop useless axis
population_df.drop(labels="합계", axis=1, inplace=True)
population_df.drop(labels=0, axis=0, inplace=True)
population_df.drop(labels=1, axis=0, inplace=True)

population_df


# 인구상의 행정동명 csv와 법정동명 <-> 행정동명 일치시키는 csv에서의 행정동명을 일치시켜주는 함수
def population_region_name_fit():
    for i in range(len(population_df)):
        change_population_region_name = population_region_name_change_df.loc[
            population_region_name_change_df["population_region"]
            == population_df.iloc[i]["행정동명"]
        ]["mapping_region"].values
        if len(change_population_region_name) > 0:
            population_df.iloc[i]["행정동명"] = change_population_region_name[0]


population_region_name_fit()
population_df

# 사고 dataframe에 동별, 구별 인구 넣어줄 col 생성
# Derive New Features by Aggregating Over Two Different Entities
accident_df["법정동명_인구"] = np.nan
accident_df["구_인구"] = np.nan

accident_df

# 사고 별 인구 수 넣어주는 부분


# 사고 지역과 날짜 넣으면 해당 지역의 날짜의 인구를 반환해주는 함수
def get_region_population(accident_region_name, accident_year, accident_month):
    # 분기 구하기
    quarter = int((accident_month - 1) / 3 + 1)
    accident_quarter = str(accident_year) + " " + str(quarter) + "/4"

    # 인구수 담을 변수
    popluation = 0
    population_regions = []

    accident_popul_key = accident_region_name + accident_quarter

    # 사고 지역 이름
    if accident_popul_key not in region_population_mapping:
        # 사고 동(법정명)에서 인구 동(행정동명) 뽑기
        population_regions = region_mapping_df.loc[
            region_mapping_df["동리명"] == accident_region_name
        ]["읍면동명"].values

        for region in population_regions:
            # 지역에서 인구수 뽑기
            # if len(population_df.loc[population_df['행정동명'] == region][accident_quarter]) > 0:
            if (
                len(
                    population_df.loc[population_df["행정동명"] == region][accident_quarter]
                )
                > 0
            ):
                popluation += int(
                    population_df.loc[population_df["행정동명"] == region][
                        accident_quarter
                    ].values[0]
                )

        region_population_mapping[accident_popul_key] = popluation
    else:
        popluation = region_population_mapping[accident_popul_key]

    # 지역별 평균 인구도 구할거 여기서
    # region_mapping_df +=

    return popluation


# 사고 지역(도)과 날짜 넣으면 해당 지역의 날짜의 인구를 반환해주는 함수
def get_gu_region_population(accident_do_name, accident_year, accident_month):
    # 분기 구하기
    quarter = int((accident_month - 1) / 3 + 1)
    accident_quarter = str(accident_year) + " " + str(quarter) + "/4"

    accident_popul_key = accident_do_name + accident_quarter

    # 사고 지역 이름
    if accident_popul_key not in region_do_population_mapping:
        popluation = int(
            population_df.loc[
                (population_df["구"] == accident_do_name)
                & (population_df["행정동명"] == "소계")
            ][accident_quarter].values[0]
        )

        region_do_population_mapping[accident_popul_key] = popluation
    else:
        popluation = region_do_population_mapping[accident_popul_key]

    return popluation


# 사고별 해당 법정동명의 사람들 인구 수 col 추가
for i in range(len(accident_df)):
    accident_df.iat[i, 14] = get_region_population(
        accident_df.iat[i, 4], accident_df.iat[i, 10], accident_df.iat[i, 11]
    )
    accident_df.iat[i, 15] = get_gu_region_population(
        accident_df.iat[i, 3], accident_df.iat[i, 10], accident_df.iat[i, 11]
    )


accident_df


# 인구 0인 오류 동 찾음 (항동)
accident_df["법정동명_인구"].replace(0, np.NaN, inplace=True)

# Drop useless axis
accident_df.dropna(axis=0, how="any", inplace=True)

# 각 사고별 위험도 col 추가해줌

accident_df["위험도"] = np.nan

# 모델 돌릴 dataset 만듬
for i in range(len(accident_df)):
    accident_score = (
        (
            accident_df.iat[i, 6] * 10
            + accident_df.iat[i, 7] * 5
            + accident_df.iat[i, 8] * 3
            + accident_df.iat[i, 9] * 1
        )
        * 10000
    ) / accident_df.iat[i, 14]
    accident_df.iat[i, 16] = accident_score

accident_df


# 아웃라이어 처리 안한 클러스터링
# 아웃라이어 처리 안한 클러스터링
# 아웃라이어 처리 안한 클러스터링

# 데이터를 2차원 배열로 변환
X = np.array(accident_df["위험도"].values).reshape(-1, 1)

cluster_num = 4

from sklearn.cluster import KMeans

# KMeans 객체 생성 및 클러스터링 수행
kmeans = KMeans(n_clusters=cluster_num, max_iter=150)  # 클러스터 개수(K) 설정
kmeans.fit(X)

# 클러스터링 결과 확인
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Prepare colors for different clusters
colors = ["blue", "indigo", "lawngreen", "darkorange", "m", "orange"]


unique, count = np.unique(labels, return_counts=True)
uniq_cnt_zip = dict(zip(unique + 1, count))

centroids = centroids.flatten().tolist()
centroids_sorted = sorted(centroids)

risks = ["D", "C", "B", "A"]
risk_index = []

for centroid in centroids:
    risk_index.append(risks[(centroids_sorted.index(centroid))])

risk_cent_zip = dict(zip(risk_index, count))

risk_list = list(risk_cent_zip.keys())
print(risk_list)

cluster_bound = [
    (centroids_sorted[0] + centroids_sorted[1]) / 2,
    (centroids_sorted[1] + centroids_sorted[2]) / 2,
    (centroids_sorted[2] + centroids_sorted[3]) / 2,
]

for cluster in range(cluster_num):
    # find the distance from the center of the cluster to each point
    cluster_points = X[labels == cluster]

    print("Cluster {} : {}".format(str(cluster + 1), cluster_points.size))
    plt.scatter(
        cluster_points,
        np.zeros_like(cluster_points),
        c=colors[cluster],
        s=20,
        label="Risk : " + str(risk_index[cluster]),
    )

# plt.scatter(centroids, np.zeros_like(centroids),c='red',s=20)    # display centroid data in red
plt.title("K-means clustering Result")
plt.xlabel("accident risk")

print("클러스터 레이블:")
print(labels)
print("클러스터 중심:")
print(centroids)

print(risk_cent_zip)

plt.legend()
plt.show()


# A B C D등급 수 그래프로 보여주기

# risk_cent_zip의 값을 키를 기준으로 정렬
sorted_risk_cent_zip = sorted(risk_cent_zip.items(), key=lambda x: x[0], reverse=True)

# 키와 값의 리스트로 분할
labels, values = zip(*sorted_risk_cent_zip)

# 막대 그래프 표시
plt.bar(labels, values, color=colors)

# 막대 위에 레이블 추가
for i, v in enumerate(values):
    plt.text(i, v, str(v), ha="center")

# 그래프에 제목과 축 레이블 추가
plt.title("Accident category count")
plt.xlabel("Risk category")
plt.ylabel("accident count")

# 그래프 표시
plt.show()

# 아웃라이어 처리 한 클러스터링
# 아웃라이어 처리 한 클러스터링
# 아웃라이어 처리 한 클러스터링


# Noise data 처리
quantile_25 = np.quantile(accident_df["위험도"].values, 0.25)
quantile_75 = np.quantile(accident_df["위험도"].values, 0.75)

IQR = quantile_75 - quantile_25

maximum = quantile_75 + 2.0 * IQR

noise_data = []  # store noise_data in this list
noise_del_data = []  # store data without noise(outlier)

# check noise using IQR maximum
for risk in accident_df["위험도"].values:
    if risk > maximum:  # if outlier cotain that point in noise_data
        noise_data.append(risk)
    else:  # else cotain that point in noise_del_data
        noise_del_data.append(risk)

# 데이터를 2차원 배열로 변환
X = np.array(noise_del_data).reshape(-1, 1)

cluster_num = 4

from sklearn.cluster import KMeans

# KMeans 객체 생성 및 클러스터링 수행
kmeans = KMeans(n_clusters=cluster_num, max_iter=150)  # 클러스터 개수(K) 설정
kmeans.fit(X)

# 클러스터링 결과 확인
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# display cluster data with out noise


# Prepare colors for different clusters
colors = ["blue", "indigo", "lawngreen", "darkorange", "m", "orange"]


unique, count = np.unique(labels, return_counts=True)
uniq_cnt_zip = dict(zip(unique + 1, count))

centroids = centroids.flatten().tolist()
centroids_sorted = sorted(centroids)

risks = ["D", "C", "B", "A"]
risk_index = []

for centroid in centroids:
    risk_index.append(risks[(centroids_sorted.index(centroid))])

risk_cent_zip = dict(zip(risk_index, count))


# 너무 컸던 아웃라이어 개수도 위험도 A로 넣어줌
risk_cent_zip["A"] += len(noise_data)


cluster_bound = [
    (centroids_sorted[0] + centroids_sorted[1]) / 2,
    (centroids_sorted[1] + centroids_sorted[2]) / 2,
    (centroids_sorted[2] + centroids_sorted[3]) / 2,
]

for cluster in range(cluster_num):
    # find the distance from the center of the cluster to each point
    cluster_points = X[labels == cluster]

    print("Cluster {} : {}".format(str(cluster + 1), cluster_points.size))
    plt.scatter(
        cluster_points,
        np.zeros_like(cluster_points),
        c=colors[cluster],
        s=20,
        label="Risk : " + str(risk_index[cluster]),
    )

plt.scatter(
    centroids, np.zeros_like(centroids), c="red", s=20
)  # display centroid data in red
plt.title("K-means clustering Result")
plt.xlabel("accident risk")

print("클러스터 레이블:")
print(labels)
print("클러스터 중심:")
print(centroids)

print(risk_cent_zip)

plt.legend()
plt.show()


# A B C D등급 수 그래프로 보여주기

# risk_cent_zip의 값을 키를 기준으로 정렬
sorted_risk_cent_zip = sorted(risk_cent_zip.items(), key=lambda x: x[0], reverse=True)

# 키와 값의 리스트로 분할
labels, values = zip(*sorted_risk_cent_zip)

# 막대 그래프 표시
plt.bar(labels, values, color=colors)

# 막대 위에 레이블 추가
for i, v in enumerate(values):
    plt.text(i, v, str(v), ha="center")

# 그래프에 제목과 축 레이블 추가
plt.title("Risk")
plt.xlabel("Risk category")
plt.ylabel("accident count")

# 그래프 표시
plt.show()

# 사고율을 클러스터된 결과를 바탕으로 A B C D 값을 넣어줌

for i in range(len(accident_df)):
    if accident_df.iat[i, 16] < cluster_bound[0]:
        accident_df.iat[i, 16] = "D"
    elif accident_df.iat[i, 16] < cluster_bound[1]:
        accident_df.iat[i, 16] = "C"
    elif accident_df.iat[i, 16] < cluster_bound[2]:
        accident_df.iat[i, 16] = "B"
    else:
        accident_df.iat[i, 16] = "A"

# decision tree 돌릴 데이터셋 생성

# Create decision_dataset
decision_df = accident_df.loc[:, ["법정동명", "발생시간", "발생_월", "발생_요일", "위험도"]]
decision_df.head()

# 법정동명이랑 발생 요일 labelEncoder를 통해 변환

from sklearn.preprocessing import LabelEncoder

# Convert numberic data to Categorical data
region_label = LabelEncoder()
day_label = LabelEncoder()

region_label.fit(decision_df["법정동명"])
day_label.fit(decision_df["발생_요일"])

decision_df["법정동명"] = region_label.transform(decision_df["법정동명"])
decision_df["발생_요일"] = day_label.transform(decision_df["발생_요일"])

decision_df = decision_df.astype("category")


print(decision_df)

# Find best param
# 가장 좋은 RandomForest에 들어갈 Parameter를 찾음

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


X = decision_df.iloc[:, :4]  # feature
y = decision_df.iloc[:, -1]  # Target variable

params = {
    "n_estimators": [10],
    "max_depth": [7, 10, 13, 14],
    "criterion": ["gini", "entropy"],
    "bootstrap": [True, False],
}

# search best param
rf_clf = RandomForestClassifier(random_state=0, n_jobs=-1)
grid_cv = GridSearchCV(rf_clf, param_grid=params, cv=5, n_jobs=-1)
grid_cv.fit(X, y)

print("best hyper parameter: ", grid_cv.best_params_)
print("best score: {:.4f}".format(grid_cv.best_score_))

# 랜덤 포레스트 돌려보는데 k fold cross validation을 통해 테스팅 했는데
# 일반 decision tree 보다 점수가 안나옴

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

random_model = RandomForestClassifier(
    n_estimators=10, bootstrap=True, criterion="gini", random_state=0
)
random_model.fit(X, y)

kfold = KFold(n_splits=10, shuffle=True, random_state=0)

# Evaluate base model using cross-validation
base_scores = cross_val_score(random_model, X, y, cv=kfold)
base_avg_score = np.mean(base_scores)

print("\n========== Base Model Cross-Validation Scores ==========\n", base_scores)
print("\n========== Base Model Average Score ==========\n", base_avg_score)
print(
    "\n========== Base Model Cross-Validation best Score ==========\n",
    base_scores.max(),
)

# 그래서 decision tree를 돌림
# decision tree를 사용하는게 k-fold 점수 더 잘나와서 그걸로 사용할 예정


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

X = decision_df.iloc[:, :4]  # feature
y = decision_df.iloc[:, -1]  # Target variable

# Base DesicionTree model(depth = 14)
base_model = DecisionTreeClassifier(max_depth=14, min_samples_split=7)
base_model.fit(X, y)
kfold = KFold(n_splits=10, shuffle=True, random_state=0)

# Evaluate base model using cross-validation
base_scores = cross_val_score(base_model, X, y, cv=kfold)
base_avg_score = np.mean(base_scores)

print("\n========== Base Model Cross-Validation Scores ==========\n", base_scores)
print("\n========== Base Model Average Score ==========\n", base_avg_score)
print(
    "\n========== Base Model Cross-Validation best Score ==========\n",
    base_scores.max(),
)

# 예측할 동(지역), 월, 시간, 요일을 넣으면 base model 에 들어갈 X 값을 반환해줌


def make_test_df(region, month, time, day):
    # test_df = pd.Dataframe(columns = ['법정동명', '발생시간', '발생_월', '발생_요일'])
    test_df = {"법정동명": region, "발생시간": time, "발생_월": month, "발생_요일": day}

    data = pd.DataFrame(test_df, index=[0])

    data["법정동명"] = region_label.transform(data["법정동명"])
    data["발생_요일"] = day_label.transform(data["발생_요일"])

    return data


for i in range(24):
    print(
        "6월 장지동/ 수요일/ {}시/ 위험도 : {}".format(
            i, base_model.predict(make_test_df("장지동", 6, i, "Wednesday"))[0]
        )
    )


# 모든 동명 확인
regions = region_mapping_df.drop_duplicates("동리명")["동리명"].values
# 시간 0~23 List
hour_list = ["hour_" + str(i) for i in range(24)]
# 요일 List
day_list = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

# 월 List
month_list = ["month_" + str(i) for i in range(1, 13)]

total_list = hour_list + day_list + month_list
model_df = pd.DataFrame(columns=total_list, index=regions)
model_df.fillna(0, inplace=True)

# 모델 돌릴 dataset 만듬
for i in range(len(accident_df)):
    # accident_score = accident_df.iat[i, 6] * 10 + accident_df.iat[i, 7] * 5 + accident_df.iat[i, 8] * 3 + accident_df.iat[i, 9] * 1
    region_accident_score = (
        (
            accident_df.iat[i, 6] * 10
            + accident_df.iat[i, 7] * 5
            + accident_df.iat[i, 8] * 3
            + accident_df.iat[i, 9] * 1
        )
        * 100
    ) / accident_df.iat[i, 14]
    # 동별 위험도 넣어주기( 각 위험도 스케일링을 위해 주 * 7, 시간 * 24, 월별 * 12 ㄱㅊㄱㅊ)
    model_df.at[accident_df.iat[i, 4], accident_df.iat[i, 13]] += (
        region_accident_score * 7
    )
    model_df.at[accident_df.iat[i, 4], "hour_" + str(accident_df.iat[i, 1])] += (
        region_accident_score * 24
    )
    model_df.at[accident_df.iat[i, 4], "month_" + str(accident_df.iat[i, 11])] += (
        region_accident_score * 12
    )

    gu_accident_score = (
        (
            accident_df.iat[i, 6] * 10
            + accident_df.iat[i, 7] * 5
            + accident_df.iat[i, 8] * 3
            + accident_df.iat[i, 9] * 1
        )
        * 100
    ) / accident_df.iat[i, 15]
    # 도별 위험도 넣어주기
    model_df.at[accident_df.iat[i, 3], accident_df.iat[i, 13]] += gu_accident_score * 7
    model_df.at[accident_df.iat[i, 3], "hour_" + str(accident_df.iat[i, 1])] += (
        gu_accident_score * 24
    )
    model_df.at[accident_df.iat[i, 3], "month_" + str(accident_df.iat[i, 11])] += (
        gu_accident_score * 12
    )

print(model_df)

# 구 정보담은 dataframe
model_gu = model_df.iloc[464:, :]
print(model_gu)

# 그래프 찍기

import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 준비

x = [
    "Gangnam-gu",
    "Gangdong-gu",
    "Gangbuk-gu",
    "Gangseo-gu",
    "Gwanak-gu",
    "Gwangjin-gu",
    "Guro-gu",
    "Geumcheon-gu",
    "Nowon-gu",
    "Dobong-gu",
    "Dongdaemun-gu",
    "Dongjak-gu",
    "Mapo-gu",
    "Seodaemun-gu",
    "Seocho-gu",
    "Seongdong-gu",
    "Seongbuk-gu",
    "Songpa-gu",
    "Yangcheon-gu",
    "Yeongdeungpo-gu",
    "Yongsan-gu",
    "Eunpyeong-gu",
    "Jongno-gu",
    "Jung-gu",
    "Jungnang-gu",
]

"""
x = ['강남구', '강동구', '강북구', '강서구', '관악구', '광진구', '구로구', '금천구', '노원구', '도봉구', '동대문구', '동작구', '마포구',
     '서대문구', '서초구', '성동구', '성북구', '송파구', '양천구', '영등포구', '용산구', '은평구', '종로구', '중구', '중랑구']
"""

day = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

for i in range(24):
    y = model_gu.iloc[:, i]
    plt.figure(figsize=(10, 4))

    colors = sns.color_palette("hls", len(x))

    # 막대 그래프 그리기
    plt.bar(x, y, color=colors)

    # 그래프 제목과 축 레이블 설정
    plt.xticks(rotation=70)
    plt.title(f"Risk of hour_{str(i)}")
    plt.xlabel("Gu")
    plt.ylabel("Risk")


for i in range(7):
    y = model_gu.iloc[:, i + 24]
    plt.figure(figsize=(10, 4))

    colors = sns.color_palette("hls", len(x))

    # 막대 그래프 그리기
    plt.bar(x, y, color=colors)

    # 그래프 제목과 축 레이블 설정
    plt.xticks(rotation=70)
    plt.title("Risk of  {}".format(day[i]))
    plt.xlabel("Gu")
    plt.ylabel("Risk")

for i in range(12):
    y = model_gu.iloc[:, i + 31]
    plt.figure(figsize=(10, 4))

    colors = sns.color_palette("hls", len(x))

    # 막대 그래프 그리기
    plt.bar(x, y, color=colors)

    # 그래프 제목과 축 레이블 설정
    plt.xticks(rotation=70)
    plt.title(f"Risk of month_{str(i+1)}")
    plt.xlabel("Gu")
    plt.ylabel("Risk")


# 그래프 출력
plt.show()
