
import math

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


DATA_URL = (
    "https://raw.githubusercontent.com/"
    "it-teacher-go/data-share-/main/kobis_movies.csv"
)

APP_TITLE = "🎬 영화 유형 나누기"

st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide",
)

FEATURE_LABELS = {
    "log_first_scrn": "개봉 초기 스크린 수 (로그)",
    "log_total_audi": "누적 관객 수 (로그)",
    "days_in_top10": "10위권 일수",
    "long_run_index": "롱런 지수",
}

CIRCLE_LABELS = [
    "㉮", "㉯", "㉰", "㉱", "㉲", "㉳", "㉴"
]


@st.cache_data(show_spinner="영화 데이터를 불러오는 중입니다...")
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    required = [
        "movieCd",
        "movieNm",
        "openDt",
        "genre",
        "nation",
        "first_scrn",
        "first_show",
        "first_date",
        "peak",
        "first_week_audi",
        "total_audi",
        "days_in_top10",
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "데이터에 필요한 열이 없습니다: "
            + ", ".join(missing)
        )

    return df


def prepare_data(raw):
    df = raw.copy()

    numeric_cols = [
        "first_scrn",
        "first_week_audi",
        "total_audi",
        "days_in_top10",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    # 롱런 지수 = 누적 관객 수 / 첫 주 관객 수
    df["long_run_index"] = (
        df["total_audi"]
        / df["first_week_audi"].where(
            df["first_week_audi"] > 0
        )
    )

    # 롱런 지수가 20을 넘으면 20으로 제한
    df["long_run_index"] = (
        df["long_run_index"].clip(upper=20)
    )

    # 스크린 수와 누적 관객 수에 상용로그 적용
    df["log_first_scrn"] = (
        df["first_scrn"]
        .where(df["first_scrn"] > 0)
        .map(
            lambda x: (
                math.log10(x)
                if pd.notna(x)
                else float("nan")
            )
        )
    )

    df["log_total_audi"] = (
        df["total_audi"]
        .where(df["total_audi"] > 0)
        .map(
            lambda x: (
                math.log10(x)
                if pd.notna(x)
                else float("nan")
            )
        )
    )

    feature_cols = list(FEATURE_LABELS.keys())

    usable = df.dropna(
        subset=feature_cols + ["movieNm"]
    ).copy()

    # 필수 값이 없거나 0 이하인 영화 제외
    usable = usable[
        (usable["first_scrn"] > 0)
        & (usable["total_audi"] > 0)
        & (usable["first_week_audi"] > 0)
    ]

    return df, usable


def fit_clusters(df, features, k):
    scaler = StandardScaler()

    X = scaler.fit_transform(df[features])

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
    )

    labels = model.fit_predict(X)

    out = df.copy()
    out["_cluster_raw"] = labels

    # 평균 누적 관객이 많은 군집부터 ㉮, ㉯, ㉰ 순서로 표시
    order = (
        out.groupby("_cluster_raw")["total_audi"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    name_map = {
        raw_label: CIRCLE_LABELS[i]
        for i, raw_label in enumerate(order)
    }

    out["군집"] = out["_cluster_raw"].map(name_map)

    return out, X


st.title(APP_TITLE)

st.caption(
    "정답 범주 없이 흥행 기록의 유사성을 바탕으로 "
    "영화를 묶는 비지도학습 실습입니다."
)

with st.expander("이 앱의 분석 방법", expanded=False):
    st.markdown(
        """
        - **사용 속성:** 개봉 초기 스크린 수,
          누적 관객 수, 10위권 일수, 롱런 지수
        - **로그 변환:** 스크린 수와 누적 관객 수에
          상용로그를 적용합니다.
        - **롱런 지수:** 누적 관객 수를 첫 주 관객 수로
          나눈 값이며, 20을 넘으면 20으로 제한합니다.
        - **표준화:** 선택한 속성을 평균 0, 표준편차 1의
          척도로 바꾼 뒤 K-means를 실행합니다.
        - **해석 주의:** 군집은 작품성이나 고정된 등급이
          아니라, 선택한 흥행 기록에서 나타난 유사성입니다.
        """
    )


# 데이터 불러오기
try:
    raw_data = load_data()
    all_data, data = prepare_data(raw_data)

except Exception as exc:
    st.error("데이터를 불러오거나 처리하지 못했습니다.")
    st.exception(exc)

    st.info(
        "인터넷 연결과 데이터 URL을 확인한 뒤 "
        "앱을 다시 실행해 주세요."
    )

    st.stop()


# 데이터 요약
c1, c2, c3 = st.columns(3)

c1.metric(
    "원본 데이터 편수",
    f"{len(all_data):,}편",
)

c2.metric(
    "분석 가능한 편수",
    f"{len(data):,}편",
)

c3.metric(
    "분석에서 제외된 편수",
    f"{len(all_data) - len(data):,}편",
)

st.caption(
    "분석 가능 편수는 필수 속성이 있고, 첫 주 관객·"
    "스크린 수·누적 관객이 양수인 영화의 수입니다."
)


# 1. 군집 설정
st.subheader("1. 군집 설정")

left, right = st.columns([2, 1])

with left:
    selected_features = st.multiselect(
        "군집을 만드는 속성 선택 (최소 2개)",
        options=list(FEATURE_LABELS.keys()),
        default=list(FEATURE_LABELS.keys()),
        format_func=lambda x: FEATURE_LABELS[x],
    )

with right:
    k = st.selectbox(
        "군집 수 k",
        options=[2, 3, 4, 5, 6, 7],
        index=1,
    )


if len(selected_features) < 2:
    st.warning(
        "군집을 만들 속성을 두 개 이상 선택해 주세요."
    )
    st.stop()

if len(data) <= k:
    st.error(
        f"분석 가능한 영화가 {len(data)}편이므로 "
        f"k={k}로 군집을 만들 수 없습니다."
    )
    st.stop()


clustered, X_scaled = fit_clusters(
    data,
    selected_features,
    k,
)

display_order = [
    label
    for label in CIRCLE_LABELS
    if label in clustered["군집"].unique()
]

st.success(
    f"전체 {len(all_data):,}편 중 "
    f"{len(clustered):,}편을 분석했습니다. "
    f"선택한 속성 {len(selected_features)}개로 "
    f"{k}개 군집을 만들었습니다."
)


# 2. 2차원 산점도
st.subheader("2. 2차원 산점도")

axis1, axis2 = st.columns(2)

with axis1:
    x_feature = st.selectbox(
        "가로축",
        options=selected_features,
        index=0,
        format_func=lambda x: FEATURE_LABELS[x],
        key="x_feature",
    )

with axis2:
    y_default = min(1, len(selected_features) - 1)

    y_feature = st.selectbox(
        "세로축",
        options=selected_features,
        index=y_default,
        format_func=lambda x: FEATURE_LABELS[x],
        key="y_feature",
    )


fig2 = px.scatter(
    clustered,
    x=x_feature,
    y=y_feature,
    color="군집",
    category_orders={"군집": display_order},
    hover_name="movieNm",
    hover_data={
        "first_scrn": ":,.0f",
        "total_audi": ":,.0f",
        "days_in_top10": ":,.0f",
        "long_run_index": ":.2f",
        "_cluster_raw": False,
        "군집": True,
    },
    labels={
        x_feature: FEATURE_LABELS[x_feature],
        y_feature: FEATURE_LABELS[y_feature],
    },
    title="영화별 흥행 기록과 군집",
)

fig2.update_traces(
    marker={
        "size": 7,
        "opacity": 0.75,
    }
)

fig2.update_layout(
    legend_title_text="군집",
    height=560,
)

st.plotly_chart(
    fig2,
    use_container_width=True,
)


# 3. 3차원 산점도
st.subheader("3. 3차원 산점도")

if len(selected_features) < 3:
    st.info(
        "3차원 산점도를 보려면 군집 속성을 "
        "세 개 이상 선택해 주세요."
    )

else:
    xyz = st.columns(3)

    with xyz[0]:
        z_x = st.selectbox(
            "3D X축",
            selected_features,
            index=0,
            format_func=lambda x: FEATURE_LABELS[x],
            key="z_x",
        )

    with xyz[1]:
        z_y = st.selectbox(
            "3D Y축",
            selected_features,
            index=1,
            format_func=lambda x: FEATURE_LABELS[x],
            key="z_y",
        )

    with xyz[2]:
        z_z = st.selectbox(
            "3D Z축",
            selected_features,
            index=2,
            format_func=lambda x: FEATURE_LABELS[x],
            key="z_z",
        )

    fig3 = px.scatter_3d(
        clustered,
        x=z_x,
        y=z_y,
        z=z_z,
        color="군집",
        category_orders={"군집": display_order},
        hover_name="movieNm",
        hover_data={
            "total_audi": ":,.0f",
            "first_scrn": ":,.0f",
            "days_in_top10": ":,.0f",
        },
        labels={
            z_x: FEATURE_LABELS[z_x],
            z_y: FEATURE_LABELS[z_y],
            z_z: FEATURE_LABELS[z_z],
        },
        title="3차원 영화 군집",
    )

    fig3.update_traces(
        marker={
            "size": 3,
            "opacity": 0.8,
        }
    )

    fig3.update_layout(
        height=650,
        legend_title_text="군집",
    )

    st.plotly_chart(
        fig3,
        use_container_width=True,
    )


# 4. 군집별 요약표
st.subheader("4. 군집별 요약")

summary = (
    clustered.groupby("군집")
    .agg(
        편수=("movieNm", "size"),
        스크린_수_평균=("first_scrn", "mean"),
        누적_관객_평균=("total_audi", "mean"),
        days_in_top10_avg=("days_in_top10", "mean"),
        롱런_지수_평균=("long_run_index", "mean"),
    )
    .reindex(display_order)
    .reset_index()
)

summary = summary.rename(
    columns={
        "군집": "군집",
        "스크린_수_평균": "스크린 수 평균",
        "누적_관객_평균": "누적 관객 평균",
        "days_in_top10_avg": "10위권 일수 평균",
        "롱런_지수_평균": "롱런 지수 평균",
    }
)

summary["스크린 수 평균"] = (
    summary["스크린 수 평균"].round(1)
)

summary["누적 관객 평균"] = (
    summary["누적 관객 평균"].round(0)
)

summary["10위권 일수 평균"] = (
    summary["10위권 일수 평균"].round(1)
)

summary["롱런 지수 평균"] = (
    summary["롱런 지수 평균"].round(2)
)

st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True,
)


# 5. 군집별 관객 수 상위 영화
st.subheader("5. 군집별 누적 관객 상위 영화 5편")

for cluster_name in display_order:
    group = (
        clustered[clustered["군집"] == cluster_name]
        .sort_values("total_audi", ascending=False)
        .head(5)
    )

    cluster_count = len(
        clustered[clustered["군집"] == cluster_name]
    )

    with st.expander(
        f"{cluster_name} 군집 — {cluster_count:,}편",
        expanded=True,
    ):
        show_cols = [
            col
            for col in [
                "movieNm",
                "openDt",
                "genre",
                "nation",
                "first_scrn",
                "total_audi",
                "days_in_top10",
                "long_run_index",
            ]
            if col in group.columns
        ]

        top = group[show_cols].rename(
            columns={
                "movieNm": "영화 제목",
                "openDt": "개봉일",
                "genre": "장르",
                "nation": "국가",
                "first_scrn": "개봉 초기 스크린 수",
                "total_audi": "누적 관객",
                "days_in_top10": "10위권 일수",
                "long_run_index": "롱런 지수",
            }
        ).copy()

        if "누적 관객" in top.columns:
            top["누적 관객"] = top["누적 관객"].map(
                lambda x: f"{x:,.0f}"
            )

        if "개봉 초기 스크린 수" in top.columns:
            top["개봉 초기 스크린 수"] = (
                top["개봉 초기 스크린 수"].map(
                    lambda x: f"{x:,.0f}"
                )
            )

        if "롱런 지수" in top.columns:
            top["롱런 지수"] = top["롱런 지수"].map(
                lambda x: f"{x:.2f}"
            )

        st.dataframe(
            top,
            use_container_width=True,
            hide_index=True,
        )


# 6. 엘보우 기법
st.subheader("6. 엘보우 기법 — 적절한 군집 수 찾기")

st.write(
    "현재 선택한 속성으로 k=1부터 7까지 군집 내 "
    "제곱합을 계산합니다. 값이 크게 줄다가 "
    "감소폭이 작아지는 지점을 살펴보세요."
)

scaler_elbow = StandardScaler()

X_elbow = scaler_elbow.fit_transform(
    data[selected_features]
)

max_k = min(7, len(data) - 1)

inertias = []

for candidate_k in range(1, max_k + 1):
    model = KMeans(
        n_clusters=candidate_k,
        random_state=42,
        n_init=10,
    )

    model.fit(X_elbow)
    inertias.append(float(model.inertia_))


elbow_df = pd.DataFrame(
    {
        "군집 수 k": list(range(1, max_k + 1)),
        "거리 제곱의 합": inertias,
    }
)

elbow_fig = go.Figure()

elbow_fig.add_trace(
    go.Scatter(
        x=elbow_df["군집 수 k"],
        y=elbow_df["거리 제곱의 합"],
        mode="lines+markers",
        name="거리 제곱의 합",
    )
)

elbow_fig.add_vline(
    x=k,
    line_dash="dash",
    annotation_text=f"현재 k={k}",
)

elbow_fig.update_layout(
    xaxis_title="군집 수 k",
    yaxis_title="군집 내 거리 제곱의 합",
    height=430,
)

st.plotly_chart(
    elbow_fig,
    use_container_width=True,
)


# 감소량은 이전 k의 값 - 현재 k의 값으로 계산
elbow_df["앞보다 줄어든 값"] = (
    elbow_df["거리 제곱의 합"].shift(1)
    - elbow_df["거리 제곱의 합"]
)

elbow_df.loc[0, "앞보다 줄어든 값"] = None

elbow_display = elbow_df.copy()

elbow_display["거리 제곱의 합"] = (
    elbow_display["거리 제곱의 합"].round(2)
)

elbow_display["앞보다 줄어든 값"] = (
    elbow_display["앞보다 줄어든 값"].round(2)
)

st.dataframe(
    elbow_display,
    use_container_width=True,
    hide_index=True,
)


# 7. 실루엣 점수
if (
    1 < k < len(data)
    and clustered["_cluster_raw"].nunique() > 1
):
    try:
        score = silhouette_score(
            X_scaled,
            clustered["_cluster_raw"],
        )

        st.metric(
            "현재 k의 실루엣 점수",
            f"{score:.3f}",
        )

        st.caption(
            "실루엣 점수는 -1~1 범위이며 1에 가까울수록 "
            "군집이 서로 잘 분리된 경향을 뜻합니다. "
            "점수만으로 정답 k가 자동 결정되지는 않습니다."
        )

    except Exception:
        st.info(
            "현재 데이터에서는 실루엣 점수를 "
            "계산할 수 없습니다."
        )


# 8. 결과 해석
st.subheader("7. 분석 결과 해석")

st.markdown(
    """
    아래 내용을 직접 확인해 보고서에 작성하세요.

    1. 평균 누적 관객이 가장 큰 군집은 어느 군집인가요?
    2. 스크린 수 평균과 10위권 일수 평균이 큰 군집은
       각각 어느 군집인가요?
    3. 엘보우 그래프에서 감소폭이 눈에 띄게 작아지는
       지점은 어디인가요?
    4. 군집마다 대표 영화가 어떤 공통점을 보이나요?

    **주의:** 군집 이름은 분석자가 평균과 대표 영화를
    읽고 붙입니다. 알고리즘이 붙인 군집 번호 자체에는
    순위나 작품성의 의미가 없습니다.
    """
)

st.caption(
    "데이터 출처: 영화진흥위원회(KOBIS) 데이터 파일을 "
    "수업 자료에서 제공한 URL로 불러옵니다."
)
