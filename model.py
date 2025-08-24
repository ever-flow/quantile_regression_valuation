# BEST SO FAR (8.22)
# 최종 출력파일 수정

!pip install optuna

import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor, early_stopping
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import optuna
import warnings
warnings.filterwarnings("ignore")

# 산업 PER 통계 계산 함수
def compute_industry_per_stats(df, method, industry_cols):
    industries = df[industry_cols].stack().dropna().unique()
    per_values = {}
    for ind in industries:
        mask = df[industry_cols].apply(lambda r: ind in r.values, axis=1)
        if method == 'AG1':
            sub = df.loc[mask & df['Net Income (LTM)'].notna()]
            if sub.empty:
                per_values[ind] = np.nan
                continue
            q1 = sub['market_cap'].quantile(0.25)
            q3 = sub['market_cap'].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 2 * iqr
            upper_bound = q3 + 2 * iqr
            sub_filtered = sub[(sub['market_cap'] >= lower_bound) & (sub['market_cap'] <= upper_bound)]
            if sub_filtered.empty:
                per_values[ind] = np.nan
            else:
                total_market_cap = sub_filtered['market_cap'].sum()
                total_net_income = sub_filtered['Net Income (LTM)'].sum()
                per_values[ind] = total_market_cap / total_net_income if total_net_income > 0 else np.nan
        else:
            sub = df.loc[mask & df['Net Income (LTM)'].gt(0) & df['Net Income (LTM)'].notna()]
            if sub.empty:
                per_values[ind] = np.nan
                continue
            per_vals = sub['market_cap'] / sub['Net Income (LTM)']
            if method == 'SA':
                per_values[ind] = per_vals.mean()
            elif method == 'MD':
                per_values[ind] = per_vals.median()
            elif method == 'NH':
                per_values[ind] = len(per_vals) / (1.0 / per_vals).sum() if len(per_vals) > 0 else np.nan
    return per_values

# 산업 EV/EBITDA 통계 계산 함수
def compute_industry_ev_ebitda_stats(df, method, industry_cols):
    industries = df[industry_cols].stack().dropna().unique()
    ev_ebitda_values = {}
    for ind in industries:
        mask = df[industry_cols].apply(lambda r: ind in r.values, axis=1)
        if method == 'AG1':
            sub = df.loc[mask & df['EBITDA (LTM)'].notna() & df['Enterprise Value (FQ0)'].notna()]
            if sub.empty:
                ev_ebitda_values[ind] = np.nan
                continue
            q1 = sub['Enterprise Value (FQ0)'].quantile(0.25)
            q3 = sub['Enterprise Value (FQ0)'].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 2 * iqr
            upper_bound = q3 + 2 * iqr
            sub_filtered = sub[(sub['Enterprise Value (FQ0)'] >= lower_bound) & (sub['Enterprise Value (FQ0)'] <= upper_bound)]
            if sub_filtered.empty:
                ev_ebitda_values[ind] = np.nan
            else:
                total_ev = sub_filtered['Enterprise Value (FQ0)'].sum()
                total_ebitda = sub_filtered['EBITDA (LTM)'].sum()
                ev_ebitda_values[ind] = total_ev / total_ebitda if total_ebitda > 0 else np.nan
        else:
            sub = df.loc[mask & df['EBITDA (LTM)'].gt(0) & df['EBITDA (LTM)'].notna() & df['Enterprise Value (FQ0)'].notna()]
            if sub.empty:
                ev_ebitda_values[ind] = np.nan
                continue
            ev_vals = sub['Enterprise Value (FQ0)'] / sub['EBITDA (LTM)']
            if method == 'SA':
                ev_ebitda_values[ind] = ev_vals.mean()
            elif method == 'MD':
                ev_ebitda_values[ind] = ev_vals.median()
            elif method == 'NH':
                ev_ebitda_values[ind] = len(ev_vals) / (1.0 / ev_vals).sum() if len(ev_vals) > 0 else np.nan
    return ev_ebitda_values

# 개별 기업의 PER 기반 추정 시가총액 계산
def compute_per_feature(df, per_values, industry_cols):
    def feature(row):
        nets = row['Net Income (LTM)']
        if pd.isna(nets):
            return np.nan
        inds = [row[c] for c in industry_cols if pd.notna(row[c])]
        pers = [per_values.get(i, np.nan) for i in inds if not np.isnan(per_values.get(i, np.nan))]
        return np.mean(pers) * nets if pers else np.nan
    return df.apply(feature, axis=1)

# 개별 기업의 EV/EBITDA 기반 추정 시가총액 계산
def compute_ev_ebitda_feature(df, ev_ebitda_values, industry_cols):
    def feature(row):
        ebitda = row['EBITDA (LTM)']
        if pd.isna(ebitda):
            return np.nan
        inds = [row[c] for c in industry_cols if pd.notna(row[c])]
        ev_ebitdas = [ev_ebitda_values.get(i, np.nan) for i in inds if not np.isnan(ev_ebitda_values.get(i, np.nan))]
        if not ev_ebitdas:
            return np.nan
        ev_estimate = np.mean(ev_ebitdas) * ebitda
        net_debt = row.get('Net Debt (LTM)', 0)
        return ev_estimate - net_debt if pd.notna(net_debt) else np.nan
    return df.apply(feature, axis=1)

# 도메인 및 시간 기반 피처 엔지니어링 클래스
class DomainFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        Xt = self._transform(X.copy())
        self.cols_ = Xt.columns.tolist()
        return self

    def transform(self, X):
        Xt = self._transform(X.copy())
        return Xt.reindex(columns=self.cols_, fill_value=0)

    def _transform(self, X):
        X = X.drop('market', axis=1, errors='ignore')
        tcols = [c for c in X.columns if '(' in c and 'LTM' in c]
        tf = {}
        for c in tcols:
            base, per = c.split(' (')
            tf.setdefault(base, {})[per.rstrip(')')] = c

        if 'EBIT' in tf and 'Depreciation' in tf:
            for p, e in tf['EBIT'].items():
                d = tf['Depreciation'].get(p)
                if d:
                    X[f'EBITDA ({p})'] = X[e] + X[d]
            tf['EBITDA'] = {p: f'EBITDA ({p})' for p in tf['EBIT']}

        bases = ['Total Assets', 'Total Liabilities', 'Equity', 'Net Debt',
                 'Revenue', 'EBIT', 'EBITDA', 'Net Income', 'Net Income After Minority', 'Dividends']

        def safe_div(a, b):
            return a.div(b.replace({0: np.nan}))

        for v in bases:
            periods = tf.get(v, {})
            seq = [p for p in ['LTM-3', 'LTM-2', 'LTM-1', 'LTM'] if p in periods]
            grs = []
            for i in range(1, len(seq)):
                prev, curr = seq[i-1], seq[i]
                r = safe_div(X[periods[curr]] - X[periods[prev]], X[periods[prev]])
                suffix = '-2' if (prev, curr) == ('LTM-3', 'LTM-2') else '-1' if (prev, curr) == ('LTM-2', 'LTM-1') else ''
                X[f'{v}_growth{suffix}'] = r
                grs.append(r)
            if grs:
                X[f'{v}_avg_growth'] = pd.concat(grs, axis=1).mean(axis=1)
                X[f'{v}_volatility'] = X[[periods[p] for p in seq]].std(axis=1)
            if 'LTM-3' in periods and 'LTM' in periods:
                X[f'{v}_CAGR'] = np.where(
                    X[periods['LTM-3']] > 0,
                    (X[periods['LTM']] / X[periods['LTM-3']])**(1/3) - 1,
                    np.nan
                )

        def calc(a, b, name, per):
            if per in tf.get(a, {}) and per in tf.get(b, {}):
                X[f'{name} ({per})'] = safe_div(X[tf[a][per]], X[tf[b][per]])

        ratios = [
            ('Equity', 'Total Assets', 'BAR'),
            ('Total Liabilities', 'Equity', 'DBR'),
            ('Revenue', 'Total Assets', 'SAR'),
            ('EBIT', 'Revenue', 'OMR'),
            ('EBITDA', 'Revenue', 'EMR'),
            ('Net Income', 'Total Assets', 'EAR'),
            ('Net Income', 'Equity', 'EBR')
        ]
        for p in ['LTM', 'LTM-1', 'LTM-2', 'LTM-3']:
            for a, b, n in ratios:
                calc(a, b, n, p)
            if p in ['LTM-2', 'LTM-1', 'LTM']:
                ni = tf.get('Net Income After Minority', {}).get(p)
                eq = tf.get('Equity', {}).get(p)
                prev = {'LTM-2': 'LTM-3', 'LTM-1': 'LTM-2', 'LTM': 'LTM-1'}[p]
                ep = tf.get('Equity', {}).get(prev)
                if ni and eq and ep:
                    avg_eq = (X[eq] + X[ep]) / 2
                    X[f'ROE ({p})'] = safe_div(X[ni], avg_eq)

        ratio_bases = ['BAR', 'DBR', 'SAR', 'OMR', 'EMR', 'EAR', 'EBR', 'ROE']
        for v in ratio_bases:
            periods = {p: f'{v} ({p})' for p in ['LTM-3', 'LTM-2', 'LTM-1', 'LTM'] if f'{v} ({p})' in X.columns}
            seq = [p for p in ['LTM-3', 'LTM-2', 'LTM-1', 'LTM'] if p in periods]
            grs = []
            for i in range(1, len(seq)):
                prev, curr = seq[i-1], seq[i]
                r = safe_div(X[periods[curr]] - X[periods[prev]], X[periods[prev]])
                suffix = '-2' if (prev, curr) == ('LTM-3', 'LTM-2') else '-1' if (prev, curr) == ('LTM-2', 'LTM-1') else ''
                X[f'{v}_growth{suffix}'] = r
                grs.append(r)
            if grs:
                X[f'{v}_avg_growth'] = pd.concat(grs, axis=1).mean(axis=1)
                X[f'{v}_volatility'] = X[[periods[p] for p in seq]].std(axis=1)
            if 'LTM-3' in periods and 'LTM' in periods:
                X[f'{v}_CAGR'] = np.where(
                    X[periods['LTM-3']] > 0,
                    (X[periods['LTM']] / X[periods['LTM-3']])**(1/3) - 1,
                    np.nan
                )

        dep_cols = [c for c in X.columns if c.startswith('Depreciation')]
        if dep_cols:
            X.drop(columns=dep_cols, inplace=True)

        return X

# Optuna를 통한 LightGBM 하이퍼파라미터 최적화
def optimize_lightgbm(X, y, selected_features, n_trials=100):
    def objective(trial):
        param = {
            'objective': 'quantile',
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 10, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 200),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        coverages, mean_ratios, ratios = [], [], []
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx][selected_features], X.iloc[val_idx][selected_features]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            low_model = LGBMRegressor(alpha=0.1, **param)
            low_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='quantile',
                callbacks=[early_stopping(stopping_rounds=10, verbose=False)],
                categorical_feature=[col for col in X_train.columns if col in ['EMSEC1', 'EMSEC2', 'EMSEC3', 'EMSEC4']]
            )
            q_low = np.clip(low_model.predict(X_val), y.min() * 0.01, None)

            high_model = LGBMRegressor(alpha=0.9, **param)
            high_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='quantile',
                callbacks=[early_stopping(stopping_rounds=10, verbose=False)],
                categorical_feature=[col for col in X_train.columns if col in ['EMSEC1', 'EMSEC2', 'EMSEC3', 'EMSEC4']]
            )
            q_high = np.maximum(high_model.predict(X_val), q_low)

            coverages.append(((y_val >= q_low) & (y_val <= q_high)).mean())
            mean_ratios.append((q_high / q_low).mean())
            ratios.append((q_high / q_low).std())

        mean_cov = np.mean(coverages)
        mean_ratio = np.mean(mean_ratios)
        ratio_std = np.mean(ratios)
        penalty = max(0, (0.8 - mean_cov) * 100)
        score = 0.6 * mean_ratio + 0.4 * ratio_std + penalty

        trial.set_user_attr('mean_cov', mean_cov)
        trial.set_user_attr('mean_ratio', mean_ratio)
        trial.set_user_attr('ratio_std', ratio_std)

        return score

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)
    best_params = study.best_params
    best_params.update({
        'objective': 'quantile',
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    })

    best_trial = study.best_trial
    cv_metrics = {
        'mean_cov': best_trial.user_attrs['mean_cov'],
        'mean_ratio': best_trial.user_attrs['mean_ratio'],
        'ratio_std': best_trial.user_attrs['ratio_std']
    }
    return best_params, cv_metrics

# 최적화된 LightGBM 퀀타일 모델 실행
def quick_lightgbm_quantile(df, name, all_industry_cols):
    df = df.copy()

    if 'EBITDA (LTM)' not in df and 'EBIT (LTM)' in df and 'Depreciation (LTM)' in df:
        df['EBITDA (LTM)'] = df['EBIT (LTM)'] + df['Depreciation (LTM)']

    df['Listing_Age_Days'] = (pd.to_datetime('2024-12-31') - pd.to_datetime(df['Listing Date'])).dt.days
    df.drop(columns=['Listing Date'], inplace=True)

    methods = ['SA', 'MD', 'NH', 'AG1']
    for method in methods:
        per_values = compute_industry_per_stats(df, method, all_industry_cols)
        ev_ebitda_values = compute_industry_ev_ebitda_stats(df, method, all_industry_cols)
        df[f'industry_PER_est_{method}'] = compute_per_feature(df, per_values, all_industry_cols)
        df[f'industry_EV_EBITDA_est_{method}'] = compute_ev_ebitda_feature(df, ev_ebitda_values, all_industry_cols)

    for col in all_industry_cols:
        if col in df:
            df[col] = df[col].astype('category')

    X = df.select_dtypes(include=[np.number, 'category']).drop(
        columns=['Market Cap (2024-12-31)', 'market_cap', 'Enterprise Value (FQ0)', 'ticker'], errors='ignore')
    y = df['market_cap']

    fe = DomainFeatureEngineer().fit(X, y)
    X_fe = fe.transform(X)

    num_cols = X_fe.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    X_fe[num_cols] = scaler.fit_transform(X_fe[num_cols])

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    oof_q_low = np.zeros(len(y))
    oof_q_high = np.zeros(len(y))
    coverages, mean_ratios, ratios = [], [], []

    for train_idx, val_idx in kf.split(X_fe):
        X_train, X_val = X_fe.iloc[train_idx], X_fe.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        selector = LGBMRegressor(n_estimators=200, random_state=42)
        selector.fit(X_train, y_train)
        imp = selector.feature_importances_
        imp = imp / imp.sum()
        imp_series = pd.Series(imp, index=X_train.columns).sort_values(ascending=False)
        selected = imp_series[imp_series > 0.001].index.tolist()
        max_features = min(len(X_train.columns), int(len(X_train) / 30))
        selected = imp_series.nlargest(max_features).index.tolist()
        X_train_sel = X_train[selected]
        X_val_sel = X_val[selected]

        best_params, cv_metrics = optimize_lightgbm(X_train_sel, y_train, selected)

        low_model = LGBMRegressor(alpha=0.1, **best_params)
        high_model = LGBMRegressor(alpha=0.9, **best_params)
        low_model.fit(X_train_sel, y_train, categorical_feature=[col for col in selected if col in all_industry_cols])
        high_model.fit(X_train_sel, y_train, categorical_feature=[col for col in selected if col in all_industry_cols])

        q_low = np.clip(low_model.predict(X_val_sel), y.min() * 0.01, None)
        q_high = np.maximum(high_model.predict(X_val_sel), q_low)

        oof_q_low[val_idx] = q_low
        oof_q_high[val_idx] = q_high

        coverages.append(((y_val >= q_low) & (y_val <= q_high)).mean())
        mean_ratios.append((q_high / q_low).mean())
        ratios.append((q_high / q_low).std())

    mean_cov = np.mean(coverages)
    mean_ratio = np.mean(mean_ratios)
    ratio_std = np.mean(ratios)
    penalty = max(0, (0.8 - mean_cov) * 100)
    score = 0.6 * mean_ratio + 0.4 * ratio_std + penalty

    # EMSEC 및 EMTEC 열을 포함하여 최종 결과 데이터프레임 생성

    # 1. 저장할 EMSEC, EMTEC 관련 열 이름 목록 생성
    em_cols_to_keep = [col for col in df.columns if col.startswith('EMSEC') or col.startswith('EMTEC')]

    # 2. 티커와 EMSEC, EMTEC 열을 먼저 포함하여 기본 데이터프레임 생성
    res = df[['ticker'] + em_cols_to_keep].copy()

    # 3. 예측 결과 및 계산된 열 추가
    res['Actual'] = y
    res['Q0.1'] = oof_q_low
    res['Q0.9'] = oof_q_high
    res['Within'] = (y >= oof_q_low) & (y <= oof_q_high)
    res['Ratio'] = oof_q_high / oof_q_low

    print(f"\n[{name}] OOF 성능 메트릭:")
    print(f"  Coverage: {mean_cov:.4f}")
    print(f"  Mean Ratio: {mean_ratio:.4f}")
    print(f"  Ratio Std: {ratio_std:.4f}")
    print(f"  Score: {score:.4f}")

    return res

if __name__ == "__main__":
    PATH = "/content/heatmap_data_with_SE_v2.xlsx"
    df = pd.read_excel(PATH, sheet_name="Sheet1")

    # industry_cols = ['Industry1', 'Industry2', 'Industry3', 'Industry4', 'Industry5'] # 이 변수는 현재 사용되지 않음
    emsec_cols = ['EMSEC1', 'EMSEC2', 'EMSEC3', 'EMSEC4', 'EMSEC5']
    all_industry_cols = emsec_cols
    for col in all_industry_cols:
        if col in df:
            df[col] = df[col].astype('category')

    df = df[df["market"].isin(["KOSDAQ", "KOSDAQ GLOBAL", "KOSPI"]) & (df["Market Cap (2024-12-31)"] > 0)]
    df["market_cap"] = df["Market Cap (2024-12-31)"]

    df = df[df['EMSEC1'].notna()].reset_index(drop=True)

    num_cols = df.select_dtypes(include=[np.number]).columns
    df = df[df[num_cols].isnull().mean(axis=1) <= 0.5].reset_index(drop=True)

    # Isolation Forest로 이상치 제거
    iso_num_cols = df.select_dtypes(include=[np.number]).columns.drop(['market_cap'], errors='ignore')
    iso_data = df[iso_num_cols].copy()
    iso_data_nonnull = iso_data.dropna()
    if not iso_data_nonnull.empty:
        iso = IsolationForest(contamination=0.05, random_state=42)
        outliers = iso.fit_predict(iso_data_nonnull)
        non_outliers = outliers != -1
        non_outlier_indices = iso_data_nonnull.index[non_outliers]
        df = df.loc[non_outlier_indices].reset_index(drop=True)
    else:
        print("Isolation Forest 적용 불가: 유효 데이터 없음")

    kosdaq = df[df["market"].isin(["KOSDAQ", "KOSDAQ GLOBAL"])]
    kospi = df[df["market"] == "KOSPI"]

    for market_df, market_name in [(kosdaq, "KOSDAQ & KOSDAQ GLOBAL"), (kospi, "KOSPI")]:
        if market_df.empty:
            print(f"\n[{market_name}] 데이터가 없어 모델 학습을 건너뜁니다.")
            continue
        print(f"\n[{market_name}] 모델 적합 중...")
        res = quick_lightgbm_quantile(market_df, market_name, all_industry_cols)
        res.to_excel(f"{market_name}_results.xlsx", index=False)
        print(f"[{market_name}] 결과가 '{market_name}_results.xlsx' 파일로 저장되었습니다.")
