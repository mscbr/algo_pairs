from tools.kalman_filters.py_kalman import KalmanFilterAverage, KalmanFilterRegression
from tools.kalman_filters.vanilla_kalman import VanillaKalmanFilter
from tools.spread_features import half_life
from statsmodels.tsa.stattools import adfuller, coint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uuid
import os
import sys
from pathlib import Path

parent_dir = os.path.abspath('..')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


class Coint_Analyzer:
    def __init__(
        self,
        raw_data_path="../raw_data/",
        closing_prices_container_paths=[
            "Binance_Historical_15m_FUTURES_20_days_2022-07-14T12:00:43"],
        processed_data_path="../processed/",
        cache_path=None,
        interval="15m",
        observations_filter=None,
        days_filter=None,
        corr_filter=None
    ):
        self.raw_data_path = raw_data_path
        self.closing_prices_container_paths = closing_prices_container_paths
        self.processed_data_path = processed_data_path
        self.cache_path = cache_path

        self.observations_filter = observations_filter
        self.days_filter = days_filter
        self.df_observations = None

        self.uuid = str(uuid.uuid4())
        self.interval = interval
        self.interval_to_days_map = {
            "1m": 1440,
            "3m": 480,
            "5m": 288,
            "15m": 96,
            "30m": 48,
            "1h": 24
        }

        self.df = None
        self.corr_pairs = None
        self.corr_filter = corr_filter
        self.coint_pairs = None
        self.coint_pvals = None
        self.corr_coint_pairs = None

        self._closings_csv_to_df()

        if not os.path.exists(self.processed_data_path):
            Path(self.processed_data_path).mkdir(parents=True, exist_ok=True)

    def __repr__(self):
        return "Closing price cointegration analyzer"

    def process_raw_data(self, closing_prices_container_paths=None):
        if closing_prices_container_paths is not None:
            self.closing_prices_container_paths = closing_prices_container_paths

        self.uuid = str(uuid.uuid4())

        self.generate_co_matrices()
        self.get_trading_pairs()

    def _closings_csv_to_df(self):
        # reading Close values and merging to one DF
        df_closings = pd.DataFrame()

        for path in self.closing_prices_container_paths:
            with os.scandir('%s%s' % (self.raw_data_path, path)) as entries:
                for entry in entries:
                    instrument = "_".join(entry.name.split("_")[0:2])
                    df = pd.read_csv('%s%s/%s' %
                                     (self.raw_data_path, path, entry.name), index_col="Date")
                    df = df[["Close"]].copy()
                    df.columns = [instrument]
                    df_closings = pd.concat([df_closings, df], axis=1)

        # filtering data based on amount of observations in DF
        if self.days_filter != None or self.observations_filter != None:
            self.df = df_closings
            self._filter_by_observations(
                days=self.days_filter, observations=self.observations_filter)
        else:
            self.df = df_closings.dropna()

    def generate_co_matrices(
        self,
        generate_excel=False,
        vanilla_kalman=False,
        create_cache=False,
        save_spreads=False,
        raw_spread=False,
    ):
        # CORRELATION
        self._get_correlated_pairs(
            generate_excel=generate_excel, corr_filter=self.corr_filter)
        # COINTEGRATION
        self._get_cointegrated_pairs(
            vanilla_kalman=vanilla_kalman, raw_spread=raw_spread, create_cache=create_cache, save_spreads=save_spreads)

    def _get_correlated_pairs(self, generate_excel=False, corr_filter=None):
        corr_matrix = self.df.pct_change().corr(method='pearson')
        if generate_excel:
            corr_matrix.to_excel("%scorr_matrix_%s_%s.xlsx" %
                                 (self.processed_data_path, self.interval, self.uuid))

        au_corr = corr_matrix.corr().unstack()
        labels_to_drop = self._get_redundant_corr_pairs(corr_matrix)
        au_corr = au_corr.drop(
            labels=labels_to_drop).sort_values(ascending=False)
        au_corr.dropna(inplace=True)

        indexes = []
        values = []
        for idx in au_corr.index:
            indexes.append("%s-%s" % (idx[0], idx[1]))
            values.append(au_corr[idx])
        corr_pairs_df = pd.DataFrame(index=indexes, data=values)

        if corr_filter:
            corr_pairs_df = corr_pairs_df.loc[corr_pairs_df.iloc[:, 0]
                                              > corr_filter]

        self.corr_pairs = corr_pairs_df
        try:
            corr_pairs_df.to_csv("%scorr_pairs_%s_%s.csv" %
                                 (self.processed_data_path, self.interval, self.uuid))
        except:
            print("Couldn't save correlated pairs to files")

    def _get_cointegrated_pairs(
        self,
        vanilla_kalman=False,
        raw_spread=False,
        create_cache=False,
        save_spreads=False
    ):
        cache = None
        cache_result = None

        if save_spreads:
            save_spreads_dir = "%scoint_pairs_%s_%s/" % (
                self.processed_data_path, self.interval, self.uuid)
            Path(save_spreads_dir).mkdir(parents=True, exist_ok=True)

        df = self.df.copy()

        pairs = []
        corr_pairs_names = [pair.split("-")
                            for pair in list(self.corr_pairs.index)]

        # INIT CACHE HANDLING
        if create_cache and not self.cache_path:
            cache = pd.DataFrame(index=list(self.corr_pairs.index),
                                 data={
                                     'analyzed': [False],
                                     'corr': [np.nan],
                                     'adf': [np.nan],
                                     'hurst': [np.nan],
                                     'half_life': [np.nan],
            })
            cache_path = ("%scoint_anal_cache_%s_%s.csv" %
                          (self.processed_data_path, self.interval, self.uuid))
            self.cache_path = cache_path
        elif self.cache_path:
            cache = pd.read_csv(self.cache_path, index_col=0)
        if cache is not None:
            cache_finished_pairs = list(
                cache.loc[cache.analyzed == True].index)
            pairs_to_analyze = [
                pair for pair in list(self.corr_pairs.index) if pair not in cache_finished_pairs]
            corr_pairs_names = [pair.split("-")
                                for pair in pairs_to_analyze]

        # RUNNING ANALYSIS
        print("STARTING ANALYSIS")
        for i, corr_pair in enumerate(corr_pairs_names):
            print("Performing coint test %s %s" % (i, len(corr_pairs_names)))
            inst_1 = corr_pair[0]
            inst_2 = corr_pair[1]
            result = coint(df[inst_1], df[inst_2])
            if cache is not None:
                cache_result = [np.nan, np.nan, np.nan, np.nan]

            # testing for spread stationarity
            if result[1] < 0.05:
                if vanilla_kalman:
                    mkf = VanillaKalmanFilter(delta=1e-4, R=2)
                    spread, hedge_ratio = mkf.regression(
                        df[inst_1], df[inst_2])
                elif raw_spread:
                    spread = df[inst_1] - ((df[inst_1]/df[inst_2]*df[inst_2]))
                else:
                    # more variations could be implemented
                    state_means = KalmanFilterRegression(
                        df[inst_1], df[inst_2])
                    hedge_ratio = - state_means[:, 0]
                    spread = df[inst_2] + (df[inst_1] * hedge_ratio)

                result_adf = adfuller(spread)
                if result_adf[1] < 0.01 and result_adf[0] < result_adf[4]["10%"]:
                    # is it mean reverting
                    hurst = self._get_hurst_exponent(np.array(spread))
                    if hurst <= 0.5:
                        print("hurst", hurst)
                        index = "%s-%s" % (inst_1, inst_2)
                        half_life_value = 1  # half_life(spread)
                        print("half_life", half_life_value)
                        pairs.append(
                            (index, self.corr_pairs.loc[index][0], result_adf[0], hurst, half_life_value))
                        if save_spreads is not None:
                            print("SAVING SPREAD")
                            df_spread = df[[inst_1, inst_2]].copy()
                            df_spread.loc[:, 'spread'] = spread
                            try:
                                df_spread.to_csv(
                                    "%s%s_%s_spread.csv" % (save_spreads_dir, inst_1, inst_2))
                                df_spread["spread"].plot(figsize=(12, 8))
                                plt.grid()
                                plt.rcParams['figure.facecolor'] = 'lavender'
                                plt.savefig("%s%s_%s_spread.png" %
                                            (save_spreads_dir, inst_1, inst_2))
                                plt.clf()
                            except:
                                print("Couldn't save spread for %s_%s" %
                                      (inst_1, inst_2))
                        if cache is not None:
                            cache_result = [
                                self.corr_pairs.loc[index][0],
                                result_adf[0], hurst, half_life_value]
            # UPDATING CACHE
            if cache is not None:
                try:
                    cache.loc["%s-%s" % (inst_1, inst_2)] = [
                        True,
                        *cache_result
                    ]
                    cache.to_csv(self.cache_path)
                except:
                    print("Couldn't write coint cache")

        try:
            indexes = []
            corr = []
            adf = []
            hurst = []
            half_life_values = []
            for column in pairs:
                indexes.append(column[0])
                corr.append(column[1])
                adf.append(column[2])
                hurst.append(column[3])
                half_life_values.append(column[4])

            # ADDING CACHED RESULTS
            if cache is not None:
                cache.dropna(inplace=True)
                for i in range(0, len(cache)):
                    if cache.index[i] not in indexes:
                        column = cache.iloc[i].tolist()
                        indexes.append(cache.index[i])
                        corr.append(column[1])
                        adf.append(column[2])
                        hurst.append(column[3])
                        half_life_values.append(column[4])

            coint_pairs_df = pd.DataFrame(index=indexes)
            coint_pairs_df['corr'] = corr
            coint_pairs_df['adf'] = adf
            coint_pairs_df['hurst'] = hurst
            coint_pairs_df.sort_values(
                by='corr', ascending=False, inplace=True)
            coint_pairs_df.to_csv("%scoint_pairs_%s_%s.csv" %
                                  (self.processed_data_path, self.interval, self.uuid))

        except:
            print("Couldn't save cointegrated pairs to files")

        self.coint_pairs = coint_pairs_df

    def _get_redundant_corr_pairs(self, df_corr_matrix):
        '''Get diagonal and lower triangular pairs of correlation matrix'''
        pairs_to_drop = set()
        cols = df_corr_matrix.columns
        for i in range(0, df_corr_matrix.shape[1]):
            for j in range(0, i+1):
                pairs_to_drop.add((cols[i], cols[j]))
        return pairs_to_drop

    def _get_hurst_exponent(self, time_series):
        """Returns the Hurst Exponent of the time series vector ts"""
        # Create the range of lag values
        lags = range(2, 20)
        # Calculate the array of the variances of the lagged differences
        tau = [np.sqrt(np.std(np.subtract(time_series[lag:],
                                          time_series[:-lag]))) for lag in lags]
        # Use a linear fit to estimate the Hurst Exponent
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        # Return the Hurst exponent from the polyfit output
        return poly[0]*2.0

    def _filter_by_observations(self, days=None, observations=None):
        df_observations = pd.DataFrame(columns=["observations"])
        for column in self.df.columns:
            df_observations.loc[column] = len(self.df[column].dropna())

        df_observations["days"] = df_observations["observations"] / \
            self.interval_to_days_map[self.interval]
        df_observations.sort_values(
            by="observations", ascending=False, inplace=True)
        self.df_observations = df_observations

        if days:
            target_instruments = df_observations.loc[df_observations.days > days].index
            self.df = self.df.filter(items=target_instruments).dropna()

        if observations:
            target_instruments = df_observations.loc[df_observations.observations >
                                                     observations].index
            self.df = self.df.filter(items=target_instruments).dropna

    def get_trading_pairs(self, corr_path=None, coint_path=None):
        df_corr = None
        df_coint = None
        if corr_path is not None and coint_path is not None:
            df_corr = pd.read_csv(corr_path)
            df_coint = pd.read_csv(coint_path)
        elif self.corr_pairs is not None and self.coint_pairs is not None:
            df_corr = self.corr_pairs.copy()
            df_coint = self.coint_pairs.copy()

        if df_corr is None or df_coint is None:
            return

        df_corr_coint_pairs = pd.DataFrame(
            columns=["corr", "adf", "hurst", "granger_12", "granger_21"])
        for idx in df_corr.index:
            if idx in df_coint.index:
                df_corr_coint_pairs.loc[idx] = [df_corr.loc[idx][0], df_coint.loc[idx]
                                                [0], df_coint.loc[idx][1], df_coint.loc[idx][2], df_coint.loc[idx][3]]

        self.corr_coint_pairs = df_corr_coint_pairs
        try:
            df_corr_coint_pairs.to_csv("%scorr_coint_pairs_%s_%s.csv" %
                                       (self.processed_data_path, self.interval, self.uuid))
        except:
            print("Data couldn't be stored in a static file.")
        return df_corr_coint_pairs
