from tools.kalman_filters.py_kalman import KalmanFilterAverage, KalmanFilterRegression
from tools.kalman_filters.vanilla_kalman import VanillaKalmanFilter
from tools.spread_features import half_life
from statsmodels.tsa.stattools import adfuller, coint
import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import uuid
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

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
        corr_filter=None,
        half_life_min=None,
        half_life_max=None,
        lookback_days=None,
        spread_method="kalman",
    ):
        self.raw_data_path = raw_data_path
        self.closing_prices_container_paths = closing_prices_container_paths
        self.processed_data_path = processed_data_path
        self.cache_path = cache_path

        self.observations_filter = observations_filter
        self.days_filter = days_filter
        self.df_observations = None

        self.half_life_min = half_life_min
        self.half_life_max = half_life_max
        self.lookback_days = lookback_days
        self.spread_method = spread_method

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
                    # Ensure index is datetime
                    df.index = pd.to_datetime(df.index)
                    
                    # Handle duplicate indices
                    df = df[~df.index.duplicated(keep='first')]
                    
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
        create_cache=False,
        save_spreads=False,
        remove_quote_currency_pairs=True,
        spread_method=None,
    ):
        if spread_method is not None:
            self.spread_method = spread_method
        # CORRELATION
        self._get_correlated_pairs(
            generate_excel=generate_excel, corr_filter=self.corr_filter, remove_quote_currency_pairs=remove_quote_currency_pairs)
        # COINTEGRATION
        self._get_cointegrated_pairs(
            create_cache=create_cache, save_spreads=save_spreads)

    def _get_correlated_pairs(self, generate_excel=False, corr_filter=None, remove_quote_currency_pairs=True):
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
        
        if remove_quote_currency_pairs:
            corr_pairs_df = self._remove_quote_currency_pairs(corr_pairs_df)

        if corr_filter:
            corr_pairs_df = corr_pairs_df.loc[corr_pairs_df.iloc[:, 0]
                                              > corr_filter]

        self.corr_pairs = corr_pairs_df
        try:
            corr_pairs_df.to_csv("%scorr_pairs_%s_%s.csv" %
                                 (self.processed_data_path, self.interval, self.uuid))
        except (IOError, OSError) as e:
            logger.warning("Couldn't save correlated pairs to files: %s", e)

    def _get_cointegrated_pairs(
        self,
        create_cache=False,
        save_spreads=False,
    ):
        cache = None
        cache_result = None

        if save_spreads:
            save_spreads_dir = "%scoint_pairs_%s_%s/" % (
                self.processed_data_path, self.interval, self.uuid)
            Path(save_spreads_dir).mkdir(parents=True, exist_ok=True)

        df = self.df.copy()

        if self.lookback_days is not None:
            n_obs = int(self.lookback_days * self.interval_to_days_map[self.interval])
            if len(df) > n_obs:
                df = df.iloc[-n_obs:]

        pairs = []
        corr_pairs_names = [pair.split("-") for pair in list(self.corr_pairs.index)]

        # INIT CACHE HANDLING
        if create_cache and not self.cache_path:
            cache = pd.DataFrame(index=list(self.corr_pairs.index),
                                data={
                                    'analyzed': False,
                                    'corr': np.nan,
                                    'adf': np.nan,
                                    'hurst': np.nan,
                                    'half_life': np.nan,
            })
            cache_path = ("%scoint_anal_cache_%s_%s.csv" %
                        (self.processed_data_path, self.interval, self.uuid))
            self.cache_path = cache_path
        elif self.cache_path:
            cache = pd.read_csv(self.cache_path, index_col=0)
        if cache is not None:
            cache_finished_pairs = list(cache.loc[cache.analyzed == True].index)
            pairs_to_analyze = [
                pair for pair in list(self.corr_pairs.index) if pair not in cache_finished_pairs]
            corr_pairs_names = [pair.split("-") for pair in pairs_to_analyze]

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
                spread, hedge_ratio = self._compute_spread(df[inst_1], df[inst_2], self.spread_method)

                result_adf = adfuller(spread)
                if result_adf[1] < 0.05:
                    hurst = self._get_hurst_exponent(np.array(spread))
                    if hurst <= 0.5:
                        print("hurst", hurst)
                        index = "%s-%s" % (inst_1, inst_2)
                        half_life_value = half_life(spread)
                        print("half_life", half_life_value)

                        if self.half_life_min is not None and half_life_value < self.half_life_min:
                            continue
                        if self.half_life_max is not None and half_life_value > self.half_life_max:
                            continue

                        pairs.append(
                            (index, self.corr_pairs.loc[index][0], result_adf[0], hurst, half_life_value))

                        if save_spreads:
                            print("SAVING SPREADS")
                            df_spread = df[[inst_1, inst_2]].copy()
                            df_spread['spread'] = spread

                            try:
                                df_spread.to_csv("%s%s_%s_spread.csv" % (save_spreads_dir, inst_1, inst_2))

                                plt.figure(figsize=(12, 6))
                                plt.plot(spread, label='spread (%s)' % self.spread_method, color='blue', alpha=0.7)
                                plt.title(f'{self.spread_method} spread: {inst_1} vs {inst_2}')
                                plt.xlabel('Time')
                                plt.ylabel('Spread Value')
                                plt.legend()
                                plt.grid(True)
                                plt.rcParams['figure.facecolor'] = 'lavender'
                                plt.savefig("%s%s_%s_spread.pdf" % (save_spreads_dir, inst_1, inst_2))
                                plt.close()

                            except Exception as e:
                                print(f"Couldn't save spread for {inst_1}_{inst_2}: {str(e)}")

                        if cache is not None:
                            cache_result = [
                                self.corr_pairs.loc[index][0],
                                result_adf[0], hurst, half_life_value]
            
            # UPDATING CACHE (unchanged from your original code)
            if cache is not None:
                try:
                    cache.loc["%s-%s" % (inst_1, inst_2)] = [True, *cache_result]
                    cache.to_csv(self.cache_path)
                except (IOError, OSError, KeyError) as e:
                    logger.warning("Couldn't write coint cache: %s", e)

        # FINAL SAVE (unchanged from your original code)
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
            coint_pairs_df['half_life'] = half_life_values
            coint_pairs_df.sort_values(by='corr', ascending=False, inplace=True)
            coint_pairs_df.to_csv("%scoint_pairs_%s_%s.csv" %
                                (self.processed_data_path, self.interval, self.uuid))

        except (IOError, OSError, ValueError) as e:
            logger.warning("Couldn't save cointegrated pairs to files: %s", e)

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
        max_lag = min(100, len(time_series) // 2)
        lags = range(2, max(max_lag, 20))
        tau = [np.std(np.subtract(time_series[lag:],
                                  time_series[:-lag])) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]

    def _compute_spread(self, series_1, series_2, method):
        """Compute spread and hedge ratio using the specified method.

        Parameters
        ----------
        series_1, series_2 : pd.Series
        method : str  — "kalman" | "ols" | "pykalman"

        Returns
        -------
        (spread, hedge_ratio)
        """
        if method == "kalman":
            mkf = VanillaKalmanFilter(delta=1e-4, R=2)
            spread, hedge_ratio = mkf.regression(series_1, series_2)
        elif method == "ols":
            X = sm.add_constant(series_2)
            model = sm.OLS(series_1, X).fit()
            hedge_ratio = model.params.iloc[1]
            spread = series_1 - hedge_ratio * series_2
        elif method == "pykalman":
            state_means = KalmanFilterRegression(series_1, series_2)
            hedge_ratio = -state_means[:, 0]
            spread = series_2 + (series_1 * hedge_ratio)
        else:
            raise ValueError(f"Unknown spread_method: {method!r}. Use 'kalman', 'ols', or 'pykalman'.")
        return spread, hedge_ratio

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
            self.df = self.df.filter(items=target_instruments).dropna()

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
            columns=["corr", "adf", "hurst"])
        for idx in df_corr.index:
            if idx in df_coint.index:
                df_corr_coint_pairs.loc[idx] = [
                    df_corr.loc[idx][0],
                    df_coint.loc[idx]['adf'],
                    df_coint.loc[idx]['hurst'],
                ]

        self.corr_coint_pairs = df_corr_coint_pairs
        try:
            df_corr_coint_pairs.to_csv("%scorr_coint_pairs_%s_%s.csv" %
                                       (self.processed_data_path, self.interval, self.uuid))
        except (IOError, OSError) as e:
            logger.warning("Data couldn't be stored in a static file: %s", e)
        return df_corr_coint_pairs
    
    def _remove_quote_currency_pairs(self, df):
        """Remove pairs where both instruments share the same base currency
        (e.g. BTC_USDT vs BTC_BUSD), keeping only pairs with different bases."""
        pairs_to_keep = []
        for pair_name in df.index:
            parts = pair_name.split('-')
            base1 = parts[0].split('_')[0]
            base2 = parts[1].split('_')[0]
            if base1 != base2:
                pairs_to_keep.append(pair_name)
        return df.loc[pairs_to_keep]

    def validate_pairs_on_timeframe(
        self,
        pairs_df,
        validation_interval,
        validation_closing_prices_container_paths,
    ):
        """Validate discovered pairs on a different timeframe (e.g. 15m after 1h discovery).

        Loads closing prices for the validation interval, then for each pair in
        pairs_df runs: cointegration test, ADF on spread, Hurst exponent and
        half-life.  Returns a DataFrame with the original metrics plus validation
        columns and a ``validated`` boolean.
        """
        # Load validation-interval closing prices (only instruments in pairs_df)
        instruments = set()
        for pair_name in pairs_df.index:
            inst_1, inst_2 = pair_name.split("-")
            instruments.add(inst_1)
            instruments.add(inst_2)

        df_val = pd.DataFrame()
        for path in validation_closing_prices_container_paths:
            with os.scandir('%s%s' % (self.raw_data_path, path)) as entries:
                for entry in entries:
                    instrument = "_".join(entry.name.split("_")[0:2])
                    if instrument not in instruments:
                        continue
                    df = pd.read_csv(
                        '%s%s/%s' % (self.raw_data_path, path, entry.name),
                        index_col="Date",
                    )
                    df.index = pd.to_datetime(df.index)
                    df = df[~df.index.duplicated(keep='first')]
                    df = df[["Close"]].copy()
                    df.columns = [instrument]
                    df_val = pd.concat([df_val, df], axis=1)

        df_val.dropna(inplace=True)

        if df_val.empty:
            logger.warning("No validation data loaded — returning empty result.")
            return pd.DataFrame()

        # Trim to lookback_days if configured
        if self.lookback_days is not None and validation_interval in self.interval_to_days_map:
            n_obs = int(self.lookback_days * self.interval_to_days_map[validation_interval])
            if len(df_val) > n_obs:
                df_val = df_val.iloc[-n_obs:]

        results = []
        for pair_name in pairs_df.index:
            inst_1, inst_2 = pair_name.split("-")

            if inst_1 not in df_val.columns or inst_2 not in df_val.columns:
                logger.warning("Skipping %s — instrument(s) missing in validation data.", pair_name)
                continue

            row = pairs_df.loc[pair_name].to_dict()

            # Cointegration test
            coint_result = coint(df_val[inst_1], df_val[inst_2])
            coint_pval = coint_result[1]

            if coint_pval >= 0.05:
                row.update({'val_coint_pval': coint_pval, 'val_adf': np.nan,
                            'val_hurst': np.nan, 'val_half_life': np.nan,
                            'validated': False})
                results.append((pair_name, row))
                continue

            # Build spread
            spread, _ = self._compute_spread(df_val[inst_1], df_val[inst_2], self.spread_method)

            # ADF on spread
            result_adf = adfuller(spread)
            adf_stat = result_adf[0]
            adf_pval = result_adf[1]

            # Hurst exponent
            hurst_val = self._get_hurst_exponent(np.array(spread))

            # Half-life
            half_life_val = half_life(spread)

            validated = (
                adf_pval < 0.05
                and hurst_val <= 0.5
                and (self.half_life_min is None or half_life_val >= self.half_life_min)
                and (self.half_life_max is None or half_life_val <= self.half_life_max)
            )

            row.update({
                'val_coint_pval': coint_pval,
                'val_adf': adf_stat,
                'val_hurst': hurst_val,
                'val_half_life': half_life_val,
                'validated': validated,
            })
            results.append((pair_name, row))

        if not results:
            return pd.DataFrame()

        result_df = pd.DataFrame.from_dict(
            {name: data for name, data in results}, orient='index')
        return result_df
