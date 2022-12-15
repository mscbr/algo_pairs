import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from math import sqrt
from itertools import product
from tools.kalman_filters.py_kalman import KalmanFilterAverage, KalmanFilterRegression
from tools.kalman_filters.vanilla_kalman import VanillaKalmanFilter
from tools.metrics import calculate_annualized_mean, calculate_annualized_std, calculate_max_dd
from tools.spread_features import get_current_df_zscore
from tools.logging import get_pos_neg_color

import uuid
import os
import sys
from pathlib import Path

parent_dir = os.path.abspath('..')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


class IterativeBacktester:
    def __init__(
        self,
        pair,
        ticker_interval,
        raw_data_path,
        save_dir_path="../processed_data/",
        tc=0.0006
    ):

        # global settings
        self.pair = pair
        self.pair_str = "%s-%s" % (pair[0], pair[1])
        self.save_dir_path = save_dir_path
        self.raw_data_path = raw_data_path
        self.ticker_interval = ticker_interval
        self.uuid = str(uuid.uuid4())

        # current backtest settings
        self.tp = None
        self.sl = None
        self.window = None
        self.iter_start = None
        self.z_short_perc = None
        self.z_long_perc = None
        self.param_str = None
        self.chart_spread = None

        # current optimization
        self.results = None

        # data
        self.tp_year = None
        self.data = None
        self.result = None
        self.opt_overview = None
        self.tc = tc

        self._get_data()

        if not os.path.exists(self.save_dir_path):
            Path(self.save_dir_path).mkdir(parents=True, exist_ok=True)

    def __repr__(self):
        return "Pairs trading backtester"

    def _get_data(self):
        entries = []
        for instrument in self.pair:
            entries.append("%s_%s.csv" % (instrument, self.ticker_interval))
        # removing duplicates
        entries = list(set(entries))

        df_closings = pd.DataFrame()
        for entry in entries:
            instrument = "_".join(entry.split("_")[0:-1])
            df = pd.read_csv('%s/%s' % (self.raw_data_path, entry),
                             index_col="Date", parse_dates=["Date"])
            df = df[["Close"]].copy()
            df.columns = [instrument]

            df["%s_returns" % instrument] = df[instrument] / \
                df[instrument].shift(1)

            df["%s_returns_log" % instrument] = np.log(df[instrument] /
                                                       df[instrument].shift(1))

            df_closings = pd.concat([df_closings, df], axis=1)

        df_closings.dropna(inplace=True)
        df_closings = df_closings.T.drop_duplicates().T
        self.data = df_closings
        self.tp_year = (
            self.data.shape[0] / ((self.data.index[-1] - self.data.index[0]).days / 365.25))

        # wholesample spread for charting purposes
        state_means = KalmanFilterRegression(
            df_closings[self.pair[0]], df_closings[self.pair[1]])

        hr = - state_means[:, 0]
        spread = df_closings[self.pair[1]] + (df_closings[self.pair[0]] * hr)
        self.chart_spread = spread
        # test
        # self.chart_spread = df_closings[self.pair[0]
        #                                 ] / df_closings[self.pair[1]]

    def optimize_pair(
        self,
        windows_range=(400, 410, 80),
        long_entries_range=(4, 5, 1),
        short_entries_range=(97, 98, 1),
        sl_range=(-0.15, -0.14, 0.03),
        tp_range=(0.02, 0.03, 0.01),
        use_pykalman=False,
        avg_kalman=False,
        silent=True,
        report=False,
        charts=False,
        cache_process=False,
        cache_path=None,
    ):
        long_entries_range = np.arange(*long_entries_range)
        short_entries_range = np.arange(*short_entries_range)
        sl_range = [round(flt, 2) for flt in np.arange(*sl_range)]
        tp_range = [round(flt, 2) for flt in np.arange(*tp_range)]
        windows_range = np.arange(*windows_range)
        combinations = list(product(
            long_entries_range,
            short_entries_range,
            sl_range,
            tp_range,
            windows_range))

        start_date = self.data.index[windows_range[-1]]

        if cache_path:
            opt_overview = pd.read_csv(cache_path, index_col=0)
            cache_end = len(opt_overview.dropna().index)
            combinations = combinations[cache_end:]
        else:
            indexes = ["_".join([str(num) for num in index_arr])
                       for index_arr in combinations]
            opt_overview = pd.DataFrame(index=indexes, data=np.array(combinations), columns=[
                                        "le", "se", "sl", "tp", "window"])
            opt_overview["net"] = np.NaN
            opt_overview["max_dd"] = np.NaN

        report_dir = None
        if report:
            report_dir = "%s_%s/" % ("_".join(self.pair), self.ticker_interval)

        if cache_process and not cache_path:
            cache_path = "%s%soptimization_overview_cache.csv" % (
                self.save_dir_path, report_dir)
            cache_end = 0

        if not silent:
            print("STARTING DATE: ", start_date)
            print("WINDOW | NET | MAX_DD")
            print(25*"=")

        for idx, comb in enumerate(combinations):
            print(idx, "/", len(combinations))
            self.backtest_pair(
                window=comb[4],
                z_long_perc=comb[0],
                z_short_perc=comb[1],
                tp=comb[3],
                sl=comb[2],
                use_pykalman=use_pykalman,
                avg_kalman=avg_kalman,
                start_date=start_date,
                report_dir=report_dir,
                charts=charts)

            # try:
            #     sharpe = ((self.result.net_cum.mean() / self.result.net_cum.std()) * sqrt(self.tp_year))
            # except ZeroDivisionError:
            #     sharpe = 0.0

            max_dd, _, _ = calculate_max_dd(self.result.net_cum)

            # [idx, [net, max_dd]]
            opt_overview.iloc[idx+cache_end, [-2, -1]
                              ] = [round(self.result.net_cum[-1], 3), round(max_dd, 3)]

            if cache_path:
                try:
                    opt_overview.to_csv(cache_path)
                except:
                    print("Couldn't save test iter opt to file")

            if not silent:
                print(comb, ": ", self.result.net_cum[-1], max_dd)

        opt_overview.sort_values(['net'], inplace=True)

        self.opt_overview = opt_overview

        try:
            opt_overview.to_csv("%s%s%s_optimization_overview_%s.csv" % (
                self.save_dir_path, report_dir, "_".join(self.pair), self.uuid))
        except:
            print("Couldn't save test iter opt to file")

    def get_Kalman_spread(self, series1, series2, avg_kalman=False):
        if avg_kalman is True:
            state_means = KalmanFilterRegression(
                KalmanFilterAverage(series1), KalmanFilterAverage(series2))
        else:
            state_means = KalmanFilterRegression(series1, series2)

        hedge_ratio = - state_means[:, 0]
        spread = series1 + (series2 * hedge_ratio)

        return spread

    def backtest_pair(
        self,
        window=48,
        z_long_perc=5,
        z_short_perc=95,
        tp=0.01,
        sl=-0.01,
        use_pykalman=False,
        avg_kalman=False,
        rolling_z_window=None,
        stationarity_testing=False,
        start_date=None,
        silent=True,
        report_dir="./",
        charts=False
    ):
        stationary_spread = True
        self.tp = tp
        self.sl = sl
        self.window = window
        self.z_short_perc = z_short_perc
        self.z_long_perc = z_long_perc
        self.param_str = "w_{}_p_{}_{}_sltp_{}_{}".format(
            window, z_long_perc, z_short_perc, str(sl).replace(".", "d"), str(tp).replace(".", "d"))

        inst1 = self.pair[0]
        inst2 = self.pair[1]
        series1 = self.data[inst1].to_numpy()
        series2 = self.data[inst2].to_numpy()
        returns1 = self.data["%s_returns" % inst1].to_numpy()
        returns2 = self.data["%s_returns" % inst2].to_numpy()

        signals = np.empty((0, 1), dtype="float64")
        inst1_position = np.empty((0, 1), dtype="float64")
        inst2_position = np.empty((0, 1), dtype="float64")
        spread_record = np.empty((0, 1), dtype="float64")
        zscore = np.empty((0, 1), dtype="float64")
        z_long_record = np.empty((0, 1), dtype="float64")
        z_short_record = np.empty((0, 1), dtype="float64")
        gross_returns = np.empty((0, 1), dtype="float64")
        net_returns = np.empty((0, 1), dtype="float64")

        signal = 0
        old_signal = 0
        current_return = 0
        position1 = 0
        position2 = 0

        self.iter_start = window
        if start_date is not None:
            try:
                self.iter_start = self.data.index.get_loc(start_date)
            except:
                print("Couldn't find start_date: %s" % start_date)

        # moving through the sample
        for t in range(self.iter_start, len(self.data)-1):
            # because of sampling index is off when sample[t]
            sample_series1 = series1[t-window:t+2]
            sample_series2 = series2[t-window:t+2]

            old_signal = signal
            old_position1 = position1
            old_position2 = position2
            gross = 0
            net = 0

            if use_pykalman:
                spread = self.get_Kalman_spread(
                    sample_series1, sample_series2, avg_kalman=avg_kalman)
            else:
                mkf = VanillaKalmanFilter(delta=1e-4, R=2)
                spread = mkf.regression(sample_series1, sample_series2)

            # STATIONARY TESTING (ADF 10%)
            if stationarity_testing:
                adf = adfuller(spread)
                stationary_spread = adf[0] < adf[4]["10%"]

            if rolling_z_window:
                current_z_score, z_score_series = get_current_df_zscore(
                    spread, rolling_window=rolling_z_window)
            else:
                z_score_series = (spread-spread.mean())/spread.std()
                current_z_score = z_score_series[-1]

            z_percentiles = np.percentile(
                z_score_series, [z_long_perc, z_short_perc])
            z_long = z_percentiles[0]
            z_short = z_percentiles[1]

            if old_signal == 0:
                if current_z_score < z_long:
                    signal = 1
                elif current_z_score > z_short:
                    signal = -1
            elif old_signal != 0:
                if current_return >= tp:
                    signal = 0
                elif current_return <= sl:
                    signal = 0

            if stationary_spread != True and old_signal == 0:
                signal = 0

            position1 = -signal
            position2 = signal

            # check out what type of index is within the sample & if usage of "window+1" is legit
            gross = position1*returns1[t+1] + position2*returns2[t+1]
            net = gross - self.tc * \
                (abs(position1 - old_position1) + abs(position2 - old_position2))
            if signal == old_signal:
                current_return = (1+current_return)*(1+net)-1
            else:
                current_return = net

            inst1_position = np.vstack((inst1_position, [position1]))
            inst2_position = np.vstack((inst2_position, [position2]))
            signals = np.vstack((signals, [signal]))
            spread_record = np.vstack(
                (spread_record, [spread[-1]]))  # double-check on this
            zscore = np.vstack((zscore, [current_z_score]))
            z_long_record = np.vstack((z_long_record, [z_long]))
            z_short_record = np.vstack((z_short_record, [z_short]))
            gross_returns = np.vstack((gross_returns, [gross]))
            net_returns = np.vstack((net_returns, [net]))

            # interface: reporting daily positions and realised returns
            if silent is True:
                continue

            # current stats
            print('\033[1m \033[37m', 100 * "=")
            if signal == 1:
                print('LONG: ', inst2, ' SHORT: ',
                      inst1, " z-score entry", z_long)
            elif signal == -1:
                print('LONG: ', inst1, ' SHORT: ',
                      inst2, " z-score entry", z_short)
            cum_net = round(np.prod(1+net_returns)*100-100, 2)
            print(
                "\033[0m", t, "/", series1.shape[0]-1,
                " \033%sCUM RET:" % get_pos_neg_color(
                    cum_net), str(cum_net) + '% ',
                "\033%sCURENT NET:" % get_pos_neg_color(
                    current_return), str(round(current_return, 3)) + '% ',
                "\033%sPOS:" % get_pos_neg_color(signal), signal
            )

        gross_cum = np.reshape(np.cumprod(1+gross_returns), (-1, 1))
        net_cum = np.reshape(np.cumprod(1+net_returns), (-1, 1))

        output = np.concatenate((
            inst1_position,
            inst2_position,
            signals,
            spread_record,
            zscore,
            z_long_record,
            z_short_record,
            gross_returns,
            net_returns,
            gross_cum,
            net_cum,
        ), axis=1)

        df = pd.DataFrame(output, columns=[
            "%s_position" % inst1, "%s_position" % inst2, "signals", "spread",
            "zscore", "z_long", "z_short", "gross_returns",
            "net_returns", "gross_cum", "net_cum"], index=self.data.index.copy()[self.iter_start+1:])

        self.result = df

        if not report_dir:
            return
        save_dir_path = self.save_dir_path
        if report_dir:
            directory = "%s%s" % (self.save_dir_path, report_dir)
            if not os.path.exists(directory):
                os.mkdir(directory)
            save_dir_path = directory

        if charts:
            self._get_backtest_charts(
                save_dir_path=save_dir_path, silent=silent)
        self._get_backtest_report(save_dir_path=save_dir_path, silent=silent)
        self._save_result_to_file(save_dir_path=save_dir_path)

    def _get_backtest_report(self, save_dir_path=None, silent=False):
        # add winning positions amount & loosing positions amount
        ''' Calculates and prints various Performance Metrics.
        '''
        if save_dir_path is None and silent != False:
            return

        data = self.result.copy()

        strategy_net = round(data["net_cum"].iloc[-1], 2)
        strategy_gross = round(data["gross_cum"].iloc[-1], 2)
        max_dd, max_ddd, i = calculate_max_dd(data["net_cum"])
        ann_mean = round(calculate_annualized_mean(
            self.tp_year, data["net_cum"]), 2)
        ann_std = round(calculate_annualized_std(
            self.tp_year, data["net_cum"]), 2)
        sharpe = round(
            ((data["net_cum"].mean() / data["net_cum"].std()) * sqrt(self.tp_year)), 2)

        entry_exit_idxs = self._get_positions()
        pos = []
        neg = []
        for idxs in entry_exit_idxs:
            pnl = data["net_cum"].loc[idxs[1]] - \
                data["net_cum"].shift(1).loc[idxs[0]]
            if pnl < 0:
                neg.append(pnl)
            else:
                pos.append(pnl)

        if silent is False:
            print(100 * "=")
            print(
                "SPREAD TRADING | INSTRUMENTS = {}\nZ-PERCENTILES = {} | SL/TP = {} | WINDOW = {}".format(
                    self.pair[0]+" & "+self.pair[1], [self.z_long_perc, self.z_short_perc], [self.sl, self.tp], self.window))
            print(100 * "-")
            print("PERFORMANCE MEASURES:")
            print("Net:         {} | {}".format(
                strategy_net, round((strategy_net - 1)*100, 2)))
            print("Gross:       {} | {}".format(
                strategy_gross, round((strategy_gross - 1)*100, 2)))
            print(38 * "-")
            print("Annualized Mean:             {}".format(ann_mean))
            print("Annualized Std:              {}".format(ann_std))
            print("Sharpe Ratio:                {}".format(sharpe))
            print("Max Draw Down:               {}".format(round(max_dd, 2)))
            print("Max Draw Down Duration:      {}".format(max_ddd))
            print(38 * "-")
            print("{} winning positions w/ avg: {}".format(len(pos),
                  round(np.mean(pos), 2)))
            print("{} loosing positions w/ avg: {}".format(len(neg),
                  round(np.mean(neg), 2)))

            print(100 * "=")

        if save_dir_path is not None:
            f = open("%s%s_report_%s_%s.txt" % (
                save_dir_path, self.pair[0]+"_"+self.pair[1], self.param_str, self.uuid), "w")

            report = 100 * "=" + "\n"
            report += "SPREAD TRADING | INSTRUMENTS = {}\nZ-PERCENTILES = {} | SL/TP = {} | WINDOW = {}\n".format(
                self.pair[0]+" & "+self.pair[1], [self.z_long_perc, self.z_short_perc], [self.sl, self.tp], self.window)
            report += 100 * "-" + "\n"
            report += "PERFORMANCE MEASURES:\n"
            report += "Net:         {} | {}".format(
                strategy_net, round((strategy_net - 1)*100, 2))
            report += "Gross:       {} | {}".format(
                strategy_gross, round((strategy_gross - 1)*100, 2))
            report += 38 * "-" + "\n"
            report += "Annualized Mean:             {}\n".format(ann_mean)
            report += "Annualized Std:              {}\n".format(ann_std)
            report += "Sharpe Ratio:                {}\n".format(sharpe)
            report += "Max Draw Down:               {}\n".format(
                round(max_dd, 2))
            report += "Max Draw Down Duration:      {}\n".format(max_ddd)
            report += 38 * "-" + "\n"
            report += "{} winning positions w/ avg: {}\n".format(
                len(pos), round(np.mean(pos), 2))
            report += "{} loosing positions w/ avg: {}\n".format(
                len(neg), round(np.mean(neg), 2))
            report += 100 * "="

            f.write(report)
            f.close()

    def _get_backtest_charts(self, save_dir_path=None, silent=False):
        if save_dir_path is None and silent != False:
            return

        result = self.result.copy()
        fig, axs = plt.subplots(
            3, 1, gridspec_kw={'height_ratios': [2, 1, 2]},
            sharex=True, figsize=(18, 20))
        fig.patch.set_facecolor('lavender')
        fig.suptitle("PAIR: %s | INT: %s\nWINDOW: %s" % (
            self.pair[0]+"-"+self.pair[1], self.ticker_interval, self.window), fontsize=24)

        axs[0].plot(result["net_cum"], color="blue", label="NET")
        axs[0].plot(result["gross_cum"], color="orange", label="GROSS")
        axs[0].grid()
        axs[0].legend()
        axs[0].set_xlabel("PNL | NET: {}% | GROSS: {}%".format(
            round((result.net_cum[-1] - 1)*100, 2), round((result.gross_cum[-1] - 1)*100, 2)), fontsize=22)

        axs[1].plot(result["zscore"], color="black", label="ZSCORE")
        axs[1].plot(result["z_long"], color="green", label="ZLONG")
        axs[1].plot(result["z_short"], color="red", label="ZSHORT")
        axs[1].grid()
        axs[1].legend()
        axs[1].set_xlabel("ENTRIES", fontsize=22)

        spread = self.chart_spread.iloc[self.iter_start+1:].copy()
        long = spread.copy()
        short = spread.copy()
        long[result.signals != 1] = np.NaN
        short[result.signals != -1] = np.NaN
        axs[2].plot(spread, color="slategrey", label="SPREAD")
        axs[2].plot(long, color="green", label="LONG")
        axs[2].plot(short, color="red", label="SHORT")
        axs[2].grid()
        axs[2].legend()
        axs[2].set_xlabel("POSITIONS", fontsize=22)

        if save_dir_path is not None:
            plt.savefig("%s%s_charts_%s_%s.png" % (
                save_dir_path, self.pair[0]+"-"+self.pair[1], self.param_str, self.uuid))

    def _save_result_to_file(self, save_dir_path):
        self.result.to_csv("%s%s_result_%s_%s.csv" % (
            save_dir_path, self.pair[0]+"_"+self.pair[1], self.param_str, self.uuid))

    def _get_positions(self):
        ques = self.result.loc[
            ((self.result.signals == 0) & (self.result.signals.shift(1) != 0))
            | ((self.result.signals != 0) & (self.result.signals.shift(1) == 0))]
        grouped_idx = []
        for i, index in enumerate(ques.index):
            if i % 2 == 1 and i+1 < len(ques.index):
                grouped_idx.append([index, ques.index[i+1]])

        # [[entry_timestamp, exit_timestamp], ...]
        return grouped_idx
