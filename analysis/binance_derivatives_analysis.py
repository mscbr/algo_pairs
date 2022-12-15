from modules.IterativeBacktester import IterativeBacktester
from modules.CointAnalyzer import Coint_Analyzer
import os
import sys
parent_dir = os.path.abspath('..')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


backtester = IterativeBacktester(
    pair=["ENJUSDT_FUTURES", "ZENUSDT_FUTURES"],
    ticker_interval="30m",
    raw_data_path="raw_data/BINANCE_DOWNLOADER_TESTS/binance_historical_30m_FUTURES_6b1f260d-1873-417a-ae21-1317557f5930/",
    save_dir_path="processed_data/BACKTESTER_TEST/",
    opt_cache_path=None,
    tc=0.0006
)

backtester.optimize_pair(
    windows_range=(100, 1000, 190),
    long_entries_range=(99, 100, 1),
    short_entries_range=(2, 3, 1),
    sl_range=(-0.20, -0.03, 0.08),
    tp_range=(0.02, 0.05, 0.01),
    report=True,
    charts=True,
)

# backtester.backtest_pair(
#   z_long_perc=1,
#   z_short_perc=99,
#   tp=0.02,
#   sl=-0.05,
#   charts=True,
#   silent=False,
#   window=600,
#   start_date="2021-01-01 00:00:00",
#   stationarity_testing=False,
# )
