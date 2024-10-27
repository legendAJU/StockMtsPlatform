import baostock as bs
import pandas as pd
from datetime import datetime, timedelta
from multiprocessing import Pool
import chinese_calendar as calender

# 定义查询每个时间区间数据的函数
def query_hs300_stocks_in_range(date_range):
    start_date, end_date = date_range
    all_hs300_stocks = pd.DataFrame()

    # 登陆系统
    

    current_date = start_date
    lg = bs.login()
    while current_date <= end_date:
        if lg.error_code != '0':
            print(f"Login failed with error: {lg.error_msg}")
            return pd.DataFrame()
        date_str = current_date.strftime("%Y-%m-%d")
        if calender.is_holiday(current_date):
            current_date += timedelta(days=1)
            continue
        print(f"Querying date {current_date.strftime('%Y-%m-%d')}")
        rs = bs.query_hs300_stocks(date=date_str)
        if rs.error_code == '0':
            hs300_stocks = []
            while rs.next():
                stock_data = rs.get_row_data()
                stock_code = stock_data[1]  # 假设股票代码在第二列
                # 查询股票的每日信息
                k_rs = bs.query_history_k_data_plus(stock_code,
                                                    "date,code,open,close,high,low,volume",
                                                    start_date=date_str, end_date=date_str,
                                                    frequency="d", adjustflag="3")
                if k_rs.error_code == '0':
                    k_data = k_rs.get_row_data()
                    if k_data != []:
                        hs300_stocks.append(k_data)
                        
            if hs300_stocks:
                result = pd.DataFrame(hs300_stocks, columns=['date',
                                                             'code',
                                                             'open','close','high','low','volume'])
                all_hs300_stocks = pd.concat([all_hs300_stocks, result])
        else:
            print(f"Error querying date {date_str}: {rs.error_msg}")
        current_date += timedelta(days=1)

    # 登出系统
    #bs.logout()
    return all_hs300_stocks

# 将时间范围划分为多个子区间
def split_date_range(start_date, end_date, n_splits):
    delta = (end_date - start_date) // n_splits
    ranges = []
    for i in range(n_splits):
        range_start = start_date + i * delta
        if i == n_splits - 1:
            range_end = end_date
        else:
            range_end = range_start + delta - timedelta(days=1)
        ranges.append((range_start, range_end))
    return ranges

if __name__ == "__main__":
    # 设置起始日期和结束日期
    start_date = datetime.strptime("2008-01-01", "%Y-%m-%d")
    end_date = datetime.strptime("2023-01-01", "%Y-%m-%d")

    # 将时间范围划分为4个部分
    date_ranges = split_date_range(start_date, end_date, n_splits=64)

    # 使用多进程并行查询每个时间区间的数据
    with Pool(processes=64) as pool:
        results = pool.map(query_hs300_stocks_in_range, date_ranges)

    # 合并所有进程的结果
    all_hs300_stocks = pd.concat(results)

    # 结果集输出到csv文件
    all_hs300_stocks.to_csv("/home/cseadmin/mz/StockMtsPlatform_v2/get_csi_data/all_hs300_stocks.csv", encoding="gbk", index=False)

    print("Data extraction completed.")
