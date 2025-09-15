import yfinance as yf


def munger_score(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info

    pe = info.get("trailingPE", None)
    roe = info.get("returnOnEquity", None) * 100 if info.get("returnOnEquity") else None
    operating_cf = info.get("operatingCashflow", None)
    net_income = info.get("netIncomeToCommon", None)
    debt_ratio = info.get("totalDebt", 0) / info.get("totalAssets", 1)

    score = 0
    if pe and pe <= 15:
        score += 1
    if roe and roe >= 15:
        score += 1
    if (
        operating_cf
        and operating_cf > 0
        and abs(operating_cf - net_income) / abs(net_income) < 0.3
    ):
        score += 1
    if debt_ratio <= 0.5:
        score += 1

    return {
        "PE": pe,
        "ROE": roe,
        "经营现金流健康": operating_cf > 0,
        "资产负债率": debt_ratio,
        "总分": score,
    }


print(munger_score("AAPL"))
