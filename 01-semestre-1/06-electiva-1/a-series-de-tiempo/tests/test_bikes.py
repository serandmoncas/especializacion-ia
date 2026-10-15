import pandas as pd
from applied_ts.bikes import load_hourly

HEADER = (
    "instant,dteday,season,yr,mnth,hr,holiday,weekday,workingday,weathersit,"
    "temp,atemp,hum,windspeed,casual,registered,cnt\n"
)


def test_fills_gaps_and_flags_long_outages(tmp_path):
    rows = []
    hours = [0, 1, 3, *range(10, 24)]  # falta la 2 (hueco corto) y 4–9 (bloque de 6 h)
    for i, hr in enumerate(hours):
        rows.append(f"{i},2011-01-01,1,0,1,{hr},0,6,0,1,0.2,0.2,0.8,0.1,1,4,5")
    (tmp_path / "hour.csv").write_text(HEADER + "\n".join(rows) + "\n")
    df = load_hourly(tmp_path)
    assert len(df) == 24 and df.index.freq == "h"
    assert df.loc[pd.Timestamp("2011-01-01 02:00"), "cnt"] == 0
    assert not df.loc[pd.Timestamp("2011-01-01 02:00"), "cierre"]
    assert df.loc["2011-01-01 04:00":"2011-01-01 09:00", "cierre"].all()
    assert df["cierre"].sum() == 6
