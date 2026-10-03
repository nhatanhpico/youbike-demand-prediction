"""Build the teaching dataset from official monthly ZIP archives.
Usage: python build_dataset.py --raw-dir /path/to/archives --output-dir ./output
Required filenames: youbike-202509.zip, youbike-202510.zip, youbike-202511.zip.
Requires Python 3.9+ and NumPy. Download URLs are in sources.csv.
"""
import argparse
import collections
import csv
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np

STATIONS = [
    ('S01', '捷運公館站(2號出口)', 'MRT Gongguan Station (Exit 2)'),
    ('S02', '捷運公館站(3號出口)', 'MRT Gongguan Station (Exit 3)'),
    ('S03', '臺大第一活動中心西南側', 'NTU First Student Activity Center (Southwest)'),
    ('S04', '臺大小福樓東側', 'NTU Xiaofu Building (East)'),
    ('S05', '臺大男一舍前', 'NTU Male Dormitory 1 (Front)'),
    ('S06', '臺大總圖書館西南側', 'NTU Main Library (Southwest)'),
    ('S07', '臺大天文數學館南側', 'NTU Astronomy-Mathematics Building (South)'),
    ('S08', '臺大管理學院一館', 'NTU College of Management Building 1'),
]

def write_csv(path, rows, fields=None):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw-dir', required=True)
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    selected = {s[1] for s in STATIONS}
    counts = collections.Counter()
    city = collections.Counter()
    station_days = collections.Counter()
    audits = []
    for month in ('202509', '202510', '202511'):
        path = Path(args.raw_dir) / f'youbike-{month}.zip'
        audit = dict(month=month, raw_rows=0, excluded_rows=0,
                     rows_with_blank_fields=0, non_hourly_timestamps=0,
                     selected_borrowings=0)
        digest = hashlib.sha256()
        with open(path, 'rb') as f:
            for block in iter(lambda: f.read(1048576), b''):
                digest.update(block)
        audit['zip_sha256'] = digest.hexdigest()
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if not name.lower().endswith('.csv'):
                    continue
                with io.TextIOWrapper(z.open(name), encoding='utf-8-sig') as f:
                    for row in csv.reader(f):
                        audit['raw_rows'] += 1
                        if len(row) != 7:
                            audit['excluded_rows'] += 1
                            continue
                        if any(not x for x in row):
                            audit['rows_with_blank_fields'] += 1
                        t, station = row[0], row[1]
                        # Validate the fields needed to count departures only.
                        if (len(t) != 19 or not station or
                            t[:7] != month[:4] + '-' + month[4:] or
                            t[10] != ' ' or not t[11:13].isdigit() or
                            not 0 <= int(t[11:13]) < 24):
                            audit['excluded_rows'] += 1
                            continue
                        if t[14:] != '00:00':
                            audit['non_hourly_timestamps'] += 1
                        h = t[:13]
                        city[h] += 1
                        if station in selected:
                            counts[station, h] += 1
                            station_days[station, t[:10]] += 1
                            audit['selected_borrowings'] += 1
        audits.append(audit)
        print(json.dumps(audit), flush=True)

    start, end = dt.datetime(2025, 9, 1), dt.datetime(2025, 12, 1)
    times = [start + dt.timedelta(hours=i) for i in range(int((end-start).total_seconds()/3600))]
    assert len(times) == 2184
    rows, station_summary = [], []
    for sid, cn, en in STATIONS:
        values = [counts[cn, t.strftime('%Y-%m-%d %H')] if city[t.strftime('%Y-%m-%d %H')] else None for t in times]
        train_values = [v for t, v in zip(times, values) if t < dt.datetime(2025,11,1) and v is not None]
        q75 = float(np.percentile(train_values, 75))
        assert q75 > 0
        for i, (t, y) in enumerate(zip(times, values)):
            lag1 = values[i-1] if i >= 1 else None
            lag24 = values[i-24] if i >= 24 else None
            history = values[i-24:i] if i >= 24 else []
            mean24 = sum(history)/24 if len(history)==24 and None not in history else None
            split = 'train' if t < dt.datetime(2025,11,1) else 'validation' if t < dt.datetime(2025,11,16) else 'test'
            rows.append(dict(
                forecast_time=t.isoformat()+'+08:00', station_id=sid, station_name=en,
                hour=t.hour, day_of_week=t.weekday(), is_weekend=int(t.weekday()>=5),
                rentals_previous_hour=lag1, rentals_same_hour_previous_day=lag24,
                rentals_mean_previous_24h=mean24,
                target_rentals_next_hour=y,
                high_demand_threshold=q75,
                target_high_demand=int(y>q75) if y is not None else None,
                split=split,
                target_zero_assumed=int(y==0) if y is not None else 0,
                station_day_has_records=int(station_days[cn,t.strftime('%Y-%m-%d')]>0),
                city_hour_has_records=int(city[t.strftime('%Y-%m-%d %H')]>0),
                model_ready=int(None not in (y, lag1, lag24, mean24))))
        station_summary.append(dict(station_id=sid,station_name=en,
            september_rentals=sum(v for (s,h),v in counts.items() if s==cn and h.startswith('2025-09')),
            october_rentals=sum(v for (s,h),v in counts.items() if s==cn and h.startswith('2025-10')),
            november_rentals=sum(v for (s,h),v in counts.items() if s==cn and h.startswith('2025-11')),
            days_with_borrowings=sum(station_days[cn,(start+dt.timedelta(days=i)).strftime('%Y-%m-%d')]>0 for i in range(91)),
            zero_hours_assumed=values.count(0),train_q75=q75))
    rows.sort(key=lambda r:(r['forecast_time'],r['station_id']))
    assert len(rows)==17472
    assert len({(r['station_id'],r['forecast_time']) for r in rows})==len(rows)
    assert sum(r['target_rentals_next_hour'] or 0 for r in rows)==sum(a['selected_borrowings'] for a in audits)
    write_csv(out/'youbike_hourly_english.csv', rows)
    write_csv(out/'station_summary.csv', station_summary)
    write_csv(out/'station_mapping.csv', [dict(station_id=sid,station_name=en,source_station_name=cn) for sid,cn,en in STATIONS])
    write_csv(out/'source_audit.csv', audits)
    daily = [dict(date=(start+dt.timedelta(days=i)).strftime('%Y-%m-%d'),city_borrowings=sum(city[(start+dt.timedelta(days=i,hours=h)).strftime('%Y-%m-%d %H')] for h in range(24))) for i in range(91)]
    write_csv(out/'daily_city_counts.csv', daily)
    result = dict(rows=len(rows),model_ready=sum(r['model_ready'] for r in rows),
        split_counts=dict(collections.Counter(r['split'] for r in rows)),
        zero_hours_assumed=sum(r['target_zero_assumed'] for r in rows),
        city_hours_without_records=sum(not city[t.strftime('%Y-%m-%d %H')] for t in times),
        station_days_without_records=sum(not station_days[cn,(start+dt.timedelta(days=i)).strftime('%Y-%m-%d')] for _,cn,_ in STATIONS for i in range(91)),
        stations=station_summary)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    (out/'build_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

if __name__=='__main__':
    main()
