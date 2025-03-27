```sql
SELECT 
    DATE_FORMAT(T1.timestamp, '%Y-%m-%d %H:%i') AS timestamp,
    T1.node_id,
    T1.hourly_price_day_ahead,
    T1.hourly_price_real_time,
    T2.load_forecast,
    T2.actual_load,
    T3.wind_forecast,
    T3.wind_actual,
    T4.solar_forecast,
    T4.solar_actual,
    CASE 
        WHEN T1.hourly_price_day_ahead > T1.hourly_price_real_time THEN 'High'
        ELSE 'Low'
    END AS price_peak_classification,
    CASE 
        WHEN T2.load_forecast > T2.actual_load THEN 'High'
        ELSE 'Low'
    END AS load_peak_classification,
    CASE 
        WHEN T3.wind_forecast > T3.wind_actual THEN 'High'
        ELSE 'Low'
    END AS wind_peak_classification,
    CASE 
        WHEN T4.solar_forecast > T4.solar_actual THEN 'High'
        ELSE 'Low'
    END AS solar_peak_classification,
    (T2.load_forecast - T2.actual_load) AS net_load
FROM 
    ercot_hourly_prices AS T1
JOIN