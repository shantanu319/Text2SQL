```sql
SELECT 
    COUNT(*) * 100 / (SELECT COUNT(DISTINCT STATION) FROM gsod WHERE YEAR = 2022) AS percentage
FROM 
    gsod
WHERE 
    YEAR = 2022
    AND TAVG IS NOT NULL
    AND MAXT IS NOT NULL
    AND MINT IS NOT NULL
    AND TAVG != 9999.9
    AND MAXT != 9999.9
    AND MINT != 9999.9
    AND USAF != '999999'
    AND (
        SELECT COUNT(*)
        FROM gsod AS s2
        WHERE s2.STATION = gsod.STATION
            AND s2.YEAR = gsod.YEAR
            AND s2.TAVG IS NOT NULL
            AND s2.MAXT IS NOT NULL
            AND s2.MINT IS NOT NULL
            AND s2.TAVG != 9999.9
            AND s2.MAXT != 9999.9
            AND s2.MINT != 9999.9
            AND s2.USAF != '