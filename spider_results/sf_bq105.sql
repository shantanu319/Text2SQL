```sql
SELECT 
    YEAR,
    STATE,
    (ACCIDENTS * 100000 / POPULATION) AS ACCIDENT_RATE_PER_100K,
    RANK() OVER (PARTITION BY YEAR ORDER BY ACCIDENT_RATE_PER_100K DESC) AS RANKING
FROM (
    SELECT 
        YEAR,
        STATE,
        COUNT(*) AS ACCIDENTS,
        SUM(CASE WHEN DISTRACTION_STATUS IN ('Not Distracted', 'Unknown if Distracted', 'Not Reported') THEN 0 ELSE 1 END) AS EXCLUDED_ACCIDENTS,
        SUM(POPULATION) AS POPULATION
    FROM 
        ACCIDENT_DATA
    WHERE 
        YEAR IN (2015, 2016)
        AND DISTRACTION_STATUS NOT IN ('Not Distracted', 'Unknown if Distracted', 'Not Reported')
    GROUP BY 
        YEAR, STATE
) AS EXCLUDED_ACCIDENTS
ORDER BY 
    YEAR, RANKING;
```

This SQL query calculates the number of traffic accidents per 100,000 people caused by driver distraction in each U.S. state for the years 2015 and