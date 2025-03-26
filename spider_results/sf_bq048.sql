SELECT complaint_type, ROUND(pearson_correlation_coefficient, 4) AS correlation FROM (
    SELECT complaint_type, 
           AVG(CASE WHEN year = 2011 THEN daily_average_wind_speed ELSE NULL END) - AVG(CASE WHEN year = 2020 THEN daily_average_wind_speed ELSE NULL END) AS difference_in_wind_speed,
           COUNT(*) AS total_requests,
           SUM(daily_complaint_proportion * daily_average_wind_speed) / SUM(daily_complaint_proportion) AS weighted_sum_of_daily_average_wind_speeds
    FROM (
        SELECT complaint_type, 
               YEAR(date) AS year, 
               DAYOFYEAR(date) AS day_of_year, 
               AVG(wind_speed) AS daily_average_wind_speed, 
               COUNT(*) AS total_complaints, 
               COUNT(*) * 1.0 / (COUNT(*) + COUNT(*)) AS daily_complaint_proportion
        FROM complaints
        WHERE station_id = '744860'
        GROUP BY complaint_type, year, day_of_year
    ) AS subquery
    GROUP BY complaint_type
    HAVING COUNT(*) > 3000