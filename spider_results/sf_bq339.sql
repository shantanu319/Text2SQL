SELECT MONTHNAME(date) AS month, ABS(SUM(cumulative_bike_usage_minutes)) AS abs_diff FROM (