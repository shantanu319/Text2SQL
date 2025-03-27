```sql
SELECT 
    complaint_type,
    ROUND(positive_correlation_coefficient, 4) AS positive_correlation,
    ROUND(negative_correlation_coefficient, 4) AS negative_correlation
FROM (
    SELECT 
        complaint_type,
        SUM(total_requests) OVER (PARTITION BY complaint_type) AS total_requests,
        AVG(daily_average_wind_speed) OVER (PARTITION BY complaint_type) AS avg_wind_speed,
        COUNT(*) OVER (PARTITION BY complaint_type) AS total_complaints,
        SUM(complaint_proportion * daily_average_wind_speed) OVER (PARTITION BY complaint_type) AS sum_product,
        SUM(complaint_proportion) OVER (PARTITION BY complaint_type) AS sum_complaint_proportions,
        SUM(complaint_proportion * daily_average_wind_speed) OVER (PARTITION BY complaint_type) - SUM(complaint_proportion) OVER (PARTITION BY complaint_type) * AVG(daily_average_wind_speed) OVER (PARTITION BY complaint_type) AS covariance,
        SQRT(SUM((complaint_proportion * daily_average_wind_speed) OVER (PARTITION BY complaint_type) - SUM(complaint_proportion) OVER (PARTITION BY complaint_type) * AVG(daily_average_wind_speed) OVER