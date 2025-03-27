```sql
CREATE TABLE labeled_dataset AS
SELECT 
    state_number,
    vehicle_body_type,
    number_of_drunk_drivers,
    day_of_week,
    hour_of_crash,
    CASE WHEN work_zone IS NOT 'None' THEN 1 ELSE 0 END AS work_zone_indicator,
    AVG(CASE WHEN travel_speed <= 151 AND speed_limit <= 80 THEN ABS(travel_speed - speed_limit) ELSE NULL END) / COUNT(*) AS avg_speed_difference,
    CASE 
        WHEN avg_speed_difference < 20 THEN 0
        WHEN avg_speed_difference < 40 THEN 1
        WHEN avg_speed_difference < 60 THEN 2
        WHEN avg_speed_difference < 80 THEN 3
        ELSE 4
    END AS speed_difference_level,
    CASE 
        WHEN COUNT(DISTINCT person_id) > 1 THEN 1
        ELSE 0
    END AS multiple_persons_label
FROM 
    nhtsa_traffic_fatality_data
WHERE 
    COUNT(DISTINCT person_id) > 1
GROUP BY 
    state_number,
    vehicle_body_type,
    number_of_drunk_drivers,
    day_of_week,
    hour_of_crash