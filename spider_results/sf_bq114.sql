SELECT city_1, city_2, ABS(city_1_pm25 - city_2_pm25) AS diff FROM (
    SELECT city,