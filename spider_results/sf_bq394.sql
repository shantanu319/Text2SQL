SELECT YEAR(date) AS year, MONTH(date) AS month, SUM(ABS(AVG(air_temperature) - AVG(wet_bulb_temperature)) + ABS(AVG(dew_point_temperature) - AVG(sea_surface_temperature))) AS total_difference