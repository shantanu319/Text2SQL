SELECT T3.name AS country, (T4.population / T3.population) * 100 AS percentage_under_25 FROM countries AS T3 INNER JOIN populations AS T4 ON T3.id = T4.country_id WHERE T4.year = 2017 ORDER BY percentage_under_25 DESC LIMIT 1;