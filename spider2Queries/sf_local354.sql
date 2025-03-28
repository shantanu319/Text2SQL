```sql
SELECT DISTINCT driver_name
FROM f1_drivers
WHERE year BETWEEN 1950 AND 1959
GROUP BY driver_name, constructor_name
HAVING COUNT(DISTINCT race_round) >= 2
AND (race_round = 1 OR race_round = 3)
AND (race_round = 1 AND constructor_name = constructor_name_in_last_race)
AND (race_round = 3 AND constructor_name = constructor_name_in_first_race);
```

Note: The above query assumes that there is a table named `f1_drivers` with columns `driver_name`, `year`, `constructor_name`, and `race_round`. It also assumes that there is another column `constructor_name_in_last_race` and `constructor_name_in_first_race` that stores the constructor names of the last and first races for each driver. You may need to adjust the query based on your actual database schema. Additionally, the query assumes that the race rounds are numbered sequentially starting from 1. If your race round numbering starts from a different number, you will need to adjust the query accordingly.