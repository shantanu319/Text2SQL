SELECT state_abbr, AVG(change_percentage) AS national_average_change FROM (
    SELECT state_abbr, 
           SUM(CASE WHEN year = '2018' THEN count ELSE 0 END) - SUM(CASE WHEN year = '2015' THEN count ELSE 0 END) AS change_in_count,
           (SUM(CASE WHEN year = '2018' THEN count ELSE 0 END) / SUM(CASE WHEN year = '2015' THEN count ELSE 0 END)) * 100 AS change_percentage
    FROM homeless_data
    GROUP BY state_abbr
)