SELECT DATE FROM incidents WHERE incident_type = 'PUBLIC INTOXICATION' AND YEAR(date) = 2016 ORDER BY z_score DESC LIMIT 1, 1