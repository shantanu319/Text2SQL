SELECT * FROM patents WHERE kind_code = 'B2' AND grant_date BETWEEN '2015-01-01' AND '2018-12-31' ORDER BY (1 - (SUM(DISTINCT SUBSTRING(citation, 1, 4)) / COUNT(DISTINCT SUBSTRING(citation, 1, 4)))) DESC LIMIT 1