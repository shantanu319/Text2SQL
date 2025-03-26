SELECT borough, AVG(mean_income) AS avg_mean_income FROM (
  SELECT t.borough, i.mean_income
  FROM trees_data t
  JOIN income_data i ON t.zip = i.zip
  WHERE t.median_income > 0 AND i.mean_income > 0 AND t.borough IS NOT NULL
)