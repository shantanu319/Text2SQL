SELECT region FROM gdp_data WHERE constant_2015_usd = ( SELECT MAX(constant_2015_usd) FROM gdp_data ) GROUP BY region ORDER BY median_gdp DESC LIMIT 1;