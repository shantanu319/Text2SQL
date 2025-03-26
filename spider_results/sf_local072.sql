SELECT c.name AS CountryName,
       COUNT(*) AS TotalEntries,
       MAX(d.date) - MIN(d.date) + 1 AS LongestConsecutivePeriod,
       SUM(CASE WHEN d.city = (SELECT name FROM cities WHERE country_id = c.id AND is_capital = TRUE) THEN 1 ELSE 0 END) / COUNT(*) * 100 AS ProportionFromCapital