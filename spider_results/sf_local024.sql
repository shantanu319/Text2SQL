SELECT c.name AS Country, AVG(p.average) AS Average
FROM Player p
JOIN Match m ON p.match_id = m.id
JOIN Country c ON m.country_id = c.id
GROUP BY c.name