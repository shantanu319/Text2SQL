SELECT s.state_code, e.evaluation_group, s.state_name, MAX(t.total_acres) AS max_total_acres
FROM states s
JOIN evaluations e ON s.state_code = e.state_code
WHERE e.evaluation_type = 'EXPCURR'
GROUP BY s.state_code, e.evaluation_group