SELECT team_name, COUNT(*) as num_wins FROM games GROUP BY team_name ORDER BY num_wins DESC LIMIT 10