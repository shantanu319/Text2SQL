SELECT (COUNT(CASE WHEN uninstall_date IS NULL OR uninstall_date > '2018-09-30' THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS percentage