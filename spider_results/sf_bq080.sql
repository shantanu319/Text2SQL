SELECT DATE(trace_timestamp), 
       SUM(CASE WHEN trace_address IS NULL THEN 1 ELSE 0 END) AS external_contracts_created,
       SUM(CASE WHEN trace_address IS NOT NULL THEN 1 ELSE 0 END) AS internal_contracts_created