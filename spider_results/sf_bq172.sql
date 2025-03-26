SELECT state, SUM(total_claim_count) as total_claim_count, SUM(total_drug_cost) as total_drug_cost FROM drug_prescriptions WHERE state IN ('New York', 'California', 'Texas', 'Florida', 'Illinois') AND year = 2014 GROUP BY state ORDER BY total_claim_count DESC LIMIT 5