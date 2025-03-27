```sql
SELECT healthcare_provider_name, SUM(outpatient_cost + inpatient_cost) AS total_combined_costs
FROM healthcare_services
WHERE service_year = 2014
GROUP BY healthcare_provider_name
ORDER BY total_combined_costs DESC
LIMIT 1;
```