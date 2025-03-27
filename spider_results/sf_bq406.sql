```sql
SELECT 
    "Year",
    "Asian Growth Rate",
    "Black Growth Rate",
    "Latinx Growth Rate",
    "Native American Growth Rate",
    "White Growth Rate",
    "US Women Growth Rate",
    "US Men Growth Rate",
    "Global Women Growth Rate",
    "Global Men Growth Rate"
FROM 
    workforce_growth_rates
WHERE 
    "Year" BETWEEN 2014 AND 2024;
```