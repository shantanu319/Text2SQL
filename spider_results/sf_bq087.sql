To convert the given text into an SQL query, we need to identify the relevant tables and columns that contain the necessary data. Let's assume we have a table named `search_data` with the following relevant columns:

- `symptom`: The name of the symptom.
- `county`: The name of the county.
- `date`: The date of the search.
- `frequency`: The weekly search frequency.

Here is the SQL query to calculate the overall percentage change in the average weekly search frequency for the symptom 'Anosmia' across the specified counties:

```sql
WITH Jan2019Data AS (
    SELECT 
        COUNT(*) AS total_searches,
        SUM(frequency) AS total_frequency
    FROM 
        search_data
    WHERE 
        symptom = 'Anosmia'
        AND county IN ('Bronx County', 'Queens County', 'Kings County', 'New York County', 'Richmond County')
        AND date BETWEEN '2019-01-01' AND '2019-12-31'
),
Jan2020Data AS (
    SELECT 
        COUNT(*) AS total_searches,
        SUM(frequency) AS total_frequency
    FROM 
        search_data