```sql
SELECT 
    TIME,
    INTEREST_NAME,
    COMPOSITION_VALUE
FROM (
    SELECT 
        TIME,
        INTEREST_NAME,
        COMPOSITION_VALUE,
        ROW_NUMBER() OVER (PARTITION BY INTEREST_NAME ORDER BY COMPOSITION_VALUE DESC) AS rn_desc,
        ROW_NUMBER() OVER (PARTITION BY INTEREST_NAME ORDER BY COMPOSITION_VALUE ASC) AS rn_asc
    FROM 
        your_table_name
) ranked_interests
WHERE 
    rn_desc <= 10 OR rn_asc <= 10;
```

Replace `your_table_name` with the actual name of your table. This query uses window functions to rank interests by composition value in both ascending and descending order, then selects the top 10 and bottom 10 categories based on these rankings. The result includes the time, interest name, and composition value for each selected category.