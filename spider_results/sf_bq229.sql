```sql
SELECT 
    COUNT(CASE WHEN label = '/m/01yrx' THEN 1 ELSE NULL END) AS cat_count,
    COUNT(*) - COUNT(CASE WHEN label = '/m/01yrx' THEN 1 ELSE NULL END) AS other_count
FROM 
    images;
```

This SQL query counts the number of image URLs that have the label `/m/01yrx` (indicating they are categorized as 'cat') and those that do not have this label (categorized as 'other'). The `COUNT` function is used with conditional logic to achieve this. The first part of the `COUNT` function counts only those rows where the label matches `/m/01yrx`, while the second part subtracts this count from the total number of rows to get the count of 'other' categories.