```sql
SELECT T1.stage_name, T2.last_name
FROM entertainers AS T1
JOIN customers AS T2 ON (
    (T1.style_strength_1 = T2.style_preference_1 AND T1.style_strength_2 = T2.style_preference_2) 
    OR 
    (T1.style_strength_1 = T2.style_preference_2 AND T1.style_strength_2 = T2.style_preference_1)
)
WHERE T1.style_strength_count <= 3 AND T2.style_preference_count <= 3;
```

This query joins the `entertainers` and `customers` tables based on the condition that the first and second style preferences of the customers match the first and second style strengths of the entertainers (or in reverse order), ensuring that only pairs with up to three style strengths are considered. It then selects the entertainer's stage name and the customer's last name from the joined table. The `style_strength_count` and `style_preference_count` columns are assumed to be present in the respective tables to filter out entertainers and customers with more than three style strengths or preferences. Adjust the column names as necessary to fit your actual database schema.