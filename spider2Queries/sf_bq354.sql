```sql
SELECT 
    'Standard Acne' AS Condition,
    COUNT(CASE WHEN participant_condition IN ('L70', 'L70.0', 'L70.1', 'L70.2', 'L70.3', 'L70.4', 'L70.5', 'L70.6', 'L70.7', 'L70.8', 'L70.9') THEN 1 ELSE NULL END) * 100 / COUNT(*) AS Percentage
FROM 
    participants
WHERE 
    participant_condition IN ('L70', 'L70.0', 'L70.1', 'L70.2', 'L70.3', 'L70.4', 'L70.5', 'L70.6', 'L70.7', 'L70.8', 'L70.9');

SELECT 
    'Atopic Dermatitis' AS Condition,
    COUNT(CASE WHEN participant_condition IN ('L20', 'L20.0', 'L20.1', 'L20.2', 'L20.3', 'L