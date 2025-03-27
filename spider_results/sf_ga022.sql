To calculate the weekly customer retention rate for new customers who first used your app within the first week of September 2018, you can use the following SQL query. This assumes you have a table named `customer_events` with columns such as `user_id`, `event_type`, `event_date`, and `timezone`.

```sql
WITH NewCustomers AS (
    SELECT DISTINCT user_id
    FROM customer_events
    WHERE event_type = 'first_open'
      AND event_date >= '2018-09-01'::date
      AND event_date < '2018-09-08'::date
),
WeeklyRetention AS (
    SELECT 
        nc.user_id,
        EXTRACT(WEEK FROM ce.event_date) - EXTRACT(WEEK FROM nc.first_open_event_date) AS week_number,
        COUNT(DISTINCT ce.user_id) OVER (PARTITION BY nc.user_id ORDER BY EXTRACT(WEEK FROM ce.event_date)) AS retained_users
    FROM 
        NewCustomers nc
    JOIN 
        customer_events ce ON nc.user_id = ce.user_id
    WHERE 
        ce.event_type IN ('first_open', 'event_1', 'event_2', 'event_3')