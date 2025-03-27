```sql
SELECT p.product_name, 
       ROUND((p.total_sales_q4_2020 - p.total_sales_q4_2019) / p.total_sales_q4_2019 * 100, 2) AS percentage_point_change,
       p.rank_in_top_20_percent_of_total_sales
FROM (
    SELECT p.product_name, 
           SUM(p.quantity_sold) AS total_sales_q4_2019,
           SUM(p.quantity_sold) AS total_sales_q4_2020,
           RANK() OVER (ORDER BY SUM(p.quantity_sold) DESC) AS rank_in_top_20_percent_of_total_sales
    FROM products p
    JOIN sales s ON p.product_id = s.product_id
    WHERE s.promo_id = 999 AND s.city IN (
        SELECT c.city
        FROM cities c
        JOIN sales s ON c.city_id = s.city_id
        WHERE s.calendar_quarter_id BETWEEN 1772 AND 1776
        GROUP BY c.city
        HAVING SUM(s.quantity_sold) > SUM(LAG(SUM(s.quantity_sold)) OVER (PARTITION BY c.city ORDER BY s.calendar_quarter