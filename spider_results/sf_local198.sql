SELECT AVG(total_sales) FROM sales WHERE country IN (SELECT country FROM sales GROUP BY country HAVING COUNT(*) > 4);