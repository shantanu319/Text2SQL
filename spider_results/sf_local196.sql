```sql
SELECT 
    r.rating_category,
    AVG(c.total_amount_spent) AS avg_total_amount_spent,
    AVG(r.num_subsequent_rentals) AS avg_num_subsequent_rentals
FROM 
    ratings r
JOIN 
    (
        SELECT 
            c.customer_id,
            MIN(p.payment_date) AS first_payment_date,
            SUM(m.rental_fee) AS total_amount_spent,
            COUNT(m.rental_id) - 1 AS num_subsequent_rentals
        FROM 
            customers c
        JOIN 
            payments p ON c.customer_id = p.customer_id
        JOIN 
            movies m ON p.movie_id = m.movie_id
        GROUP BY 
            c.customer_id, r.rating_category
    ) c ON r.rating_category = c.rating_category AND c.first_payment_date = r.first_payment_date
GROUP BY 
    r.rating_category;
```

This SQL query calculates the average total amount spent and the average number of subsequent rentals for customers who have rented their first movie in a specific rating category. It joins the `ratings` table with a subquery that groups customers by their first payment date and calculates the total amount spent and the number of subsequent rentals for each customer. The results are then grouped by