SELECT 
    a.name AS top_selling_artist,
    b.name AS lowest_selling_artist,
    SUM(a.sales * c.price) AS total_spending_top_seller,
    SUM(b.sales * d.price) AS total_spending_lowest_seller