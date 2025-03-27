```sql
SELECT 
    CASE 
        WHEN T3.journal_title IS NOT NULL THEN T3.journal_title
        WHEN T3.proceedings_title IS NOT NULL THEN T3.proceedings_title
        WHEN T3.book_title IS NOT NULL THEN T3.book_title
        ELSE T3.book_series_title
    END AS venue_title
FROM 
    publications P
JOIN 
    grids G ON P.grid_id = G.id
LEFT JOIN 
    journals J ON P.publication_id = J.publication_id
LEFT JOIN 
    proceedings_proceedings PP ON P.publication_id = PP.publication_id
LEFT JOIN 
    books B ON P.publication_id = B.publication_id
LEFT JOIN 
    book_series BS ON P.publication_id = BS.publication_id
WHERE 
    YEAR(P.date_inserted) >= 2021 AND G.address_city = 'Qianjiang';
```