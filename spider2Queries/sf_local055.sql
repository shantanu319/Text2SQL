Here is a possible SQL query that could achieve the desired result:

```sql
WITH TopSellingArtist AS (
  SELECT artist_name
  FROM album_sales
  GROUP BY artist_name
  ORDER BY SUM(sales_amount) DESC
  LIMIT 1
),
LowestSellingArtist AS (
  SELECT artist_name
  FROM album_sales
  GROUP BY artist_name
  ORDER BY SUM(sales_amount) ASC
  LIMIT 1
),
TopSellingAlbums AS (
  SELECT album_id, artist_name, sales_amount
  FROM album_sales
  WHERE artist_name = (SELECT artist_name FROM TopSellingArtist)
),
LowestSellingAlbums AS (
  SELECT album_id, artist_name, sales_amount
  FROM album_sales
  WHERE artist_name = (SELECT artist_name FROM LowestSellingArtist)
),
CustomerSpending AS (
  SELECT customer_id, SUM(album_sales.sales_amount) AS total_spent
  FROM album_sales
  JOIN TopSellingAlbums ON album_sales.album_id = TopSellingAlbums.album_id
  OR album_sales.album_id IN (SELECT album_id FROM LowestSellingAlbums)
  GROUP BY customer_id
),
AverageSpending AS (
  SELECT AVG(total_spent