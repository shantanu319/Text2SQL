```sql
SELECT 
    Category,
    COUNT(*) AS TotalArticles,
    SUM(CASE WHEN ArticleText LIKE '%education%' THEN 1 ELSE 0 END) AS ArticlesWithEducation,
    (SUM(CASE WHEN ArticleText LIKE '%education%' THEN 1 ELSE 0 END) * 100.0 / COUNT(*)) AS PercentageOfEducation
FROM 
    bbc_news_articles
GROUP BY 
    Category;
```