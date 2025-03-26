SELECT TOP 5 HistologyType, COUNT(*) AS MutationCount FROM BRCA WHERE CDH1Mutation = 'Yes' GROUP BY HistologyType ORDER BY MutationCount DESC