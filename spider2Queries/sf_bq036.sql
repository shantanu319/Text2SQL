SELECT AVG(Commits) AS Average_Commits_Per_Month
FROM (
    SELECT COUNT(*) AS Commits, DATE_TRUNC('month', Commit_Date) AS Month
    FROM Repository_Events
    WHERE Language = 'Python' AND YEAR(Commit_Date) = 2016
    GROUP BY Month
) AS Monthly_Stats;