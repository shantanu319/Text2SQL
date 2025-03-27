```sql
WITH LatestEvalGroups AS (
    SELECT StateCode, EvaluationGroup, MAX(EvaluationDate) AS LatestEvaluationDate
    FROM EvaluationGroups
    GROUP BY StateCode, EvaluationGroup
),
TimberlandPlots AS (
    SELECT 
        T2.StateCode,
        T2.EvaluationGroup,
        T2.StateName,
        SUM(T3.AdjustedExpansionFactor * CASE WHEN T3.SiteProductivityClass BETWEEN 1 AND 6 THEN T3.MACR ELSE T3.SUBP END) AS TotalAcreage
    FROM 
        EvaluationTypes AS T1
    INNER JOIN 
        EvaluationGroups AS T2 ON T1.EvaluationType = T2.EvaluationType
    INNER JOIN 
        Plots AS T3 ON T2.EvaluationGroup = T3.EvaluationGroup
    WHERE 
        T1.EvaluationType = 'EXPCURR'
        AND T3.ConditionStatusCode = 1
        AND T3.ReservedStatusCode = 0
        AND T3.SiteProductivityClass BETWEEN 1 AND 6
    GROUP BY 
        T2.StateCode, T2.EvaluationGroup, T2.StateName
),
ForestlandPlots AS (
    SELECT 
        T2