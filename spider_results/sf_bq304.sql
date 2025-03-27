```sql
SELECT 
    q.Title,
    q.Body,
    COUNT(v.ViewCount) AS ViewCount
FROM 
    Questions q
JOIN 
    Tags t ON q.Tags LIKE '%' + t.TagName + '%'
JOIN 
    QuestionTags qt ON q.Id = qt.QuestionId
JOIN 
    Views v ON q.Id = v.QuestionId
WHERE 
    t.TagName IN ('android-layout', 'android-activity', 'android-intent', 'android-edittext', 'android-fragments', 'android-recyclerview', 'listview', 'android-actionbar', 'google-maps', 'android-asynctask')
    AND q.Title LIKE '%how%'
    AND q.Body LIKE '%how%'
    AND q.Title NOT LIKE '%fail%'
    AND q.Title NOT LIKE '%problem%'
    AND q.Title NOT LIKE '%error%'
    AND q.Title NOT LIKE '%wrong%'
    AND q.Title NOT LIKE '%fix%'
    AND q.Title NOT LIKE '%bug%'
    AND q.Title NOT LIKE '%issue%'
    AND q.Title NOT LIKE '%solve%'
    AND q.Title NOT LIKE '%trouble%'
    AND q.Body NOT LIKE '%fail%'
    AND q.Body NOT LIKE '%problem%'
    AND q.Body NOT LIKE '%error%'