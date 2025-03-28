To achieve this task using SQL, you can follow these steps:

1. Join the `posts_questions` and `posts_answers` tables based on the question ID.
2. Calculate the time difference between the question creation date and the earliest answer creation date.
3. Filter out answers that are not within one hour.
4. Group the results by the day of the week (using `DATE_FORMAT`) and calculate the percentage of questions that received an answer within an hour.
5. Order the results by the percentage in descending order and select the third highest percentage.

Here's the SQL query to accomplish this:

```sql
WITH AnsweredQuestions AS (
    SELECT 
        pq.question_id,
        pq.creation_date,
        MIN(pa.creation_date) AS earliest_answer_creation_date
    FROM 
        posts_questions pq
    LEFT JOIN 
        posts_answers pa ON pq.question_id = pa.parent_id
    WHERE 
        pa.creation_date IS NOT NULL
    GROUP BY 
        pq.question_id, pq.creation_date
),
TimeDifference AS (
    SELECT 
        aq.question_id,
        aq.creation_date,
        aq.earliest_answer_creation_date,
        TIMESTAMPDIFF(HOUR, aq.creation_date, aq.earliest_answer_creation_date) AS