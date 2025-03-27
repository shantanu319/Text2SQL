To convert the given text into SQL, we need to break down the requirements and create a query that processes the data accordingly. Here's a step-by-step breakdown:

1. **Data Preparation**: Assume you have a table named `interest_data` with columns:
   - `date`: The date when the data was collected.
   - `interest_name`: The name of the interest.
   - `composition`: The composition value.
   - `index_value`: The index value.

2. **Calculate Average Composition**: For each month, calculate the average composition by dividing the composition by the index value.

3. **Identify Top Interest**: For each month, identify the interest with the highest average composition value and report its average composition as the max index composition for that month.

4. **Compute Three-Month Rolling Average**: Calculate the three-month rolling average of these monthly max index compositions.

5. **Report Previous Month Interests**: Report the names and max index compositions of the top interests from one month ago and two months ago.

Here's the SQL query to achieve this:

```sql
WITH MonthlyMaxCompositions AS (
    SELECT
        DATE_TRUNC('month', date) AS month,
        interest_name,
        AVG(composition / index_value) AS avg_composition