To convert the given text into an SQL query, we need to break down the requirements and construct a query that retrieves the necessary information. Here's a step-by-step breakdown:

1. **Data Sources**: We need to access temperature data from LaGuardia (STN=725030) and JFK (STN=744860) airports for the period from 2008 to 2017.

2. **Complaint Data**: We need to retrieve complaint data for each complaint type.

3. **Correlation Analysis**: For each complaint type with more than 5000 total occurrences and a strong correlation (absolute value > 0.5) with temperature, we need to calculate:
   - Total number of complaints.
   - Total number of days with valid temperature records.
   - Pearson correlation coefficients between temperature and daily complaint count.
   - Pearson correlation coefficients between temperature and daily percentage of total complaints.

4. **Filtering**: Exclude any days with missing or invalid temperature data (such as 9999.9).

Here's the SQL query based on these requirements:

```sql
WITH TemperatureData AS (
    SELECT 
        DATE,
        STN,
        TEMP
    FROM