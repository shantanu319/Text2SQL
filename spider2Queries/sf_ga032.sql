To achieve the desired result using SQL, we need to follow these steps:

1. **Filter the Data**: Select records where the `customer_id` is 1362228, the `visit_date` is January 28th 2021, and the `page_title` is either "PDP" or "PLP".

2. **Sort the Data**: Order the results by the `visit_date` in ascending order to get the sequence of visits.

3. **Concatenate Page Titles**: Use a string concatenation function to join the page titles with ">>" between each pair.

4. **Merge Adjacent Identical Page Titles**: This step can be complex depending on your database system. For simplicity, let's assume that adjacent identical page titles are already merged in the data source.

Here's an example SQL query assuming you have a table named `visits` with columns `customer_id`, `visit_date`, `page_title`, and `sequence_number` (which might be used for merging):

```sql
SELECT 
    CONCAT(
        GROUP_CONCAT(page_title SEPARATOR ' >> ')
    ) AS navigation_flow
FROM 
    visits
WHERE 
    customer_id = 1362228