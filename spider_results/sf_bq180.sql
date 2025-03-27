```sql
SELECT 
    CASE 
        WHEN file_extension = '.py' THEN 'Python'
        ELSE 'R'
    END AS language,
    module_name,
    COUNT(*) AS occurrence_count
FROM (
    SELECT 
        file_path,
        file_extension,
        CASE 
            WHEN file_extension = '.py' THEN SUBSTRING_INDEX(SUBSTRING_INDEX(line_content, 'import ', -1), ',', 1)
            ELSE SUBSTRING_INDEX(SUBSTRING_INDEX(line_content, 'library(', -1), ')', 1)
        END AS module_name
    FROM (
        SELECT 
            file_path,
            line_number,
            line_content,
            SUBSTRING_INDEX(file_path, '.', -2) AS file_extension
        FROM (
            SELECT 
                file_path,
                line_number,
                line_content
            FROM (
                SELECT 
                    file_path,
                    line_number,
                    line_content
                FROM python_files
                UNION ALL
                SELECT 
                    file_path,
                    line_number,
                    line_content
                FROM r_files
            ) AS all_files
        ) AS filtered_files
    ) AS processed_lines
) AS extracted_modules
GROUP BY 
    language, 
    module_name
ORDER BY 
    occurrence_count DESC
LIMIT