SELECT file_type, COUNT(*) as file_count FROM (
    SELECT 'Python' AS file_type, COUNT(*) FROM github_codebase WHERE file_extension = '.py'
    UNION ALL
    SELECT 'C' AS file_type, COUNT(*) FROM github_codebase WHERE file_extension = '.c'
    UNION ALL
    SELECT 'Jupyter Notebook' AS file_type, COUNT(*) FROM github_codebase WHERE file_extension = '.ipynb'
    UNION ALL
    SELECT 'Java' AS file_type, COUNT(*) FROM github_codebase WHERE file_extension = '.java'
    UNION ALL
    SELECT 'JavaScript' AS file_type, COUNT(*) FROM github_codebase WHERE file_extension = '.js'
)