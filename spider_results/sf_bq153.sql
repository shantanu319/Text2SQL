SELECT histology_type, AVG(log10(normalized_count + 1)) AS avg_expression_level FROM