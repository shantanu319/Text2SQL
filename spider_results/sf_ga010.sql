SELECT channel, COUNT(*) as session_count FROM website_traffic WHERE month = 'December' AND year = '2020' GROUP BY channel ORDER BY session_count DESC LIMIT 1