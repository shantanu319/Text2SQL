SELECT COUNT(DISTINCT session_id) FROM visits WHERE page = '/regist/input' AND next_page = '/regist/confirm';