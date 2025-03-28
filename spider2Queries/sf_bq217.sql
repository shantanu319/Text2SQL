```sql
SELECT COUNT(*) AS pull_requests_count
FROM githubarchive_20230118
JOIN github_repos_languages ON githubarchive_20230118.repo_id = github_repos_languages.repo_id
WHERE githubarchive_20230118.created_at >= '2023-01-18'
  AND github_repos_languages.language = 'JavaScript';
```