```sql
SELECT 
    address,
    SUM(value / POW(10, 18)) AS net_balance,
    COUNT(*) AS total_transactions,
    AVG(value / POW(10, 18)) AS avg_eth_transfer,
    COUNT(DISTINCT tx_hash) AS unique_counterparties,
    COUNT(DISTINCT token_address) AS unique_tokens,
    COUNT(DISTINCT from_address) AS incoming_transactions,
    COUNT(DISTINCT to_address) AS outgoing_transactions,
    COUNT(CASE WHEN status = 'success' THEN 1 ELSE NULL END) AS successful_transactions,
    COUNT(CASE WHEN status = 'failed' THEN 1 ELSE NULL END) AS failed_transactions,
    COUNT(CASE WHEN type = 'contract_creation' THEN 1 ELSE NULL END) AS contract_creations,
    COUNT(CASE WHEN type = 'mining_reward' THEN 1 ELSE NULL END) AS mining_rewards,
    COUNT(CASE WHEN type = 'delegatecall' OR type = 'callcode' OR type = 'staticcall' THEN 1 ELSE NULL END) AS excluded_transactions,
    COUNT(CASE WHEN type = 'erc20_in' THEN 1 ELSE NULL END) AS erc20_in_count,
    COUNT(CASE