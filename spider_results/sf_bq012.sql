SELECT AVG(balance / 1000000000000000000) AS avg_balance FROM (
    SELECT address, SUM(gas_used * gas_price) AS miner_reward, SUM(incoming + outgoing - deductions) AS net_balance
    FROM traces
    WHERE status = 'success' AND type NOT IN ('delegatecall', 'callcode', 'staticcall')