SELECT MAX(net_change) AS max_net_change, MIN(net_change) AS min_net_change FROM (
    SELECT SUM(debit - credit + gas_fee) AS net_change 
    FROM ethereum_classic_transactions 
    WHERE transaction_time = '2016-10-14' AND status = 'success'
);