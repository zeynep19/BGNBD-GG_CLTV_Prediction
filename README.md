# BGNBD-GG_CLTV_Prediction
Probabilistic lifetime value estimation with time projection
BG/NBD Model
Beta Geometric / Negative Binomial distribution=Expected Number of Transaction
The BG/NBD Model probabilistically models two processes for the Expected Number of Transaction.
a. Transaction Process (Buy)
• As long as it is alive, the number of transactions to be performed by a client in a given time period is distributed poisson by the transaction rate parameter.
• As long as a customer is alive, they will continue to make random purchases around their transaction rate.
• Transaction rates vary for each customer and gamma is distributed for the entire audience (r,a)
b. Dropout Process (Till you die)
• Each customer has a dropout rate (dropout probability) with probability p.
• A customer drops with a certain probability after making a purchase.
• Dropout rates vary for each client and beta is distributed for the entire audience (a,b)
BG/NBD Model = Estimation of the expected number of sales in a certain period, conditioned for a particular customer, considering the transaction rate distribution of the whole audience and the dropout rate distribution of the whole audience.
Gamma Gamma Submodel
It is used to estimate how much profit a customer can generate on average per trade.
• The monetary value of a customer's transactions is randomly distributed around the average of the transaction values.
• The average transaction value may change between users over time, but not for a single user.
• The average transaction value is gamma distributed among all customers.
