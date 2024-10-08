## yDeposits
ySupport Vault Performance Bot

Developed as an internal support tool for the Yearn Finance ecosystem, the bot allows the ySupport team to easily provide users with detailed performance analytics for Yearn vaults, including APR and pricePerShare trends over specific blocks or time ranges. It streamlines support requests and answering common queries about vault performance.

### Features
- Historical APR and PricePerShare Reports: The bot generates detailed reports based on a vault's contract address and a specified block or time range.
- Graphical Visualizations: Outputs graphs showing pricePerShare and APR trends over time, helping visualize vault performance for different timeframes (1 day, 1 week, 1 month, 3 months, 6 months).
- Cross-Chain Support: Ethereum, Arbitrum, Polygon, Base, and Optimism
- Asset Growth Calculation: Accepts optional asset amount in input to calculate how much those assets have grown over time based on historical pricePerShare data.

### Usage
- User Input: A contributor provides the vault's contract address and either a block number or a time range (e.g. 1d, 1w, 1m, 3m, 6m).
- Data Gathering: The bot queries blockchain data to fetch and calculate the vault’s pricePerShare, APR, and other details for the requested time range or blocks.
- Report Generation: The bot generates a detailed report (text and optional graph) showing the performance over the specified period.
- Response: The report is sent to Telegram, allowing the contributor to relay the information to the user or analyze it further.

### Example
Input:  
0x92545bCE636E6eE91D88D2D017182cD0bd2fC22e 20871164`

Response:  
Vault: `DAI-2 yVault (yvDAI-2)`  
Chain: `ethereum`  
Contract: `0x92545bCE636E6eE91D88D2D017182cD0bd2fC22e`  
Blocks: `20871164` -> `20921377`  
Time: `2024-10-01 14:12:35 UTC` -> `2024-10-08 14:12:11 UTC (7.00 days)`  
pricePerShare: `1.070306797979569720` -> `1.071053017649064243`  
pricePerShare Difference: `0.000746219669494523`  
APR: `3.64%`    APY: `3.70%`  

Input:  
0x92545bCE636E6eE91D88D2D017182cD0bd2fC22e 1w

Response:  
![image](https://github.com/user-attachments/assets/22d15d55-7f6d-424c-b765-1b1cc2e9ad3a)
Vault: `DAI-2 yVault (yvDAI-2)`  
Chain: `ethereum`  
Contract: `0x92545bCE636E6eE91D88D2D017182cD0bd2fC22e`  
Blocks: `20871164` -> `20921377`  
Time: `2024-10-01 14:12:35 UTC` -> `2024-10-08 14:12:11 UTC (7.00 days)`  
pricePerShare: `1.070306797979569720` -> `1.071053017649064243`  
pricePerShare Difference: `0.000746219669494523`  
APR: `3.64%`    APY: `3.70%`  
