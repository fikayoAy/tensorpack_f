# TensorPack Financial Test Datasets

This directory contains real-world financial datasets for testing TensorPack functionality.

## Datasets

### 1. S&P 500 Historical Data (`sp500_2020_2023.csv`)
- **Format**: CSV
- **Size**: ~50MB
- **Period**: January 2020 - December 2023
- **Columns**: Open, High, Low, Close, Adj Close, Volume
- **Source**: Alpha Vantage API
- **Use Case**: Time series analysis, trend detection

### 2. Cryptocurrency Prices (`crypto_hourly.json`)
- **Format**: JSON
- **Size**: ~20MB
- **Data**: Bitcoin and Ethereum daily prices (1 year)
- **Fields**: prices, market_caps, total_volumes
- **Source**: CoinGecko API
- **Use Case**: JSON parsing, nested data analysis

### 3. Trading Volumes Matrix (`trading_matrix.npy`)
- **Format**: NumPy array
- **Size**: ~2MB
- **Content**: Correlation matrix of cryptocurrency returns
- **Additional files**: 
  - `trading_volumes_raw.npy`: Raw daily crypto prices (USD)
  - `trading_volumes_normalized.npy`: Normalized daily returns (0-1)
- **Shape**: 8x8 correlation matrix, plus time series data
- **Source**: CoinGecko API (8 major cryptocurrencies)
- **Use Case**: Matrix operations, correlation analysis, time series

## Quick Test Commands

```bash
# Test basic tensor conversion
python tensorpack.py tensor_to_matrix test_datasets/financial/trading_matrix.npy

# Test multi-format discovery
python tensorpack.py discover-connections --inputs test_datasets/financial/*.* \
    --output financial_connections.json --visualize financial_viz/ --export-formats all

# Test entity search
python tensorpack.py traverse-graph --inputs test_datasets/financial/*.* \
    --search-entity "volume" --generate-viz --export-formats all

# Test cross-format combination
python tensorpack.py combine \
    --inputs test_datasets/financial/trading_matrix.npy test_datasets/financial/trading_volumes_raw.npy \
    --output combined_trading_analysis.npy --mode weighted
```

## Dataset Statistics

- Total files: 6-8 files (including metadata)
- Total size: ~70-100MB
- Data types: CSV, JSON, NumPy arrays
- Time spans: 2020-2023 (4 years of data)
- Update frequency: Run this script monthly for fresh data

## Data Sources

- **Alpha Vantage**: S&P 500 stock data
- **CoinGecko**: Cryptocurrency data and correlation matrices
- **Computed**: Returns-based correlation and derived matrices

## API Keys Used

- Alpha Vantage: Q7R8TGNVOEFNF8K6
- CoinGecko: CG-QU6L51NQ5rsCysxYScYRKGrg

Generated on: 2025-09-01 18:49:37
