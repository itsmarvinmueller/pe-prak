# Protocol Engineering Praktikum

## Project Overview

This project simulates datacenter TCP (DCTCP) behavior using Mininet with custom RED queue configurations.

The `scripts` folder contains shell scripts and environment configuration files (`.env`) that define different RED queue profiles for TCP congestion control with ECN support.


## Folder Structure

## Usage

### Applying RED Queue Configuration with DCTCP

The `dctcp.sh` script applies RED queue settings with ECN on the Mininet switch interfaces. It takes the name of a configuration file inside `scripts/configs/` as an argument.

### Example:

```bash
chmod +x scripts/dctcp.sh
./scripts/dctcp.sh <FILE_NAME>.env
```
The script sources the selected ```.env``` file and applies the parameters to the relevant switch interfaces.

### Profiles Overview
- **conservative.env**: Marks packets cautiously to favor stable throughput and less aggressive ECN usage.

- **balanced.env**: A middle ground profile balancing responsiveness and throughput, suitable for typical datacenter workloads.

- **aggressive.env**: Aggressively marks packets to quickly respond to congestion, prioritizing low latency at the potential cost of higher throughput variability.

- **default.env**: A safe default configuration, can be customized as needed.