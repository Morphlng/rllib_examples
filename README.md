# rllib_examples

> Ray/RLlib examples that actually runs...

# Motivation

This repository aims to provide a reliable collection of scripts demonstrating the usage of Ray/RLlib. The motivation behind creating this repository stems from the challenges encountered with the examples provided directly in the official [RLlib repository](https://github.com/ray-project/ray/tree/master/rllib/examples). Many users, including myself, have found that most of these examples are difficult to run due to various issues. Some do not work at all, while others depend on specific, often outdated, versions of Ray and its dependencies. This inconsistency leads to a frustrating experience for practitioners and researchers who wish to explore and leverage Ray/RLlib for their projects.

Therefore, this repository serves as a curated set of examples that are tested and updated, focusing on usability and reproducibility. The goal is to lower the entry barrier for new users and provide a solid starting point for advanced users to build upon.

# File Structure

The file structure of this repository is designed to cater to different versions of Ray, reflecting the specific dependencies and setups required for each. Given the version-dependent nature of Ray/RLlib and its examples, we've organized the repository to ensure users can find and execute scripts with ease, corresponding to their Ray installation.

```bash
rllib-examples/
│
├── ray_1.8.0/
│   ├── basic_usage/
│   │   ├── create_model.py
│   │   ├── load_model.py
│   │   └── train_model.py
│   ├── tuned_examples/
│   │   └── ...
│   └── notebooks/
│       └── ...
│
├── ray_2.2.0/
│   ├── basic_usage/
│   │   ├── create_model.py
│   │   ├── load_model.py
│   │   └── train_model.py
│   ├── tuned_examples/
│   │   └── ...
│   └── notebooks/
│       └── ...
│
└── ray_2.8.1/
    ├── basic_usage/
    │   ├── create_model.py
    │   ├── load_model.py
    │   └── train_model.py
    ├── tuned_examples/
    │   └── ...
    └── notebooks/
        └── ...
```

- `basic_usage/`: Contains scripts demonstrating basic API functionalities such as creating a model, loading a model, and training a model.
- `tuned_examples/`: Includes scripts that are finely tuned and guaranteed to run successfully. These examples are intended to provide more practical and advanced use cases.
- `notebooks/`: Jupyter notebooks that allow for interactive experimentation and understanding of the models.

Each top-level directory corresponds to a different Ray version (e.g., ray-1.8.0, ray-2.2.0, ray-2.8.1). Inside each version-specific directory, scripts are further categorized based on their type/usage to streamline the user experience and navigation.

# Why these versions?

The selection of Ray versions in this repository is intentional, catering to different use cases and dependencies that users may encounter. Below is the rationale behind choosing each version:

## Ray 1.8.0

Ray 1.8.0 is an older, stable version that has been deprecated and is no longer supported officially, with even its documentation missing. However, it remains relevant due to its dependency by the [MARLlib](https://github.com/Replicable-MARL/MARLlib) library, which is a recent development. MARLlib users are required to use Ray 1.8.0 to leverage all the Multi-Agent Reinforcement Learning (MARL) algorithms provided by the library. Thus, understanding Ray 1.8.0 becomes essential for those looking to explore MARLlib's capabilities.

## Ray 2.2.0

Ray 2.2.0 represents the last version supporting the old Gym API (versions below 0.24.0) and does not require or support the newer gymnasium package. This version is particularly crucial for users wishing to utilize the [d4rl offline dataset](https://github.com/Farama-Foundation/D4RL), which requires an older version of Gym. As such, Ray 2.2.0 serves as a bridge for users who rely on legacy environments or datasets that have not transitioned to the latest Gym or gymnasium standards.

## Ray 2.8.1

Ray 2.8.1 is chosen for being a relatively recent version before significant structural changes were introduced in Ray/RLlib. From version 2.9.0 onwards, a substantial number of Reinforcement Learning (RL) algorithms were forced out of the core RLlib library into a separate module, [rllib_contrib](https://github.com/ray-project/ray/tree/ray-2.9.0/rllib_contrib). This move has led to complications due to the separate dependencies and compatibility issues within rllib_contrib. Therefore, we opt to use Ray 2.8.1 as it retains the comprehensive set of algorithms within RLlib itself, providing a more stable and unified framework for RL experimentation.
