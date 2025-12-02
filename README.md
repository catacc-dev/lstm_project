# LSTM Molecular Generation Project

This repository contains a basic LSTM that generates novel chemical molecules.

<details>
<summary>Table of Contents</summary>

1. [About the Project](#about-the-project)
   - [Project Features](#project-features)
   - [Model Architecture](#model-architecture)
   - [How It Works](#how-it-works)
2. [Getting Started](#getting-started)
   - [Installation](#installation)
3. [Contributing](#contributing)

</details>

## About the Project

This project implements an LSTM neural network to generate novel SMILES (Simplified Molecular Input Line Entry System) strings. It trains on a dataset of existing molecular structures and learns to generate new chemical molecules by predicting character sequences.

### Project Features

1. **Core Functionality**

- *SMILES Data Processing*: Loads and preprocesses SMILES strings with configurable length filtering
- **Character Translation**: Simplifies vocabulary by translating multi-character symbols (e.g., 'Br' → 'R', 'Cl' → 'L')
- *Sequence Padding*: Automatically pads/filters sequences with special tokens:
  - 'G': Start token
  - 'E': End token
  - 'A': Padding character
- *One-Hot Encoding*: Converts character sequences into a machine-readable one-hot encoded format
- *LSTM Model Architecture*: Multi-layer LSTM with dropout for robust training
- *Temperature-Controlled Sampling*: Generates diverse molecules with adjustable randomness in the following manner:
  - Start with 'G' token
  - For each position, predict the probability distribution of the next character
  - Sample a character using temperature-controlled multinomial sampling
  - Stop when 'E' token is generated or max_length is reached

2. **Class Structure**

The main class (`SmilesToOneHotEncoding`) that orchestrates the entire pipeline from raw SMILES data to a trained LSTM model and molecule generation.

**Key Methods:**

| Method | Purpose |
|--------|---------|
| `__init__()` | Initialize with dataset path and preprocessing parameters |
| `load_file()` | Load SMILES strings from file |
| `translation()` | Replace multi-character symbols with single characters |
| `padding()` | Add start/end tokens and pad sequences to uniform length |
| `preprocess_data()` | Orchestrate full preprocessing pipeline |
| `make_vocabulary()` | Create character-level vocabulary and character-to-index mapping |
| `str_to_encode()` | Convert SMILES strings to one-hot encoded sequences |
| `get_targets()` | Generate shifted target sequences for supervised learning |
| `build_lstm_model()` | Create and compile multi-layer LSTM model |
| `train_lstm_model()` | Train model on preprocessed sequences |
| `sample_with_temperature()` | Temperature-controlled character sampling |
| `generate_molecule()` | Generate new SMILES molecule character-by-character |

### Model Architecture

The LSTM model consists of:

1. **Input Layer**: Accepts one-hot encoded sequences of shape `(max_length, vocab_size)`
2. **LSTM Layers**: Multiple stacked LSTM layers with 256 units each
   - Return sequences to allow stacking
   - ReLU activation for improved learning
3. **Dropout Layers**: 20% dropout after each LSTM to prevent overfitting
4. **Dense Layer**: Maps LSTM output to vocabulary size
5. **Softmax Output**: Produces probability distribution over characters
  
## Getting started

### Installation
1. Clone the repo
```bash
   git clone https://github.com/catacc-dev/lstm_project.git
   cd lstm_project
```

2. Create a virtual environment
```bash
   python3 -m venv venv
   source venv/bin/activate      # Linux or macOS
   venv\Scripts\activate         # Windows
```

3. Install packages
```bash
   pip install -r requirements.txt
```

## Contributing
Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also open an issue with the tag "enhancement". Don't forget to give the project a star! Thanks again!


