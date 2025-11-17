# LSTM Molecular Generation Project

## Overview

This project implements a character-level LSTM neural network to generate novel SMILES (Simplified Molecular Input Line Entry System) strings. It trains on a dataset of existing molecular structures and learns to generate new, valid chemical molecules by predicting character sequences.

## Project Features

### Core Functionality

- **SMILES Data Processing**: Loads and preprocesses SMILES strings with configurable length filtering
- **Character Translation**: Simplifies vocabulary by translating multi-character symbols (e.g., 'Br' → 'R', 'Cl' → 'L')
- **Sequence Padding**: Automatically pads/filters sequences with special tokens:
  - 'G': Start token
  - 'E': End token
  - 'A': Padding character
- **One-Hot Encoding**: Converts character sequences into machine-readable one-hot encoded format
- **LSTM Model Architecture**: Multi-layer LSTM with dropout for robust training
- **Temperature-Controlled Sampling**: Generates diverse molecules with adjustable randomness

## Class Structure

### `SmilesToOneHotEncoding`

The main class that orchestrates the entire pipeline from raw SMILES data to trained LSTM model and molecule generation.

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

## Installation

### Requirements

```
numpy
tensorflow
scikit-learn
```

### Setup

```bash
pip install numpy tensorflow scikit-learn
```

## Usage

### Basic Example

```python
# Initialize the processor with your SMILES dataset
generator = SmilesToOneHotEncoding(
    filename="/path/to/smiles_data.txt",
    size_dataset=100,      # Number of molecules to use
    min_length=20,         # Minimum sequence length
    max_length=100         # Maximum sequence length
)

# Preprocess and get encoded sequences
sequences = generator.str_to_encode()

# Build LSTM model
model = generator.build_lstm_model(num_layers_LSTM=2)

# Train the model
history = generator.train_lstm_model(
    model, 
    sequences, 
    epochs=25, 
    batch_size=16
)

# Generate new molecules
new_molecule = generator.generate_molecule(model)
print(f"Generated SMILES: {new_molecule}")
```

### Configuration Parameters

**SmilesToOneHotEncoding Constructor:**
- `filename` (str): Path to SMILES data file
- `size_dataset` (int, default=100): Number of molecules to load
- `min_length` (int, default=20): Minimum molecule length
- `max_length` (int, default=100): Maximum molecule length (padding/truncation target)

**build_lstm_model:**
- `num_layers_LSTM` (int, default=2): Number of LSTM layers

**train_lstm_model:**
- `epochs` (int, default=25): Training epochs
- `batch_size` (int, default=16): Samples per gradient update

**sample_with_temperature:**
- `temperature` (float, default=0.8): Controls sampling randomness
  - 0 < temp < 1: More conservative, predictable samples
  - temp = 1: Standard probability sampling
  - temp > 1: More diverse, random samples

## Data Format

The input SMILES file should contain one SMILES string per line:
```
CC(C)Cc1ccc(cc1)[C@@H](C)C(O)=O
O=C(O)Cc1ccccc1
```

The loader automatically extracts the first column (before space and comma) from each line.

## Model Architecture

The LSTM model consists of:

1. **Input Layer**: Accepts one-hot encoded sequences of shape `(max_length, vocab_size)`
2. **LSTM Layers**: Multiple stacked LSTM layers with 256 units each
   - Return sequences to allow stacking
   - ReLU activation for improved learning
3. **Dropout Layers**: 20% dropout after each LSTM to prevent overfitting
4. **Dense Layer**: Maps LSTM output to vocabulary size
5. **Softmax Output**: Produces probability distribution over characters

## How It Works

1. **Preprocessing**: SMILES strings are loaded, translated, and padded with special tokens
2. **Tokenization**: Each character is mapped to an integer index
3. **Encoding**: Sequences are converted to one-hot vectors
4. **Training**: LSTM learns to predict the next character given the previous sequence
5. **Generation**: New molecules are generated character-by-character using temperature-controlled sampling

## Training Details

- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam
- **Metric**: Accuracy
- **Train/Test Split**: 80/20
- **Data Augmentation**: Sequential shift-based target generation

## Generation Process

When generating new molecules:

1. Start with 'G' token
2. For each position, predict the probability distribution of the next character
3. Sample a character using temperature-controlled multinomial sampling
4. Stop when 'E' token is generated or max_length is reached
5. Return the generated SMILES string

## Future Enhancements

- [ ] Molecular validity verification (e.g., using RDKit)
- [ ] Custom sampling strategies (beam search, nucleus sampling)
- [ ] Variable-length sequence support
- [ ] Multi-objective generation (optimize for specific molecular properties)
- [ ] Attention mechanisms for improved long-range dependencies
- [ ] Bidirectional LSTM variants

## Notes

- The character-to-index mapping uses 1-based indexing from Keras Tokenizer, which is adjusted to 0-based for one-hot encoding
- Special tokens ('G', 'E', 'A') are learned as part of the vocabulary
- The model uses pre-padding during generation to maintain consistent input shapes
- Temperature sampling provides a balance between diversity and validity

## License

This project is part of the LSTM exploration and research initiative.
