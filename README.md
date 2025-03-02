# Advanced AES Encryption with Chaotic Clocking

This project explores enhancing AES encryption with chaotic clocking to defend against side-channel and fault injection attacks. By introducing frequency fluctuations, it disrupts potential attack patterns. The project uses machine learning to classify attack types based on these fluctuations, demonstrating the potential of chaotic clocking to improve security.

## Features

- **Attack Resistance:** Uses chaotic clocking to resist side-channel and fault injection attacks.
- **Machine Learning Integration:** Employs a Random Forest Classifier to detect and classify attack types.
- **Dynamic Clocking:** Adjusts clock frequencies dynamically in response to detected attacks.
- **Simulation and Testing:** Includes simulation of encryption processes and attack scenarios to validate the effectiveness of countermeasures.

## How to Use

1. Clone the repository to your local machine.

    ```bash
    git clone https://github.com/manya-kj/advanced_AES_encryption
    ```

2. Install python libraries.

    ```bash
    pip install -r requirements.txt 
    ```

3. Run the encryption script.

    ```bash
    python aes_encryp.py
    ```

## Dependencies

- Python 3.11.2
- NumPy
- scikit-learn
- matplotlib

## Notes

- The project simulates a secure encryption environment.
- Users can modify parameters to observe different outcomes.
- Contribute to enhance encryption security in IoT and embedded systems.