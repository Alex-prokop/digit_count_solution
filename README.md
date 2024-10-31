# Project Digit Count Solution

## Installation and Run

### Clone the repository:

   ```bash
   git clone https://github.com/Alex-prokop/digit_count_solution
   cd digit_count_solution
   ```

### Running with Python:

1. **Create and activate a virtual environment** (Python 3.10 is recommended):

   ```bash
   python3.10 -m venv venv
   source venv/bin/activate  # For Linux/Mac
   venv\Scripts\activate     # For Windows
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:

   ```bash
   python digit_recognition.py
   ```

> **Note**: Make sure Python 3.10 is installed.

---

### Running with JavaScript:

1. **Install dependencies**:

   ```bash
   npm install
   ```

2. **Run the application**:

   - To start the main application:
     ```bash
     npm start
     ```

> **Note**: Make sure Node.js is installed.

## Technologies Used

**For Python**:
- Used TensorFlow and Keras to load MNIST data, build the model, train it, and save it in `.h5` format.

**For JavaScript**:
- Used TensorFlow.js and the `mnist` package to load data, create and train the model, and save it in TensorFlow.js format optimized for web applications.
