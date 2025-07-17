Snap2MIDI is not available on pip; so to install it, you would have to do this from the github repo. The following steps below highlight whay you should do.

1. First of all, clone the official repo:

    ```
    git clone https://github.com/Nkcemeka/Snap2MIDI.git
    ```

2. Create a virtual environment:

    ```
    python3 -m venv venv
    ```

3. Activate the environment (for example):

    ```
    source ./venv/bin/activate
    ```

4. Go to the root directory and run the following:

    ```
    pip install -e .
    ```

Once you follow steps above, you should be able to train the models and make inference as desired. If something goes wrong, please write an issue [here](https://github.com/Nkcemeka/Snap2MIDI/issues).