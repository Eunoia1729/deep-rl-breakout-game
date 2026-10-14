# deep-rl-breakout-game

Using Deep Q Networks (DQN) to play Atari Breakout, with TensorFlow/Keras and `gym` + `ale-py`.

The agent follows the setup from the DeepMind Nature paper: frame skipping with max-pooling, 84x84 grayscale
frames, a stack of the last 4 frames, an experience replay buffer, and a periodically synced target network.

This repository has two parts:

| File | What it is |
| --- | --- |
| [`train.py`](train.py) | Tested command-line training script. Use this to train. |
| [`deep_q_network.ipynb`](deep_q_network.ipynb) | Early exploratory notebook the script grew out of. |

## Train

Requires Python 3.11 (see `.python-version`): `gym==0.26.2` and `ale-py==0.8.1` have no wheels for newer versions.

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python train.py --episodes 1000                    # start fresh
python train.py --model models/<saved>.keras       # resume from a checkpoint
```

Checkpoints are written to `models/` (or `--save-dir`) every 20 episodes. They are git-ignored.

## Notebook status

The notebook is kept for reference and is not maintained. It is written against the same old `gym`/`ale-py`
versions, so its `!pip install "gym[atari, accept-rom-license]"` cell will not work on a Google Colab runtime
that defaults to Python 3.12. It also reads and writes models in a Google Drive folder
(`/content/drive/My Drive/dqn`); create it or edit the path in the second cell. To use it today, run it on
Python 3.11, or port it to `gymnasium` and a current `ale-py`.

## License

[MIT](LICENSE)
