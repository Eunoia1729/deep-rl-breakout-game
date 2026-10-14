# deep-rl-breakout-game
Using Deep Q Networks to play Atari Breakout game

## Status and requirements

This notebook is an early exploration written against `gym==0.26.2` and `ale-py==0.8.1`.
Those versions only provide wheels up to **Python 3.11**, so the `!pip install "gym[atari, accept-rom-license]"`
cell will not work on a Google Colab runtime that defaults to Python 3.12. Use a Python 3.11 environment, or
port the notebook to `gymnasium` and a current `ale-py`.

It also reads and writes models in a Google Drive folder (`/content/drive/My Drive/dqn`); create that folder or
edit the path in the second cell before running.

A cleaned-up, tested training script based on this notebook is available as a standalone project.
