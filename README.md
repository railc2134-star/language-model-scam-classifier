# ScamGuardAI

ScamGuardAI is a multi-domain machine learning project focused on experimenting with deep learning models across NLP, reinforcement learning, time-series forecasting, and real-time applications.

The repository contains multiple independent AI systems built for research and learning purposes, including scam detection, transformer models, RL agents, and Discord bots.

---

## What this project includes

- Scam detection models for Discord and chat messages using BERT and embedding-based networks  
- Transformer and cross-attention models for question answering tasks  
- Reinforcement learning agents (DQN, SAC, Actor-Critic) for Gymnasium environments  
- Time-series prediction models for cryptocurrency price forecasting  
- Discord bots using trained models for real-time inference and automation  

---

## NLP / Scam Detection

- Detects phishing and scam messages  
- Uses BERT-based classifiers and embedding models  
- Trained on multiple Discord scam datasets  
- Real-time inference via Discord bot integration  

---

## Transformer Models

- Encoder-decoder architectures  
- Cross-attention based question answering model  
- BERT-based embeddings for context understanding  
- Experiments on sequence-to-sequence learning  

---

## Reinforcement Learning

- DQN implementation for CartPole  
- SAC and Actor-Critic implementations for continuous control  
- Experiments on Pendulum and MuJoCo environments (Ant)  

---

## Time Series Prediction

- LSTM-based Bitcoin price prediction model  
- Transformer-based forecasting model  
- Sliding window training on historical price data  

---

## Discord Bots

- Scam detection bot for live message classification  
- Game management bot with team balancing system  
- Integration with trained PyTorch models for inference  

---

## Requirements

- Python 3.10+
- PyTorch
- Transformers
- Gymnasium
- Datasets
- Discord.py  

---

## Notes

This is an experimental research project. Models are trained on mixed datasets and are intended for learning and experimentation, not production use.

---

## Future Improvements

- unify all models into a single framework  
- improve reinforcement learning stability  
- add experiment tracking and logging  
- optimize dataset pipelines  
- build a unified inference API  
