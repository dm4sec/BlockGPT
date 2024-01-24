# BlockGPT  

This is a coarse implementation of [BlockGPT](#BlockGPT_dblp) proposed by Yu et al. Although we argue the practicality of this tool according to the stated FPR and the bias of the transaction in real world, the tool is implemented anyway.

The BlockGPT consists of 3 modules, they are ITR Builder, Trainer, and Detector module separately. 


Credit to [Bella-LYT](https://github.com/Bella-LYT) and [zzy](https://github.com/zzzzyying).

# HOWTO
1. Prerequisites:
```
$ git clone https://github.com/dm4sec/BlockGPT.git
$ cd BlockGPT
$ virtualenv -p python3 venv
$ source venv/bin/activate
$ pip3 install -r requirements.txt
```
2. use `python3 M1.ITR_Builder.py` to get txs and build data. A local archive node is preferred for the given node in `config.py` is fairly slow.
3. use `python3 M2.Trainer.py --train-tokenizer` to train a tokenizer.
4. use `python3 M2.Trainer.py --train-classifier` to train a classifier.
5. use `python3 M3.Detector.py` to detect abnormally txs.

## Reference
<span id='BlockGPT_dblp'></span> 
```
1. Yu Gai, Liyi Zhou, Kaihua Qin, Dawn Song, Arthur Gervais:
Blockchain Large Language Models. IACR Cryptol. ePrint Arch. 2023: 592 (2023)  
2. https://github.com/BLOCK-GPT-NEW/blockGPT  
3. https://github.com/sec3-service/Owl-LM
4. https://ethervm.io/
```
