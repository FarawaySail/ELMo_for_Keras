# ELMo词向量的使用
## 安装和数据准备
1. 在[blim-tf](https://github.com/allenai/bilm-tf)完成所需的安装操作，import所需的库：
```Python
from bilm import TokenBatcher, BidirectionalLanguageModel, weight_layers, dump_token_embeddings
import h5py
```
2. 在[Elmo官网](https://allennlp.org/elmo)上下载对应的模型，即weights和options。
## 数据处理
在NLP任务中，数据处理之后会得到word2idx和idx2word这样的两个字典，利用idx2word这个字典将word按照idx的顺序写入一个txt文件中，类似这样：
```Python
vocab_file = 'vocab_small.txt'
with open(vocab_file, 'w') as fout:
    for i in range(len(config.i2w)):
        fout.write(config.i2w[i]+'\n')
```
我们得到四个文件，vocab_file，options_file，weight_file，用于保存模型的token_embedding_file。像这样：
```Python
options_file = './ELMo/elmo_2x4096_512_2048cnn_2xhighway_options.json'
weight_file =  './ELMo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'
vocab_file = 'vocab_small.txt'
token_embedding_file = 'elmo_token_embeddings.hdf5'
```
## 具体操作
主要就是使用blim库中的dump_token_embedding函数，这个函数能将vocab_file里面的token经过模型得到的embedding matrix保存在一个新的hdf5文件里面，然后我们从这个hdf5文件中取出matrix即输出值，将这个matrix作为我们搭建的keras模型embedding层的权值即可。具体如下：
```Python
dump_token_embeddings(
    config.vocab_file, config.options_file, config.weight_file, config.token_embedding_file
	)
	tf.reset_default_graph()
	f = h5py.File(config.token_embedding_file,'r')
	wt = f['embedding'][:]
```
模型搭好后将wt作为权值：
```Python
model.get_layer('word_emb').set_weights([word_vector])
word_emb.trainable = False
```
