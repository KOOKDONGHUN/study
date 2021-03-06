import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

tf.set_random_seed(0)
np.random.seed(0)

# data
sentences = ["나 고양이 좋다",
             "나 강아지 좋다", "나 동물 좋다",
             "강아지 고양이 동물",
             "여자친구 고양이 강아지 좋다",
             "고양이 생선 우유 좋다",
             "강아지 생선 싫다 우유 좋다",
             "강아지 고양이 눈 좋다",
             "나 여자친구 좋다",
            "여자친구 나 싫다",
             "여자친구 나 영화 책 음악 좋다",
             "나 게임 만화 애니 좋다",
             "고양이 강아지 싫다",
             "강아지 고양이 좋다"]

word_sequence = " ".join(sentences).split(" ") # 쭉피고 다시 공백으로 스플릿 

word_list = " ".join(sentences).split(" ")
word_list = list(set(word_list)) # 공백을 기준으로 나눈 토큰들중 중복을 제거
word_dict = {w : i for i, w in enumerate(word_list)}

skip_gram = []

for i in range(1, len(word_sequence) -1):
    target = word_dict[word_sequence[i]]
    context = [word_dict[word_sequence[i -1]], word_dict[word_sequence[i + 1]]]

    for w in context:
        skip_gram.append([target, w])

def random_batch(data, size):
    random_inputs = []
    random_labels = []

    random_index = np.random.choice(range(len(data)), size, replace=False)

    for i in random_index:
        random_inputs.append(data[i][0])
        random_labels.append([data[i][1]])

    return random_inputs, random_labels

# parameters
epochs = 300
lr = 0.1
batch_size = 20
embedding_size = 2 # 이게 window size 말하는거 최대 몇개 까지 가까운 단어? 토큰? 을 선택 할 것 인가
num_sampled = 15
voc_size = len(word_list) # 중복을 제외한 총 단어의 개수 

# model
inputs = tf.placeholder(tf.int32, shape=[batch_size])
labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

embeddings = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0)) # word2vec 모델의 결과 갑인 임베딩 벡터를 저장할 변수다. (-1에서 1사이)

selected_embed = tf.nn.embedding_lookup(embeddings, inputs) # embeddings에서 inputs 값에 대한 inverse같은 느낌 말그대로 lookup이니까

nce_weights = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
nce_bias = tf.Variable(tf.zeros([voc_size]))

loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_bias, labels, selected_embed, num_sampled, voc_size))
                    #   tf.nn.nce_loss(nce_weights, nce_biases, labels, selected_embed, num_sampled, voc_size))


train_op = tf.train.AdamOptimizer(lr).minimize(loss)

# train
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for step in range(1, epochs + 1):
        batch_inputs, batch_labels = random_batch(skip_gram, batch_size)

        _, loss_val = sess.run([train_op, loss],
                               feed_dict={inputs : batch_inputs,
                                          labels : batch_labels})

        if step % 10 == 0:
            print("loss at step", step, ": ", loss_val)

    trained_embeddings = embeddings.eval()

from matplotlib import font_manager, rc 
# print(font_manager.get_fontconfig_fonts()) 
# font_name = font_manager.FontProperties(fname="C:\\Users\\bitcamp\\Downloads\\Nanumsquare_ac_TTF\\Nanumsquare_ac_TTF\\NanumSquare_acL.ttf").get_name() # matplot 에서 한글을 표시하기 위한 설정 
# matplotlib.rc('font', family='NanumGothic')
plt.rcParams['font.family'] = 'Malgun Gothic'

# visualization
for i, label in enumerate(word_list):
    x, y = trained_embeddings[i]
    plt.scatter(x,y)
    plt.annotate(label, xy=(x, y), xytext=(5,2),
                 textcoords="offset points", ha='right', va='bottom')

plt.show()