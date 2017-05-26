import sys
import nltk
import re
from nltk.corpus import wordnet as wn
import numpy as np
from threading import Thread
import random
import queue
import operator
import math
sentences = []
sen_weights = {}
node_weights = {}
sen_token_map = {}
index_vec = {}
context_vec = {}
sen_context = {}
vocabulary = []
token_map = {}
index_words = []
thresh = 0
l = []
stopset = set(nltk.corpus.stopwords.words('english'))
nouns = {x.name().split('.',1)[0] for x in wn.all_synsets('n')}
def getSentences(data):
    global l
    l = nltk.tokenize.sent_tokenize(data)
    for sen in l:
        sen = sen.lower()
        sen = re.sub('[^A-Za-z0-9 ]+','',sen)
        sentences.append(sen)
    return sentences
def generateVocabulary(sentences):
    #stopset = set(nltk.corpus.stopwords.words('english'))
    for sen in sentences:
        """sen = sen.lower()
        sen = re.sub('[^A-Za-z0-9 ]+','',sen)"""
        tokens = nltk.tokenize.word_tokenize(sen);
        tokens = [w for w in tokens if not w in stopset]
        for token in tokens:
            if token not in vocabulary:
                token_map[token] = 1
                vocabulary.append(token)
            else:
                token_map[token] = token_map[token] + 1
            if sen not in sen_token_map:
                sen_token_map[sen] = [token]
            else:
                sen_token_map[sen].append(token)
def getIndexWords():
    tagged = nltk.pos_tag(vocabulary)
    s = 0
    c = 0
    global index_words
    #print(tagged)
    for token in tagged:
        key = token[0]
        tag = token[1]
        if tag[0]=='N' or tag =="JJ":
            s = s + token_map[key]
            c = c + 1
    thresh = float(float(s)/c)

    for token in tagged:
        key = token[0]
        tag = token[1]
        if (tag[0]=='N' or tag =="JJ" )and token_map[key]>thresh:
            #print(key)
            #print(token_map[key])
            index_words.append(key)
    
    

def getNodeWeights():
    n = len(index_words)
    for sentence in sentences:
        """sentence = sentence.lower()
        sentence = re.sub('[^A-Za-z0-9 ]+','',sentence)"""
        q = np.zeros(shape=(n,1))
        i = 0
        while(i<n):
            count = 0
            if index_words[i] in sentence.split(" "):
                count = sentence.split(" ").count(index_words[i])
            q[i] = count*token_map[index_words[i]]
            i = i + 1
        sen_weights[sentence] = q


def cosine_similarity(X,Y):
    sum = np.sum(np.multiply(X,Y))
    norm_x = np.linalg.norm(X,ord=2)
    norm_y = np.linalg.norm(Y,ord=2)
    denominator = float(norm_x*norm_y)
    if denominator==0:
        sim = 0
        return sim
    sim = float(float(sum)/float(denominator))
    return sim

def getNodeSim():
    n = len(index_words)
    avg_vec = np.zeros(shape=(n,1))
    for key,values in sen_weights.items():
        vec = values
        avg_vec = np.add(avg_vec,vec)
    avg_vec = float(1/len(sentences))*avg_vec
    for key,values in sen_weights.items():
        vec = values
        node_weights[key] = cosine_similarity(avg_vec,vec)
    
def SemanticSim():
    window = 2
    size_of_index = len(vocabulary)/2
    size_of_index = int(size_of_index)
    for word in vocabulary:
        if word not in index_vec:
            #print(word)
            index = np.zeros(shape=(size_of_index,1))
            rand = random.sample(range(0,size_of_index-1),4)
            a = rand[0]
            b = rand[1]
            c = rand[2]
            d = rand[3]
            """print(a)
            print(b)
            print(c)
            print(d)"""
            index[a] = 1
            index[b] = 1
            index[c] = -1
            index[d] = -1
            #print(index)
            index_vec[word] = index
    average_vec = np.zeros(shape=(size_of_index,1))
    for sen in sentences:
        tokens = nltk.tokenize.word_tokenize(sen)
        tokens = [w for w in tokens if not w in stopset]
        i = 0
        n = len(tokens)
        while i<n:
            left_window = i - window
            right_window = i + window
            context = np.zeros(shape=(size_of_index,1))
            left_vec = np.zeros(shape=(size_of_index,1))
            right_vec = np.zeros(shape=(size_of_index,1))
            if left_window>=0 and right_window<=n-1:
                j = i-1
                while j>=left_window and j>=0:
                    left_vec = np.add(left_vec,index_vec[tokens[j]])
                    j = j - 1
                j= i+1
                #print(right_window)
                while j<=right_window and j<n:
                    #print(j)
                    right_vec = np.add(right_vec,index_vec[tokens[j]])
                    j = j + 1
                context = np.add(left_vec,right_vec)
            elif left_window>=0:
                j = i-1
                while j>=left_window and j>0:
                    left_vec = np.add(left_vec,index_vec[tokens[j]])
                    j = j - 1
                context = left_vec
            elif right_window<=n-1:
                j = i+1
                while j<=right_window and j<n:
                    right_vec = np.add(right_vec,index_vec[tokens[j]])
                    j = j + 1
                context = right_vec
            context_vec[tokens[i] + "" + sen + "" + str(i)] = context
            average_vec = np.add(average_vec,context)
            i = i + 1
    average_vec = float(1/float(len(context_vec)))*average_vec
    for sen in sentences:
        tokens = nltk.tokenize.word_tokenize(sen)
        tokens = [w for w in tokens if not w in stopset]
        i = 0
        n = len(tokens)
        sum = np.zeros(shape=(size_of_index,1))
        while i<n:
            x = np.subtract(context_vec[tokens[i] + "" + sen + "" + str(i)],average_vec)
            sum = np.add(sum,x)
            i = i + 1
        sum = float(1/(float(n)))*sum
        sen_context[sen] = sum           
                
def constructGraph(q):
    n = len(sentences)
    graph = np.zeros(shape=(n,n))
    i = 0
    while i<n:
        j = i
        while j<n:
            if i==j:
                graph[i][j] = 1
            else:
                X = sen_context[sentences[i]]
                Y = sen_context[sentences[j]]
                graph[i][j] = cosine_similarity(X,Y)
                graph[j][i] = graph[i][j]
            j = j + 1
        i = i + 1
    q.put(graph)

def PageRank(graph,q):
    i = 0
    PR = {}
    d = 0.85
    n = len(sentences)
    x = float(1/float(n))
    while i<n:
        PR[i] = x
        i = i + 1
    m = 0
    out = {}
    i = 0
    while i<n:
        x = 0
        j = 0
        while j<n:
            x = x + graph[i][j]
            j = j + 1
        out[i] = x
        i = i + 1        
    while m<1000:
        #print(m)
        flag = 0
        #print(PR)
        i = 0
        while i<n:
            j = 0
            sum_wij= 0
            while j<n:
                if i!=j:
                        k = 0
                        wji = graph[j][i]
                        #print(i)
                        #print(j)
                        #print(PR[j])
                        y = wji*float(float(PR[j])/float(out[j]))
                        prev = sum_wij
                        if(math.isinf(sum_wij)==False):
                            sum_wij = sum_wij + y
                        j = j + 1
                else:
                        j = j + 1
            val = PR[i]
            PR[i] = (1-d)+d*sum_wij
            if math.isinf(PR[i]):
                PR[i] = val
                flag = 1
                break
            i = i + 1
        if flag==1:
            break
        m = m + 1
    q.put(PR)
        
def generateSummary(PR,amt):
    PR = sorted(PR.items(), key=operator.itemgetter(1))
    PR.reverse()
    n = len(PR)
    amt = float(amt/100)
    summary = ""
    i = 0
    for key in PR:
        if float(float(i)/n)>= amt:
            break
        index = key[0]
        summary = summary + l[index]+" "
        i = i + 1
    print(summary)
    
if __name__ == '__main__':
    #print(sys.argv)
    data = sys.argv[1]
    #print(sys.argv[2])
    amt = sys.argv[2]
    if(amt==""):
        amt = 25
    #amt = 25
    #data = " reentrant lock is a synchronization primitive that may be acquired multiple times by the same thread. Internally, it uses the concepts of “owning thread” and “recursion level” in addition to the locked/unlocked state used by primitive locks. In the locked state, some thread owns the lock; in the unlocked state, no thread owns it."
    sentences = getSentences(data)
    generateVocabulary(sentences)
    getIndexWords()
    #print(index_words)
    getNodeWeights()
    #print(sen_weights)
    thread = Thread(target= getNodeSim,args=())
    thread_context = Thread(target=SemanticSim,args=())
    thread.start()
    thread_context.start()
    thread.join()
    thread_context.join()
    #print(node_weights)
    #print(sen_context)
    q = queue.Queue()
    thread_graph_construction= Thread(target=constructGraph,args=[q])
    thread_graph_construction.start()
    thread_graph_construction.join()
    graph = q.get()
    #print(graph)
    thread_pgrank = Thread(target=PageRank,args=[graph,q])
    thread_pgrank.start()
    thread_pgrank.join()
    PR = q.get()
    thread_summary = Thread(target=generateSummary,args=[PR,float(amt)])
    thread_summary.start()
    thread_summary.join()
    #print(l)"""
    


