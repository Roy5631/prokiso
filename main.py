import flask
from flask_cors import CORS
import base64

import gensim
from gensim.models import KeyedVectors
import MeCab
import numpy as np
from scipy import spatial

app = flask.Flask(__name__)
CORS(app)

@app.route('/predict',methods = ['POST', 'GET'])

def predict():
    
  if flask.request.method == 'POST':

  
    if flask.request.get_json().get('feature'):
        
        response = {
                "success": False
            }
        headers = {
            'Access-Control-Allow-Origin': '*'
        }
    
        feature = flask.request.get_json().get('feature')
    
      
    #   f = open("./entity_vector.model.bin", "rb")
        global model
        model = KeyedVectors.load_word2vec_format("./entity_vector.model.bin", binary=True)
        global mecab
        mecab=MeCab.Tagger("-Owakati")
        response['matomeru'] = mtmr(feature)
        return (flask.jsonify(response),200,headers)



#パスをわかつ
#入力は既存のパスリスト、出力は単語リスト
def ePath(ls):
    ans=[]
    for i in range(len(ls)):
        ans.append(ls[i].split("/"))
    return ans

#拡張子消す
#入力はファイルリスト、出力は拡張子なしファイルリスト
def eExt(ls):
    ans=[]
    for i in range(len(ls)):
        ans.append(ls[i].rsplit('.',1)[0])
    return ans


#入力はファイル名リスト、出力は分かち書きしたファイルリスト
def wakati(ls):
    ans=[]
    for i in range(len(ls)):
        wakati=mecab.parse(ls[i])
        ans.append(wakati.split())
    return ans

#書類名をベクトルにする(分かちした後の単語のベクトルを全てたす)
#入力は分かち書き済ファイルリスト、出力は書類のベクトルリスト
def toVec(l):
    nls=[]
    k=200
    for i in range(len(l)):
        n=len(l[i])
        if len(l[i])==1:
            try:
                nls.append(model[l[i][0]])
            except:
                nls.append(np.array([0 for i in range(k)]))
        else:
            try:
                t1=model[l[i][0]]
                for j in range(1,len(l[i])):
                    try:
                        t2=model[l[i][j]]
                        t1=t1+t2
                    except:
                        n-=1
            except:
                t1=np.array([0 for i in range(k)])
                n-=1
                for j in range(1,len(l[i])):
                    try:
                        t2=model[l[i][j]]
                        t1=t1+t2
                    except:
                        n-=1
            try:
                nls.append(t1/n)
            except:
                nls.append(t1)
    return nls


#一番近いものを見つける
#入力はベクトルリストと新たに分けるファイル名、出力は一番似たもののインデックス
def find(ls,fn):
    fn=mecab.parse(fn).split()
    n=len(fn)
    nls=0
    k=200
    if len(fn)==1:
        try:
            nls=model[fn[0]]
        except:
            nls=np.array([0 for i in range(k)])
    else:
        try:
            t1=model[fn[0]]
            for j in range(1,len(fn)):
                try:
                    t2=model[fn[j]]
                    t1=t1+t2
                except:
                    n-=1
        except:
            t1=np.array([0 for i in range(k)])
            n-=1
            for j in range(1,len(fn)):
                try:
                    t2=model[fn[j]]
                    t1=t1+t2
                except:
                    n-=1
        try:
            nls=t1/n
        except:
            nls=t1
    
    mx=1 - spatial.distance.cosine(nls, ls[0])
    mxn=0
    for i in range(1,len(ls)):
        h=1 - spatial.distance.cosine(nls, ls[i])
        if h>mx:
            mx=h
            mxn=i
    return mxn

#path化
def pt(pls,n,fn):
    ans=pls[n].rsplit('/',1)[0]
    return ans+"/"+fn


#ファイル分する
#入力はパスのリストと、振り分けるファイルの名前
def mtmr(lst):
    fn=lst[-1]
    #pathから/を消す
    ls=ePath(lst[:len(lst)-1])
    #拡張子を消す
    for i in range(len(ls)):
        ls[i]=eExt(ls[i])
    #分かち書き
    for i in range(len(ls)):
        tmp=wakati(ls[i])
        ls[i]=tmp[0]
        for j in range(1,len(tmp)):
            ls[i]+=tmp[j]
    #ベクトル化
    ls=toVec(ls)
    #見つける
    n=find(ls,fn)
    return pt(lst,n,fn)


if __name__ == '__main__':
    print('*flask starting server')
    app.run(host = '0.0.0.0', port = 50000,threaded=True)
    app.debug = True